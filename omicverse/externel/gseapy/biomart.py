import logging
import os
from collections.abc import Iterable
from io import StringIO
from typing import Any, AnyStr, Dict, Iterable, List, Optional, Set, Tuple, Union
from xml.etree import cElementTree as ET

import pandas as pd
import requests

from .utils import log_init, mkdirs, retry


class Biomart:
    """query from BioMart"""

    def __init__(self, host: str = "www.ensembl.org", verbose: bool = False):
        """simple API to BioMart services.

        How to query validated dataset, attributes, filters.
        Example::
        >>> from gseapy import Biomart
        >>> bm = Biomart(verbose=False, host="ensembl.org")
        >>> ## view validated marts
        >>> marts = bm.get_marts()
        >>> ## view validated dataset
        >>> datasets = bm.get_datasets(mart='ENSEMBL_MART_ENSEMBL')
        >>> ## view validated attributes
        >>> attrs = bm.get_attributes(dataset='hsapiens_gene_ensembl')
        >>> ## view validated filters
        >>> filters = bm.get_filters(dataset='hsapiens_gene_ensembl')
        >>> ## query results
        >>> queries = ['ENSG00000125285','ENSG00000182968'] # a python list
        >>> results = bm.query(dataset='hsapiens_gene_ensembl',
                            attributes=['entrezgene_id', 'go_id'],
                            filters={'ensembl_gene_id': queries}
                            )
        """
        self._id = str(id(self))
        self._logger = log_init(
            name="Biomart" + self._id,
            log_level=logging.INFO if verbose else logging.WARNING,
            filename=None,
        )
        self._set_host(host)

        self.attributes_xml = []
        self.filters_xml = []
        self.dataset_xml = ""

        params = {
            "host": self.host,
            "version": "1.0",
            "virtualSchemaName": "default",
            "formatter": "TSV",
            "header": 0,
            "uniqueRows": 1,
            "configVersion": "0.6",
            "completionStamp": 0,
        }

        self.header = (
            """https://%(host)s/biomart/martservice?query="""
            """<?xml version="%(version)s" encoding="UTF-8"?>"""
            """<!DOCTYPE Query>"""
            """<Query virtualSchemaName="%(virtualSchemaName)s" formatter="%(formatter)s" """
            """header="%(header)s" uniqueRows="%(uniqueRows)s" count="" """
            """datasetConfigVersion="%(configVersion)s" completionStamp="%(completionStamp)s">"""
        )

        self.header = self.header % params
        self.footer = "</Dataset></Query>"
        self.reset()

        # get supported marts
        self._marts = self.get_marts()["Mart"].to_list()

    def __del__(self):
        handlers = self._logger.handlers[:]
        for handler in handlers:
            handler.close()  # close file
            self._logger.removeHandler(handler)

    def _set_host(self, host: str):
        """set host"""

        hosts = ["www.ensembl.org", "asia.ensembl.org", "useast.ensembl.org"]
        hosts.insert(0, host)
        secure = ""

        # if self._secure:
        #     secure = "s"
        # if host not work, select next
        i = 0
        while i < len(hosts):
            url = "http{}://{}/biomart/martservice".format(secure, hosts[i])
            request = requests.head(url)
            if request.status_code in [200]:
                self.host = hosts[i]
                break
            else:
                self._logger.warning(
                    "host {} is not reachable, will try {} ".format(
                        hosts[i], hosts[i % len(hosts)]
                    )
                )
            i += 1
        if i == len(hosts):
            raise ValueError(
                "host is not reachable. Please check your input or try again later."
            )

    def add_filter(self, name: str, value: Iterable[str]):
        """
        key: filter names
        value: Iterable[str]
        """
        if isinstance(value, Iterable):
            value = ",".join([str(v) for v in list(value)])
        _filter = ""
        if name.lower().startswith("with"):
            _filter = """<Filter name="%s" excluded="%s"/>""" % (name, value)
        else:
            _filter = """<Filter name="%s" value="%s"/>""" % (name, value)
        self.filters_xml.append(_filter)

    def add_attribute(self, attribute: str):
        _attr = """<Attribute name="%s"/>""" % attribute
        self.attributes_xml.append(_attr)

    def add_dataset(self, dataset: str):
        self.dataset_xml = """<Dataset name="%s" interface="default" >""" % dataset

    def reset(self):
        self.attributes_xml = []
        self.filters_xml = []
        self.dataset_xml = ""

    def get_xml(self):
        xml = self.header
        xml += self.dataset_xml
        for line in self.filters_xml:
            xml += line
        for line in self.attributes_xml:
            xml += line
        xml += self.footer
        return xml

    def get_marts(self):
        """Get available marts and their names."""
        url = "https://{host}/biomart/martservice?type=registry&requestid=gseapy{i}".format(
            host=self.host, i=self._id
        )
        resp = requests.get(url)
        if resp.ok:
            # marts = pd.read_xml(resp.text)
            marts = [e.attrib for e in ET.XML(resp.text)]
            marts = pd.DataFrame(marts)
            marts = marts.loc[:, ["database", "displayName", "name"]]
            marts.columns = ["Version", "DisplayName", "Mart"]
            return marts.loc[:, ["Mart", "Version"]]

        return resp.text

    def get_datasets(self, mart: str = "ENSEMBL_MART_ENSEMBL"):
        """Get available datasets from mart you've selected"""
        if mart not in self._marts:
            raise ValueError(
                "Provided mart name (%s) is not valid. see 'names' attribute" % mart
            )

        url = "https://{host}/biomart/martservice?type=datasets&mart={mart}".format(
            host=self.host, mart=mart
        )
        resp = requests.get(url)
        if resp.ok:
            if resp.text.startswith("Problem"):
                return resp.text
            datasets = [
                record.split("\t")
                for record in resp.text.split("\n")
                if len(record) > 1
            ]
            datasets = pd.DataFrame(datasets).iloc[:, 1:3]
            datasets.columns = ["Dataset", "Description"]
            return datasets
        return resp.text

    def get_attributes(self, dataset: str = "hsapiens_gene_ensembl"):
        """Get available attritbutes from dataset you've selected"""
        # assert dataset in

        url = "https://{host}/biomart/martservice?type=attributes&dataset={dataset}".format(
            host=self.host, dataset=dataset
        )
        resp = requests.get(url)
        if resp.ok:
            attributes = [text.split("\t") for text in resp.text.strip().split("\n")]
            attributes = pd.DataFrame(attributes).iloc[:, :3]
            attributes.columns = ["Attribute", "Description", "Additional"]
            return attributes
        return resp.text

    def get_filters(self, dataset: str = "hsapiens_gene_ensembl"):
        """Get available filters from dataset you've selected"""
        # filters = super(Biomart, self).filters(dataset)
        # if dataset not in [x for k in self.valid_attributes.keys() for x in self.valid_attributes[k]]:
        #     raise ValueError("provided dataset (%s) is not found. see valid_attributes" % dataset)
        url = (
            "https://{host}/biomart/martservice?type=filters&dataset={dataset}".format(
                host=self.host, dataset=dataset
            )
        )

        resp = requests.get(url)
        if resp.ok:
            if str(resp.text).startswith("Query ERROR"):
                return resp.text
            filters = [text.split("\t") for text in resp.text.strip().split("\n")]
            filters = pd.DataFrame(filters).iloc[:, [0, 1, 3, 5]]
            filters.columns = ["Filter", "Description", "Additional", "InputType"]
            return filters
        return resp.text

    def query(
        self,
        dataset: str = "hsapiens_gene_ensembl",
        attributes: Optional[List[str]] = [],
        filters: Optional[Dict[str, Iterable[str]]] = {},
        filename: Optional[str] = None,
    ):
        """mapping ids using BioMart.

        :param dataset: str, default: 'hsapiens_gene_ensembl'
        :param attributes: str, list, tuple
        :param filters: dict, {'filter name': list(filter value)}
        :param host: www.ensembl.org, asia.ensembl.org, useast.ensembl.org
        :return: a dataframe contains all attributes you selected.

        Example:

            >>> queries = {'ensembl_gene_id': ['ENSG00000125285','ENSG00000182968'] } # need to be a python dict
            >>> results = bm.query(dataset='hsapiens_gene_ensembl',
                                   attributes=['ensembl_gene_id', 'external_gene_name', 'entrezgene_id', 'go_id'],
                                   filters=queries)
        """
        if not attributes:
            attributes = [
                "ensembl_gene_id",
                "external_gene_name",
                "entrezgene_id",
                "go_id",
            ]
        if isinstance(attributes, str):
            attributes = attributes.split(",")

        if not isinstance(filters, dict):
            raise ValueError("filters only accept a dict object")

        df = self.query_simple(
            dataset=dataset, filters=filters, attributes=attributes, filename=None
        )
        if df is None:
            return
        elif isinstance(df, str):
            print(df)
            return df

        if "entrezgene_id" in df.columns:
            df["entrezgene_id"] = df["entrezgene_id"].astype(pd.Int32Dtype())

        self.results = df
        # save file to cache path.
        if filename is not None:
            mkdirs(os.path.dirname(filename))
            df.to_csv(filename, sep="\t", index=False)

        return df

    def query_simple(
        self,
        dataset: str = "hsapiens_gene_ensembl",
        attributes: List[str] = [],
        filters: Dict[str, Iterable[str]] = {},
        filename: Optional[str] = None,
    ):
        """
        This function is a simple version of BioMart REST API.
        same parameter to query().

        However, you could get cross page of mapping. such as Mouse 2 human gene names

        **Note**: it will take a couple of minutes to get the results.
        A xml template for querying biomart. (see https://gist.github.com/keithshep/7776579)

        Example::
            >>> from gseapy import Biomart
            >>> bm = Biomart()
            >>> results = bm.query_simple(dataset='mmusculus_gene_ensembl',
                                          attributes=['ensembl_gene_id',
                                                      'external_gene_name',
                                                      'hsapiens_homolog_associated_gene_name',
                                                      'hsapiens_homolog_ensembl_gene'])
        """
        self.reset()
        self.add_dataset(dataset)
        self._logger.debug("Add attributes")
        for at in attributes:
            self.add_attribute(at)
        self._logger.debug("Add filters")
        for n, v in filters.items():
            self.add_filter(n, v)
        self._logger.debug("Build xml")
        self._xml = self.get_xml()
        s = retry(num=5)
        response = s.get(self._xml)
        if response.ok:
            if str(response.text).startswith("Query ERROR"):
                self._logger.error(response.text)
                return
            df = pd.read_table(StringIO(response.text), header=None, names=attributes)
            if filename is not None:
                df.to_csv(filename, sep="\t", index=False)
            self.results = df
            return df
        return response.text