# -*- coding: utf-8 -*-

import sys, logging, json, os
import requests
import pandas as pd
import xml.etree.ElementTree as ET 
from io import StringIO
from numpy import in1d, isin
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from .biomart import Biomart
from .utils import unique, DEFAULT_LIBRARY, DEFAULT_CACHE_PATH, mkdirs
from collections.abc import Iterable


def gsea_cls_parser(cls):
    """Extract class(phenotype) name from .cls file.

    :param cls: the a class list instance or .cls file which is identical to GSEA input .
    :return: phenotype name and a list of class vector.
    """

    if not isinstance(cls, str) and isinstance(cls, Iterable) :
        classes = list(cls)
        sample_name= unique(classes)
    elif isinstance(cls, str) :
        with open(cls) as c:
            file = c.readlines()
        classes = file[2].strip('\n').split(" ")
        sample_name = file[1].lstrip("# ").strip('\n').split(" ")
        
        tmp = set(sample_name) & set(classes) 
        if len(tmp) < 2: # classes and sample_name are different
            s1 = classes[0]
            for i, c in enumerate(classes):
                if c == s1:
                    classes[i] = sample_name[0]
                else:
                    classes[i] = sample_name[1]
    else:
        raise Exception('Error parsing sample name!')
    
    if len(sample_name) != 2:
            raise Exception("Input groups have to be 2!")
    
    return sample_name[0], sample_name[1], classes

def gsea_edb_parser(results_path):
    """Parse results.edb file stored under **edb** file folder.

    :param results_path: the .results file located inside edb folder.
    :return: 
        a dict contains enrichment_term, hit_index,nes, pval, fdr.
    """
    
    xtree = ET.parse(results_path)
    xroot = xtree.getroot() 
    res = {}
    # dict_keys(['RANKED_LIST', 'GENESET', 'FWER', 'ES_PROFILE',
    # 'HIT_INDICES', 'ES', 'NES', 'TEMPLATE', 'RND_ES', 'RANK_SCORE_AT_ES',
    # 'NP', 'RANK_AT_ES', 'FDR'])
    for node in xroot.findall('DTG'):
        enrich_term = node.attrib.get('GENESET').split("#")[1]
        es_profile = node.attrib.get('ES_PROFILE').split(" ")
        # rank_es = term.get('RND_ES').split(" ")
        hit_ind = node.attrib.get('HIT_INDICES').split(" ")
        es_profile = [float(i) for i in es_profile ]
        hit_ind = [float(i) for i in hit_ind ]
        # rank_es = [float(i) for i in rank_es ]
        nes = node.attrib.get('NES')
        pval = node.attrib.get('NP')
        fdr =  node.attrib.get('FDR')
        # fwer = node.attrib.get('FWER')
        logging.debug("Enriched Gene set is: "+ enrich_term)
        res[enrich_term] =[hit_ind, nes, pval, fdr]
    return res


def gsea_gmt_parser(gmt, organism='Human', min_size = 3, max_size = 1000, gene_list=None):
    """Parse gene_sets.gmt(gene set database) file or download from enrichr server.

    :param str gmt: the gene_sets.gmt file or an enrichr library name.
                    checkout full enrichr library name here: https://maayanlab.cloud/Enrichr/#libraries
    
    :param str organism: choose one from { 'Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm' }.
                         This arugment has not effect if input is a `.gmt` file.

    :param min_size: Minimum allowed number of genes from gene set also the data set. Default: 3.
    :param max_size: Maximum allowed number of genes from gene set also the data set. Default: 1000.

    :param gene_list: Used for filtering gene set. Only used this argument for :func:`gsea` method.

    :return: Return a new filtered gene set database dictionary.

    **DO NOT** filter gene sets, when use :func:`replot`. Because ``GSEA`` Desktop have already
    done this for you.

    """

    if gmt.lower().endswith(".gmt"):
        logging.info("User Defined gene sets is given.......continue..........")
        with open(gmt) as genesets:
             genesets_dict = { line.strip().split("\t")[0]: line.strip().split("\t")[2:]
                              for line in genesets.readlines()}
    else:
        logging.info("Downloading and generating Enrichr library gene sets...")

        names = DEFAULT_LIBRARY
        if gmt not in DEFAULT_LIBRARY:
            names = get_library_name(organism=organism)

        if gmt in names:
            """
            define max tries num
            if the backoff_factor is 0.1, then sleep() will sleep for
            [0.1s, 0.2s, 0.4s, ...] between retries.
            It will also force a retry if the status code returned is 500, 502, 503 or 504.
            """
            s = requests.Session()
            retries = Retry(total=5, backoff_factor=0.1,
                            status_forcelist=[ 500, 502, 503, 504 ])
            s.mount('http://', HTTPAdapter(max_retries=retries))
            # query string
            ENRICHR_URL = 'http://amp.pharm.mssm.edu/Enrichr/geneSetLibrary'
            query_string = '?mode=text&libraryName=%s'
            # get
            response = s.get( ENRICHR_URL + query_string % gmt, timeout=None, verify=False)
        else:
            raise Exception("gene_set files(.gmt) not found")
            
        if not response.ok:
            raise Exception('Error fetching enrichment results, check internet connection first.')

        genesets_dict = { line.strip().split("\t")[0]:
                          list(map(lambda x: x.split(",")[0], line.strip().split("\t")[2:]))
                          for line in response.iter_lines(chunk_size=1024, decode_unicode='utf-8')}



    # filtering dict
    if sys.version_info[0] >= 3 :
        genesets_filter =  {k: v for k, v in genesets_dict.items() if len(v) >= min_size and len(v) <= max_size}
    elif sys.version_info[0] == 2:
        genesets_filter =  {k: v for k, v in genesets_dict.iteritems() if len(v) >= min_size and len(v) <= max_size}
    else:
        raise Exception("System failure. Please Provide correct input files")
    if gene_list is not None:
        subsets = sorted(genesets_filter.keys())
        for subset in subsets:
            tag_indicator = isin(gene_list, genesets_filter.get(subset), assume_unique=True)
            tag_len = sum(tag_indicator)
            if tag_len <= min_size or tag_len >= max_size:
                del genesets_filter[subset]
            else:
                continue
    # some_dict = {key: value for key, value in some_dict.items() if value != value_to_remove}
    # use np.intersect1d() may be faster???
    filsets_num = len(genesets_dict) - len(genesets_filter)
    logging.info("%04d gene_sets have been filtered out when max_size=%s and min_size=%s"%(filsets_num, max_size, min_size))

    if filsets_num == len(genesets_dict):
        raise Exception("No gene sets passed throught filtering condition!!!, try new paramters again!\n" +\
                         "Note: Gene names for gseapy is case sensitive." )
    else:
        return genesets_filter

def get_library_name(organism='Human'):
    """return enrichr active enrichr library name. 
    see also: https://maayanlab.cloud/modEnrichr/

    :param str database: Select one from { 'Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm' } 
    :return: a list of enrichr libraries from selected database
    
    """
    default = [ 'human','mouse','hs', 'mm',
                'homo sapiens', 'mus musculus',
                'h. sapiens', 'm. musculus']
    _organisms = {
                'Fly': ['fly', 'd. melanogaster', 'drosophila melanogaster'],
                'Yeast': ['yeast', 's. cerevisiae', 'saccharomyces cerevisiae'],
                'Worm': ['worm', 'c. elegans', 'caenorhabditis elegans', 'nematode'],
                'Fish': ['fish', 'd. rerio', 'danio rerio', 'zebrafish']
                }
    ENRICHR_URL = 'http://maayanlab.cloud'
    database=''
    if organism.lower() in default:
        database = 'Enrichr' 
    else:
        for k, v in _organisms.items():
            if organism.lower() in v :
                database = k+'Enrichr'
                break

    if not database.endswith('Enrichr'):
        raise LookupError("""No supported database. Please input one of these:
                            ('Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm') """)
    # make a get request to get the gmt names and meta data from Enrichr
    # old code
    # response = requests.get('http://amp.pharm.mssm.edu/Enrichr/geneSetLibrary?mode=meta')
    # gmt_data = response.json()
    # # generate list of lib names
    # libs = []
    # # get library names
    # for inst_gmt in gmt_data['libraries']:
    #     # only include active gmts
    #     if inst_gmt['isActive'] == True:
    #         libs.append(inst_gmt['libraryName'])
    lib_url='%s/%s/datasetStatistics'%(ENRICHR_URL, database)
    response = requests.get(lib_url, verify=True)
    if not response.ok:
        raise Exception("Error getting the Enrichr libraries")
    libs_json = json.loads(response.text)
    libs = [lib['libraryName'] for lib in libs_json['statistics']]

    return sorted(libs)


