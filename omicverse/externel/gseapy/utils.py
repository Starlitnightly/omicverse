import errno
import logging
import os
import sys

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

DEFAULT_CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache/gseapy")


def unique(seq):
    """Remove duplicates from a list in Python while preserving order.

    :param seq: a python list object.
    :return: a list without duplicates while preserving order.

    """

    seen = set()
    seen_add = seen.add
    """
    The fastest way to sovle this problem is here
    Python is a dynamic language, and resolving seen.add each iteration
    is more costly than resolving a local variable. seen.add could have
    changed between iterations, and the runtime isn't smart enough to rule
    that out. To play it safe, it has to check the object each time.
    """

    return [x for x in seq if x not in seen and not seen_add(x)]


def mkdirs(outdir):
    """create new directory"""
    try:
        os.makedirs(outdir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
        pass


def log_init(name, log_level=logging.INFO, filename=None):
    """logging

    :param name: logger name
    :log_level: refer to logging module
    :filename: if given a filename, write log to a file (only works in commandline)

    """
    # inside python console enviroment. only need one root logger
    if hasattr(sys, "ps1"):
        name = "gseapy"
    logger = logging.getLogger(name)
    # clear old root logger handlers
    if logger.hasHandlers():
        log_close(logger)
        # logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't find root logger
    # init a root logger
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="%(asctime)s %(name)s::[%(levelname)-8s] %(message)s",
    #     # filename=filename,
    #     # filemode="w",
    # )
    # # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(log_level)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add handlers
    logger.addHandler(console)
    # only write log file when in command line
    if (not hasattr(sys, "ps1")) and filename:
        fhandler = logging.FileHandler(filename=filename, mode="w")
        fhandler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s::[%(levelname)-8s] %(message)s"
        )
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
    # logger.handlers.clear()
    # logger.removeHandler(fh)
    return logger


def log_close(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def retry(num=5):
    """ "retry connection.

    define max tries num
    if the backoff_factor is 0.1, then sleep() will sleep for
    [0.1s, 0.2s, 0.4s, ...] between retries.
    It will also force a retry if the status code returned is 500, 502, 503 or 504.

    """
    s = requests.Session()
    retries = Retry(total=num, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retries))

    return s


# CONSTANT
DEFAULT_LIBRARY = [
    "GO_Biological_Process_2013",
    "GO_Biological_Process_2015",
    "GO_Cellular_Component_2013",
    "GO_Cellular_Component_2015",
    "GO_Molecular_Function_2013",
    "GO_Molecular_Function_2015",
    "GeneSigDB",
    "HumanCyc_2015",
    "Human_Gene_Atlas",
    "Human_Phenotype_Ontology",
    "Humancyc_2016",
    "KEGG_2013",
    "KEGG_2015",
    "KEGG_2016",
    "MGI_Mammalian_Phenotype_2013",
    "MGI_Mammalian_Phenotype_Level_3",
    "MGI_Mammalian_Phenotype_Level_4",
    "MSigDB_Computational",
    "MSigDB_Oncogenic_Signatures",
    "Mouse_Gene_Atlas",
    "Panther_2015",
    "Panther_2016",
    "Reactome_2013",
    "Reactome_2015",
    "Reactome_2016",
    "WikiPathways_2013",
    "WikiPathways_2015",
    "WikiPathways_2016",
]