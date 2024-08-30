
import os, errno, logging
import requests
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from os.path import expanduser

DEFAULT_CACHE_PATH = os.path.join(expanduser("~"), ".cache/gseapy")

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

    try:
        os.makedirs(outdir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
        pass

class GSLogger(object):
    # singleton
    __instance = None
    def __new__(cls, outlog, log_level=logging.INFO):
        """Singleton
        __new__ handles object creation and __init__ handles object initialization.
        __new__ accepts cls as it's first parameter and __init__ accepts self.
        __new__ excute first, then __init__
        """
        if GSLogger.__instance is None:
            GSLogger.__instance = object.__new__(cls)
            logger = log_init(outlog, log_level)
            GSLogger.__instance.logger = logger
        return GSLogger.__instance


def log_init(outlog, log_level=logging.INFO):
    """logging start"""
    # clear old root logger handlers
    logging.getLogger("gseapy").handlers = []
    # init a root logger
    logging.basicConfig(level    = logging.DEBUG,
                        format   = 'LINE %(lineno)-4d: %(asctime)s [%(levelname)-8s] %(message)s',
                        filename = outlog,
                        filemode = 'w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(log_level)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add handlers
    logging.getLogger("gseapy").addHandler(console)
    logger = logging.getLogger("gseapy")
    # logger.setLevel(log_level)
    # logger.handlers.clear()
    # logger.removeHandler(fh)
    return logger

def retry(num=5):
    """"retry connection.
    
        define max tries num
        if the backoff_factor is 0.1, then sleep() will sleep for
        [0.1s, 0.2s, 0.4s, ...] between retries.
        It will also force a retry if the status code returned is 500, 502, 503 or 504.    
    
    """
    s = requests.Session()
    retries = Retry(total=num, backoff_factor=0.1,
                    status_forcelist=[500, 502, 503, 504])
    s.mount('http://', HTTPAdapter(max_retries=retries))

    return s

# CONSTANT
DEFAULT_LIBRARY=['GO_Biological_Process_2013',
				 'GO_Biological_Process_2015',
				 'GO_Cellular_Component_2013',
				 'GO_Cellular_Component_2015',
				 'GO_Molecular_Function_2013',
				 'GO_Molecular_Function_2015',
				 'GeneSigDB',
				 'HumanCyc_2015',
				 'Human_Gene_Atlas',
				 'Human_Phenotype_Ontology',
				 'Humancyc_2016',
				 'KEGG_2013',
				 'KEGG_2015',
				 'KEGG_2016',
				 'MGI_Mammalian_Phenotype_2013',
				 'MGI_Mammalian_Phenotype_Level_3',
				 'MGI_Mammalian_Phenotype_Level_4',
				 'MSigDB_Computational',
				 'MSigDB_Oncogenic_Signatures',
				 'Mouse_Gene_Atlas',
				 'Panther_2015',
				 'Panther_2016',
				 'Reactome_2013',
				 'Reactome_2015',
				 'Reactome_2016',
				 'WikiPathways_2013',
				 'WikiPathways_2015',
				 'WikiPathways_2016']