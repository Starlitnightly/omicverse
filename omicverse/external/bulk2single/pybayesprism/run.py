# In R:
# load("your_dir/tutorial.gbm.rdata") 
# write.csv(bk.dat, file = "./data/bk_dat.csv")
# write.csv(sc.dat, file = "./data/sc_dat.csv")
# write(cell.type.labels, file = "./data/cell_type_labels.csv")
# write(cell.state.labels, file = "./data/cell_state_labels.csv")


import pandas as pd
import numpy as np
import pickle
import gzip
import importlib
import time

from pybayesprism import extract
from pybayesprism import gibbs
from pybayesprism import joint_post
from pybayesprism import optim
from pybayesprism import prism
from pybayesprism import process_input
from pybayesprism import references
from pybayesprism import theta_post
from pybayesprism import compare


def reload():
    # importlib.reload(compare)
    importlib.reload(extract)
    importlib.reload(gibbs)
    importlib.reload(joint_post)
    importlib.reload(optim)
    importlib.reload(prism)
    importlib.reload(process_input)
    # importlib.reload(references)
    importlib.reload(theta_post)



# bk_dat = pd.read_csv("./data/bk_dat.csv", index_col = 0).astype(np.int32)
# sc_dat = pd.read_csv("./data/sc_dat.csv", index_col = 0).astype(np.int32)
# cell_type_labels = list(pd.read_csv("./data/cell_type_labels.csv", header = None).iloc[:,0])
# cell_state_labels = list(pd.read_csv("./data/cell_state_labels.csv", header = None).iloc[:,0])

# with gzip.open("./data/d.pkl.gz", "wb") as f:
#     pickle.dump((bk_dat, sc_dat, cell_type_labels, cell_state_labels), f, protocol = 4) 

def a(comp = False):
    reload()
    global bk_dat, sc_dat_filtered_pc, cell_type_labels, cell_state_labels
    with gzip.open("./data/data.pkl.gz", "rb") as f:
        bk_dat, sc_dat, cell_type_labels, cell_state_labels = pickle.load(f)
    sc_dat_filtered = process_input.cleanup_genes(sc_dat, "count.matrix", "hs", \
                    ["Rb", "Mrp", "other_Rb", "chrM", "MALAT1", "chrX", "chrY"], 5)
    sc_dat_filtered_pc = process_input.select_gene_type(sc_dat_filtered, ["protein_coding"])
    compare.step1(bk_dat, sc_dat_filtered_pc, cell_type_labels, cell_state_labels) if comp else None
    


def b(comp = False):
    reload()
    global my_prism
    my_prism = prism.Prism.new(reference = sc_dat_filtered_pc, 
                               mixture = bk_dat, input_type = "count.matrix", 
                               cell_type_labels = cell_type_labels, 
                               cell_state_labels = cell_state_labels, 
                               key = "tumor", 
                               outlier_cut = 0.01, 
                               outlier_fraction = 0.1)
    compare.step2(my_prism) if comp else None


def c(comp = False):
    reload()
    global bp
    bp = my_prism.run(n_cores = 36, update_gibbs = False)
    compare.step3(bp) if comp else None


def d(comp = False):
    reload()
    global bp_res
    bp_res = bp.update_theta(opt_control = {'n.cores' : 20})
    compare.step4(bp_res) if comp else None

