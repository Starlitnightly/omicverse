import argparse
import pandas as pd
import pickle
import numpy as np

#from pybayesprism import prism
from .prism import Prism

parser = argparse.ArgumentParser(description = 'Python Implementation of BayesPrism')

parser.add_argument('-n', dest = 'ncores', help = 'CPU cores')

parser.add_argument('-x', dest = 'file_x', help = 'Bulk count matrix')
parser.add_argument('-x.type', dest = 'file_x_type', help = 'Bulk file type')

parser.add_argument('-ref', dest = 'file_ref', help = 'Reference count matrix')
parser.add_argument('-ref.file.type', dest = 'file_ref_type', help = 'Reference file type')
parser.add_argument('-ref.data.type', dest = 'data_type', help = 'Reference data type')

parser.add_argument('-species', dest = 'species', choices = ['hs', 'mm'], help = 'Species')
parser.add_argument('-out', dest = 'out_prefix', help = 'Output prefix')

parser.add_argument('-file.cell.type', dest = 'file_cell_type', help = 'file of cell type')


args = parser.parse_args()

# unpack args
ncores = args.ncores
file_x = args.file_x
file_x_type = args.file_x_type
file_ref = args.file_ref
file_ref_type = args.file_ref_type
data_type = args.data_type.lower()
outfile = args.out_prefix
file_cell_type = args.file_cell_type
species = args.species

print("CPU cores:", ncores)
print("Bulk count matrix:", file_x)
print("Bulk file type:", file_x_type)
print("Reference count matrix:", file_ref)
print("Reference file type:", file_ref_type)
print("Reference data type:", data_type)
print("File of cell type:", file_cell_type)
print("Output prefix:", outfile)
print("Species:", species)


if file_x_type not in ["csv", "tsv"]:
    print("! Error: the bulk matrix file supports csv, tsv, xls.")
    raise ValueError("Stop [err1]")

if file_ref_type not in ["csv", "tsv"]:
    print("! Error: the reference file supports csv, tsv, xls")
    raise ValueError("Stop [err2]")

print("1) -------- Load count matrix")

try:
    ## rowname: gene name
    ## colname: sample name
    if file_x_type == "tsv":
        x_sep = "\t"
    elif file_x_type == "csv":
        x_sep = ","
    df_x = pd.read_csv(file_x, sep=x_sep, index_col=0)

    ## rowname: gene name
    ## colname: sample name
    if file_ref_type == "tsv":
        ref_sep = "\t"
    elif file_ref_type == "csv":
        ref_sep = ","
    df_ref = pd.read_csv(file_ref, sep=ref_sep, index_col=0)

    #Cell ID
    #Cell Type(Tumor/Normal)
    #Cell subtype
    #Cell tumor flag(0,1:tumor)
    df_ct = pd.read_csv(file_cell_type, sep=",")
    assert df_ct.shape[1] == 4
except:
    print("Error: files could not be loaded.")

if not df_ref.index.isin(df_x.index).all():
    x_gene = sum(df_ref.index.isin(df_x.index))
    print("* Warning: your reference matrix don't have consistent", \
            "gene annotation with your bulk matrix.", x_gene, "/", \
            len(df_x.index), " genes can be matched.")


filter_chr = ["RB", "chrM", "chrX", "chrY"] if species in ["hs", "mm"] else []
    
cell_state_labels = df_ct.iloc[:,3]
cell_type_labels = df_ct.iloc[:,2]

if df_ref.shape[1] != df_ct.shape[0] \
    or not all(colname in list(df_ct.iloc[:, 0]) for colname in df_ref.columns) \
    or not all(cell_id in df_ref.columns for cell_id in df_ct.iloc[:, 0]):
    x_cell = sum(colname not in df_ct.iloc[:, 0] for colname in df_ref.columns)
    print("* Warning: your reference matrix doesn't have consistent cell ID \
    with your cell profile.", x_cell, "/", df_ct.shape[0], " cells can be \
    matched in your reference matrix.")


# Check cell IDs consistency
if data_type == "scrna":
    idx = [list(df_ct.iloc[:,0]).index(x) if x in list(df_ct.iloc[:,0]) else None for x in df_ref.columns]
elif data_type == "gep":
    idx = [list(df_ct.iloc[:,2]).index(x) if x in list(df_ct.iloc[:,2]) else None for x in df_ref.index]
idx = np.array([i is None for i in idx])

if np.sum(idx) > 0:
    if data_type == "scrna":
        print("* Warning:", np.sum(idx), "cells in your reference matrix \
            are not in your cell profile, e.g.:", df_ref.columns[idx])
    else:
        print("* Warning:", np.sum(idx), "cells in your reference matrix \
            are not in your cell profile, e.g.:", df_ref.index[idx])


if len(idx) - np.sum(idx) < 50:
    print("! Error: the remaining cell number is less than 50, stop.")
    raise ValueError("Stop [err5]")


df_ct = df_ct.loc[~idx]
print(df_ct.iloc[0:5,])

cell_state_labels = df_ct.iloc[:, 2].tolist()
cell_type_labels = df_ct.iloc[:, 1].tolist()

if data_type == "scrna":
    df_ref = df_ref.loc[:, df_ct.iloc[:, 0].tolist()]
if data_type == "gep":
    df_ref = df_ref.loc[df_ct.iloc[:, 2].tolist(), :]

print("* Warning:", df_ct.shape[0], "Cells in your reference matrix are used to proceed the next step.")

# Check if there are normal cells
if (df_ct.iloc[:, 3] == 0).sum() == 0:
    print("! Error: no normal cells (state==0) in your cell profile, stop.")
    raise ValueError("Stop [err7]")

key = None

if (df_ct.iloc[:, 3] == 1).sum() == 0:
    pass
else:
    k = df_ct[df_ct.iloc[:, 3] == 1].iloc[:, 1].unique()
    if len(k) > 1:
        key = 'tumor'
        df_ct.loc[df_ct.iloc[:, 3] == 1, df_ct.columns[1]] = key
        df_ct.loc[df_ct.iloc[:, 3] == 1, df_ct.columns[2]] = df_ct.loc[df_ct.iloc[:, 3] == 1, df_ct.columns[2]] + '_tumor'
    elif len(k) == 1:
        if k[0] == "tumor":
            key = k[0]
        else:
            key = k[0]
    

cell_type_labels = df_ct.iloc[:, 1]
cell_state_labels = df_ct.iloc[:, 2]

print("2) -------- Start bayesPrism")

df_ref.index = df_ref.index.str.upper()
# df_ref.columns = df_ref.columns.str.upper()

df_x.index = df_x.index.str.upper()
# df_x.columns = df_x.columns.str.upper()

######################################################
# ref = pd.read_csv("./testdata/[d]ref.csv", index_col=0)
# x = pd.read_csv("./testdata/[d]x.csv", index_col=0)
# ctl = list(pd.read_csv("./testdata/[d]ctl.csv").iloc[:,0])
# csl = list(pd.read_csv("./testdata/[d]csl.csv").iloc[:,0])

# assert np.all(df_ref.index == ref.index)
# assert np.all(df_ref.columns == ref.columns)
# assert np.allclose(df_ref.to_numpy(), ref.to_numpy())
# assert np.all(df_x.index == x.index)
# assert np.all(df_x.columns == x.columns)
# assert np.allclose(df_x.to_numpy(), x.to_numpy())
# assert np.all(ctl == cell_type_labels)
# assert np.all(ctl == cell_type_labels)

# import pdb;pdb.set_trace()
######################################################
if data_type == "scrna":
    if len(filter_chr) != 0:
        ref_dat_filtered = process_input.cleanup_genes(input = df_ref.T,
                                         input_type = "count.matrix",
                                         species = species,
                                         gene_group = ["other_Rb", "chrM", "chrX", "chrY", "Rb", "Mrp"],
                                         exp_cells = 5)
    else:
        ref_dat_filtered = df_ref.T

    my_prism = Prism.new(reference = ref_dat_filtered,
                         input_type = 'count.matrix',
                         cell_type_labels = cell_type_labels,
                         cell_state_labels = cell_state_labels,
                         key = key,
                         mixture = df_x.T,
                         outlier_cut = 0.01,
                         outlier_fraction = 0.1)
    bp = my_prism.run(n_cores = int(ncores))

if data_type == "gep":
    if len(filter_chr) != 0:
        ref_gep_filtered = cleanup_genes(input = df_ref.T,
                                         input_type = "GEP",
                                         species = species,
                                         gene_group = ["Rb", "Mrp", "other_Rb", "chrM", "MALAT1", "chrX", "chrY"],
                                         exp_cells = 5)
    else:
        ref_gep_filtered = df_ref.T

    my_prism = Prism.new(reference = ref_dat_filtered,
                         input_type = 'GEP',
                         cell_type_labels = cell_type_labels,
                         cell_state_labels = cell_state_labels,
                         key = key,
                         mixture = df_x.T,
                         outlier_cut = 0.01,
                         outlier_fraction = 0.1)
    bp = my_prism.run(n_cores = int(ncores))


with open("bp.pkl", "wb") as f:
    pickle.dump(bp, f)
