# Developer guild

!!! Note
    To better understand the following guide, you may check out our [publication](https://doi.org/10.1101/2023.06.06.543913) first to learn about the general idea.

Below we describe main components of the framework, and how to extend the existing implementations.

## Framework

The omicverse code is stored in the [omicverse folder](https://github.com/Starlitnightly/omicverse/tree/master/omicverse) in the github repository, with the `__init__.py` file taking care of the import of the library functions.

A omicverse framework is primarily composed of 5 components.

- `utils`: Functions, including data, plotting, etc.
- `pp`: preprocess, including quantity control, normalize, etc.
- `bulk`: to analysis the bulk omic-seq like RNA-seq or Proper-seq.
- `single`: to analysis the single cell omic-seq like scRNA-seq or scATAC-seq
- `space`: to analysis the spatial RNA-seq
- `bulk2single`: to integrate the bulk RNA-seq and single cell RNA-seq
- `externel`: more related module included RNA-seq avoided installation and confliction

The `__init__.py` file is responsible for importing function entries within each folder, and all function functions use a file starting with `_*.py` for function writing.


## For Developer

### Externel module

In most cases, we realize that writing a module function is difficult. Therefore, we introduced the `external` module. We can directly clone the entire package from GitHub and then move the entire folder to the `external` folder. During this process, we need to pay attention to whether the License allows it and whether there is a conflict with OmicVerse's GPL license. Subsequently, we need to modify the `import` content. We need to change the packages that are not dependencies of OmicVerse from top-level imports to function-level imports.

````shell
.
├── omicverse               
├───── externel
├──────── STT
├─────────── __init__.py 
├─────────── pl
├─────────── tl
````

All imports need to ensure that there are no conflicts.

This is an error because this package is not included in the default requirements.txt of OmicVerse.

```python

import dgl 

def calculate():
    dgl.run()
    pass

```

The correct import is

```python

def calculate():
    import dgl 
    dgl.run()
    pass

```

We recommend using `try` to detect import errors, which can then guide the user to the correct installation page.


```python

def calculate():
    try:
        import dgl 
    except ImportError:
        raise ImportError(
            'Please install the dgl from https://www.dgl.ai/pages/start.html'
        )
    dgl.run()
    pass

```

### Main module

If you want to provide pull request for omicverse, you need to be clear about which module the functionality you are developing is subordinate to, e.g. `TOSICA` belongs to the algorithms of the single-cell domain, i.e., you need to add the `_tosica.py` file inside the `single` folder of `omicverse` and `_init__.py` inside the `from . _tosica import pyTOSICA` to make the omicverse add the new functionality

````shell
.
├── omicverse               
├───── single
├──────── __init__.py 
├──────── _tosica.py 
````

All functions require parameter descriptions in the following format:

```python

def preprocess(adata:anndata.AnnData, mode:str='scanpy', target_sum:int=50*1e4, n_HVGs:int=2000,
    organism:str='human', no_cc:bool=False)->anndata.AnnData:
    """
    Preprocesses the AnnData object adata using either a scanpy or a pearson residuals workflow for size normalization
    and highly variable genes (HVGs) selection, and calculates signature scores if necessary. 

    Arguments:
        adata: The data matrix.
        mode: The mode for size normalization and HVGs selection. It can be either 'scanpy' or 'pearson'. If 'scanpy', performs size normalization using scanpy's normalize_total() function and selects HVGs 
            using pegasus' highly_variable_features() function with batch correction. If 'pearson', selects HVGs 
            using scanpy's experimental.pp.highly_variable_genes() function with pearson residuals method and performs 
            size normalization using scanpy's experimental.pp.normalize_pearson_residuals() function. 
        target_sum: The target total count after normalization.
        n_HVGs: the number of HVGs to select.
        organism: The organism of the data. It can be either 'human' or 'mouse'. 
        no_cc: Whether to remove cc-correlated genes from HVGs.

    Returns:
        adata: The preprocessed data matrix. 
    """

```

## Pull request

1. You need to `fork` omicverse at first, and git clone your fork from your repository.
2. When you updated the related function development, open a pull request and waited reviewed and merged.

