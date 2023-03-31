# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import Pyomic
# -- Project information -----------------------------------------------------

project = 'Pyomic'
copyright = '2022, starlitnightly'
author = 'starlitnightly'

# The full version, including alpha/beta/rc tags
release = '1.1.7'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'recommonmark',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'nbsphinx',
#    'sphinx.ext.autoclass',
]
import sphinx_autosummary_accessors
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', sphinx_autosummary_accessors.templates_path]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']

# source_suffix = '.rst'
from recommonmark.parser import CommonMarkParser
source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']

import inspect
import sphinx_autodoc_typehints

master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_show_sourcelink = True
set_type_checking_flag = True
typehints_fully_qualified = True
napoleon_use_rtype = False
autosummary_generate = True
autosummary_generate_overwrite = True
autodoc_preserve_defaults = True
autodoc_inherit_docstrings = True
autodoc_default_options = {
    'autosummary': True
}

intersphinx_mapping = dict(
    python=('https://docs.python.org/3/', None),
    numpy=('https://numpy.org/doc/stable/', None),
    scipy=('https://docs.scipy.org/doc/scipy/', None),
    pandas=('https://pandas.pydata.org/pandas-docs/stable/', None),
    sklearn=('https://scikit-learn.org/stable/', None),
    matplotlib=('https://matplotlib.org/stable/', None),
    seaborn=('https://seaborn.pydata.org/', None),
    networkx=('https://networkx.org/documentation/stable/', None),
    anndata=('https://anndata.readthedocs.io/en/stable/', None),
    scanpy=('https://scanpy.readthedocs.io/en/stable/', None),
    torch=('https://pytorch.org/docs/stable/', None),
    ignite=('https://pytorch.org/ignite/', None),
    plotly=('https://plotly.com/python-api-reference/', None)
)

qualname_overrides = {
    'anndata._core.anndata.AnnData': 'anndata.AnnData',
    'matplotlib.axes._axes.Axes': 'matplotlib.axes.Axes',
    'networkx.classes.graph.Graph': 'networkx.Graph',
    'networkx.classes.digraph.DiGraph': 'networkx.DiGraph',
    'networkx.classes.multigraph.MultiGraph': 'networkx.MultiGraph',
    'networkx.classes.multidigraph.MultiDiGraph': 'networkx.MultiDiGraph',
    'numpy.random.mtrand.RandomState': 'numpy.random.RandomState',
    'pandas.core.frame.DataFrame': 'pandas.DataFrame',
    'scipy.sparse.base.spmatrix': 'scipy.sparse.spmatrix',
    'seaborn.axisgrid.JointGrid': 'seaborn.JointGrid',
    'torch.device': 'torch.torch.device',
    'torch.nn.modules.module.Module': 'torch.nn.Module'
}

'''
fa_orig = sphinx_autodoc_typehints.format_annotation
def format_annotation(annotation, fully_qualified=True):  # pylint: disable=unused-argument
    r"""
    Adapted from https://github.com/agronholm/sphinx-autodoc-typehints/issues/38#issuecomment-448517805
    """
    if inspect.isclass(annotation):
        full_name = f'{annotation.__module__}.{annotation.__qualname__}'
        override = qualname_overrides.get(full_name)
        if override is not None:
            return f':py:class:`~{override}`'
    return fa_orig(annotation)
sphinx_autodoc_typehints.format_annotation = format_annotation
'''