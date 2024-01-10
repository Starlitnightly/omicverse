import pandas as pd

#try:
#    import importlib.resources as res
#    package = res.files('CEFCON')
#    TFs_human = pd.read_csv(package / 'resources/hs_hgnc_tfs_lambert2018.txt', names=['gene_symbol'])
#    TFs_mouse = pd.read_csv(package / 'resources/mm_mgi_tfs.txt', names=['gene_symbol'])
#    TFs_human_animaltfdb = pd.read_csv(package / 'resources/hs_hgnc_tfs_animaltfdb4.txt', names=['gene_symbol'])
#    TFs_mouse_animaltfdb = pd.read_csv(package / 'resources/mm_mgi_tfs_animaltfdb4.txt', names=['gene_symbol'])

#except ImportError:  # ImportError occurs when using Python < 3.9
import pkg_resources
package = 'omicverse'
TFs_human_path = pkg_resources.resource_filename(package, 'CEFCON/resources/hs_hgnc_tfs_lambert2018.txt')
TFs_mouse_path = pkg_resources.resource_filename(package, 'CEFCON/resources/mm_mgi_tfs.txt')
TFs_human_animaltfdb_path = pkg_resources.resource_filename(package, 'CEFCON/resources/hs_hgnc_tfs_animaltfdb4.txt')
TFs_mouse_animaltfdb_path = pkg_resources.resource_filename(package, 'CEFCON/resources/mm_mgi_tfs_animaltfdb4.txt')

TFs_human = pd.read_csv(TFs_human_path, names=['gene_symbol'])
TFs_mouse = pd.read_csv(TFs_mouse_path, names=['gene_symbol'])
TFs_human_animaltfdb = pd.read_csv(TFs_human_animaltfdb_path, names=['gene_symbol'])
TFs_mouse_animaltfdb = pd.read_csv(TFs_mouse_animaltfdb_path, names=['gene_symbol'])

__all__ = ['TFs_human', 'TFs_mouse', 'TFs_human_animaltfdb', 'TFs_mouse_animaltfdb']
