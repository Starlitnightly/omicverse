import os
from typing import Any, Dict, List, Optional, Tuple

import nbformat
import numpy as np
import plotly.graph_objects as go
from jinja2 import Template
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

from . import __version__
from .config import PACKAGE_PATH
from .dry import dryable
from .utils import get_temporary_filename

REPORT_DIR = os.path.join(PACKAGE_PATH, 'report')
BASIC_TEMPLATE_PATH = os.path.join(REPORT_DIR, 'report_basic.ipynb')
MATRIX_TEMPLATE_PATH = os.path.join(REPORT_DIR, 'report_matrix.ipynb')

MARGIN = go.layout.Margin(t=5, r=5, b=0, l=5)  # noqa: E741


def dict_to_table(
    d: Dict[str, Any],
    column_ratio: List[int] = [3, 7],
    column_align: List[str] = ['right', 'left']
) -> go.Figure:
    """Convert a dictionary to a Plot.ly table of key-value pairs.

    Args:
        d: Dictionary to convert
        column_ratio: Relative column widths, represented as a ratio,
            defaults to `[3, 7]`
        column_align: Column text alignments, defaults to `['right', 'left']`

    Returns:
        Figure
    """
    keys = []
    values = []
    for key, value in d.items():
        if isinstance(value, list):
            keys.append(key)
            values.append(value[0])
            for val in value[1:]:
                keys.append('')
                values.append(val)
        else:
            keys.append(key)
            values.append(value)

    table = go.Table(
        columnwidth=column_ratio,
        header={'values': ['key', 'value']},
        cells={
            'values': [keys, values],
            'align': column_align
        }
    )
    figure = go.Figure(data=table)
    figure.update_layout(
        margin=MARGIN,
        xaxis_automargin=True,
        yaxis_automargin=True,
        autosize=True
    )
    return figure


def knee_plot(n_counts: List[int]) -> go.Figure:
    """Generate knee plot card.

    Args:
        n_counts: List of UMI counts

    Returns:
        Figure
    """
    knee = np.sort(n_counts)[::-1]
    scatter = go.Scattergl(x=knee, y=np.arange(len(knee)), mode='lines')
    figure = go.Figure(data=scatter)
    figure.update_layout(
        margin=MARGIN,
        xaxis_title='UMI counts',
        yaxis_title='Number of barcodes',
        xaxis_type='log',
        yaxis_type='log',
        xaxis_automargin=True,
        yaxis_automargin=True,
        autosize=True
    )
    return figure


def genes_detected_plot(n_counts: List[int], n_genes: List[int]) -> go.Figure:
    """Generate genes detected plot card.

    Args:
        n_counts: List of UMI counts
        n_genes: List of gene counts

    Returns:
        Figure
    """
    scatter = go.Scattergl(x=n_counts, y=n_genes, mode='markers')
    figure = go.Figure(data=scatter)
    figure.update_layout(
        margin=MARGIN,
        xaxis_title='UMI counts',
        yaxis_title='Genes detected',
        xaxis_type='log',
        yaxis_type='log',
        xaxis_automargin=True,
        yaxis_automargin=True,
        autosize=True
    )
    return figure


def elbow_plot(pca_variance_ratio: List[float]) -> go.Figure:
    """Generate elbow plot card.

    Args:
        pca_variance_ratio: List PCA variance ratios

    Returns:
        Figure
    """
    scatter = go.Scattergl(
        x=np.arange(1,
                    len(pca_variance_ratio) + 1),
        y=pca_variance_ratio,
        mode='markers'
    )
    figure = go.Figure(data=scatter)
    figure.update_layout(
        margin=MARGIN,
        xaxis_title='PC',
        yaxis_title='Explained variance ratio',
        xaxis_automargin=True,
        yaxis_automargin=True,
        autosize=True
    )
    return figure


def pca_plot(pc: np.ndarray) -> go.Figure:
    """Generate PCA plot card.

    Args:
        pc: Embeddings

    Returns:
        Figure
    """
    scatter = go.Scattergl(x=pc[:, 0], y=pc[:, 1], mode='markers')
    figure = go.Figure(data=scatter)
    figure.update_layout(
        margin=MARGIN,
        xaxis_title='PC 1',
        yaxis_title='PC 2',
        xaxis_automargin=True,
        yaxis_automargin=True,
        autosize=True
    )
    return figure


def write_report(
    stats_path: str,
    info_path: str,
    inspect_path: str,
    out_path: str,
    matrix_path: Optional[str] = None,
    barcodes_path: Optional[str] = None,
    genes_path: Optional[str] = None,
    t2g_path: Optional[str] = None,
) -> str:
    """Render the Jupyter notebook report with Jinja2.

    Args:
        stats_path: Path to kb stats JSON
        info_path: Path to run_info.json
        inspect_path: Path to inspect.json
        out_path: Path to Jupyter notebook to generate
        matrix_path: Path to matrix
        barcodes_path: List of paths to barcodes.txt
        genes_path: Path to genes.txt, defaults to `None`
        t2g_path: Path to transcript-to-gene mapping

    Returns:
        Path to notebook generated
    """
    template_path = MATRIX_TEMPLATE_PATH if all(
        p is not None
        for p in [matrix_path, barcodes_path, genes_path, t2g_path]
    ) else BASIC_TEMPLATE_PATH
    with open(template_path, 'r') as f, open(out_path, 'w') as out:
        template = Template(f.read())
        out.write(
            template.render(
                packages=f'#!pip install kb-python>={__version__}',
                stats_path=stats_path,
                info_path=info_path,
                inspect_path=inspect_path,
                matrix_path=matrix_path,
                barcodes_path=barcodes_path,
                genes_path=genes_path,
                t2g_path=t2g_path
            )
        )

    return out_path


def execute_report(execute_path: str, nb_path: str,
                   html_path: str) -> Tuple[str, str]:
    """Execute the report and write the results as a Jupyter notebook and HTML.

    Args:
        execute_path: Path to Jupyter notebook to execute
        nb_path: Path to Jupyter notebook to generate
        html_path: Path to HTML to generate

    Returns:
        Tuple containing executed notebook and HTML
    """
    with open(execute_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb)

    with open(nb_path, 'w') as f:
        nbformat.write(nb, f)

    with open(html_path, 'w') as f:
        html_exporter = HTMLExporter()
        html, resources = html_exporter.from_notebook_node(nb)
        f.write(html)

    return nb_path, html_path


@dryable(lambda *args, **kwargs: {})
def render_report(
    stats_path: str,
    info_path: str,
    inspect_path: str,
    nb_path: str,
    html_path: str,
    matrix_path: Optional[str] = None,
    barcodes_path: Optional[str] = None,
    genes_path: Optional[str] = None,
    t2g_path: Optional[str] = None,
    temp_dir: str = 'tmp'
) -> Dict[str, str]:
    """Render and execute the report.

    Args:
        stats_path: Path to kb stats JSON
        info_path: Path to run_info.json
        inspect_path: Path to inspect.json
        nb_path: Path to Jupyter notebook to generate
        html_path: Path to HTML to generate
        matrix_path: Path to matrix
        barcodes_path: List of paths to barcodes.txt
        genes_path: Path to genes.txt, defaults to `None`
        t2g_path: Path to transcript-to-gene mapping
        temp_dir: Path to temporary directory, defaults to `tmp`

    Returns:
        Dictionary containing notebook and HTML paths
    """
    temp_path = write_report(
        stats_path, info_path, inspect_path, get_temporary_filename(temp_dir),
        matrix_path, barcodes_path, genes_path, t2g_path
    )
    execute_report(temp_path, nb_path, html_path)

    return {'report_notebook': nb_path, 'report_html': html_path}
