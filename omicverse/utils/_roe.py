import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from anndata import AnnData
import numpy as np
from .._registry import register_function




@register_function(
    aliases=["观察预期比", "roe", "observed_expected_ratio", "细胞富集分析", "组织偏好性"],
    category="utils",
    description="Calculate ratio of observed to expected cell numbers (Ro/e) for tissue preference analysis",
    examples=[
        "# Basic Ro/e analysis",
        "roe_result = ov.utils.roe(adata, sample_key='batch',",
        "                          cell_type_key='celltype')",
        "# Custom threshold analysis",
        "roe_result = ov.utils.roe(adata, sample_key='tissue',",
        "                          cell_type_key='scsa_celltype',",
        "                          pval_threshold=0.01)",
        "# Visualize with heatmap",
        "import seaborn as sns",
        "transformed_roe = roe_result.applymap(",
        "    lambda x: '+++' if x >= 2 else ('++' if x >= 1.5 else '+/-'))",
        "sns.heatmap(roe_result, annot=transformed_roe, cmap='RdBu_r')",
        "# Statistical significance testing",
        "roe_result = ov.utils.roe(adata, sample_key='condition',",
        "                          cell_type_key='leiden')"
    ],
    related=["single.pySCSA", "utils.plot_cellproportion", "utils.embedding"]
)
def roe(
        adata: AnnData,
        sample_key: str,
        cell_type_key: str,
        pval_threshold: float = 0.05,
        expected_value_threshold: float = 5,
        order="F"
) -> pd.DataFrame:
    """Compute observed/expected (Ro/e) cell-type enrichment ratios.

    Parameters
    ----------
    adata : AnnData
        Input object with sample and cell-type annotations.
    sample_key : str
        Column in ``adata.obs`` representing sample/condition groups.
    cell_type_key : str
        Column in ``adata.obs`` representing cell-type labels.
    pval_threshold : float, default=0.05
        Significance threshold for global contingency-table test.
    expected_value_threshold : float, default=5
        Minimum expected count threshold used to judge chi-square validity.
    order : str, default='F'
        Optional comma-separated sample order for contingency table columns.
        Use ``'F'`` to keep original order.

    Returns
    -------
    pandas.DataFrame
        Ro/e matrix with cell types as rows and samples as columns.
    """
    
    # Create a contingency table
    num_cell = pd.crosstab(index=adata.obs[cell_type_key], columns=adata.obs[sample_key])
    num_cell.index.name = "cluster"
    # If necessary, reorder columns based on the given order
    if order != "F":
        col_order = order.split(',')
        num_cell = num_cell[col_order]
    # Perform chi-square test
    chi2, p, dof, expected = chi2_contingency(num_cell)
    print(f"chi2: {chi2}, dof: {dof}, pvalue: {p}")
    
    # Check if expected values are too low for chi-square test
    use_fisher = any(item < expected_value_threshold for item in expected.flatten())
    
    if use_fisher:
        # Use Fisher's exact test for 2x2 tables, or warn for larger tables
        if num_cell.shape == (2, 2):
            print(f"Some expected frequencies are less than {expected_value_threshold}, using Fisher's exact test")
            # Convert to numpy array for fisher_exact
            contingency_array = num_cell.values
            _, p_fisher = fisher_exact(contingency_array)
            print(f"Fisher's exact test p-value: {p_fisher}")
            p = p_fisher  # Use Fisher's p-value instead of chi-square
        else:
            print(f"Some expected frequencies are less than {expected_value_threshold}. "
                  f"Fisher's exact test is not available for tables larger than 2x2. "
                  f"Consider using other statistical methods. Results may be unreliable.")
    
    # Check p-value
    if p <= pval_threshold:
        expected_data = pd.DataFrame(expected, index=num_cell.index, columns=num_cell.columns)
        roe = num_cell / expected_data
        adata.uns['roe_results'] = roe
        adata.uns['expected_values'] = expected_data
        
        if use_fisher and num_cell.shape != (2, 2):
            # Mark as unreliable if we couldn't use Fisher's test for larger tables
            adata.uns['unreliable_roe_results'] = roe
        else:
            adata.uns['sig_roe_results'] = roe
    else:
        print("P-value is greater than 0.05, there is no statistical significance")
        expected_data = pd.DataFrame(expected, index=num_cell.index, columns=num_cell.columns)
        roe = num_cell / expected_data
        adata.uns['unsig_roe_results'] = roe
        adata.uns['expected_values'] = expected_data

    if 'sig_roe_results' in adata.uns:
        return adata.uns['sig_roe_results']
    elif 'unreliable_roe_results' in adata.uns:
        return adata.uns['unreliable_roe_results']
    else:
        return adata.uns['unsig_roe_results']



def roe_plot_heatmap(adata: AnnData, display_numbers: bool = False, center_value: float = 1.0,
                     color_scheme: str = 'cool',
                 custom_colors: list = None, save_path: str = None, batch_order: list = None):
    """Plot Ro/e enrichment heatmap from ``adata.uns`` results.

    Parameters
    ----------
    adata : AnnData
        Object containing Ro/e outputs generated by :func:`roe`.
    display_numbers : bool, default=False
        If ``True``, annotate cells with numeric Ro/e values.
        If ``False``, use symbolic categories.
    center_value : float, default=1.0
        Center value for diverging colormap.
    color_scheme : str, default='cool'
        Preset palette key: ``'default'``, ``'cool'``, or ``'warm'``.
    custom_colors : list or None, default=None
        Optional custom color list overriding preset scheme.
    save_path : str or None, default=None
        Optional image output path.
    batch_order : list or None, default=None
        Optional row ordering for displayed samples.

    Returns
    -------
    None
        Displays or saves the heatmap.
    """
    # Check for results in order of reliability
    if 'sig_roe_results' in adata.uns:
        roe = adata.uns['sig_roe_results']
        title_suffix = ""
    elif 'unreliable_roe_results' in adata.uns:
        roe = adata.uns['unreliable_roe_results']
        title_suffix = " (Statistical test unreliable - low expected frequencies)"
    else:
        roe = adata.uns['unsig_roe_results']
        title_suffix = " (Not significant)"

    # 如果提供了批次顺序，则按批次排序
    if batch_order:
        roe = roe.loc[batch_order]

    # 定义默认颜色方案
    color_schemes = {
        'default': ['#D73027', '#FFFFFF', '#1E682A'],
        'cool': ['#440154', '#FFFFFF', '#fde725'],
        'warm': ['#D53E4F', '#F46D43', '#FDAE61']
    }

    # 创建子图以便后续添加图例
    fig, ax = plt.subplots(figsize=(10, 7))

    # 创建定制的颜色映射，使1成为中间值
    if custom_colors:
        colors = custom_colors
    else:
        colors = color_schemes[color_scheme]

    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    custom_cmap.set_bad(color='white')

    # Choose display method based on display_numbers parameter
    if display_numbers:
        annot = roe.round(2)  # Keep two decimal places
        sns.heatmap(roe, annot=annot, cmap=custom_cmap, center=center_value, fmt='', ax=ax)
        plt.title(f"ROE Results{title_suffix}")
    else:
        transformed_roe = transform_roe_values(roe)
        sns.heatmap(roe, annot=transformed_roe, cmap=custom_cmap, center=center_value, fmt='', cbar=False, ax=ax)

        # Add legend with Nature paper thresholds
        cbar_ax = fig.add_axes([0.93, 0.2, 0.03, 0.6])
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap), cax=cbar_ax, orientation='vertical')
        cbar.set_ticks([0.0, 0.1, 0.5, 0.9, 2.0])
        cbar.set_ticklabels(['—', '+/-', '+', '++', '+++'])

    plt.title(f"Ro/e{title_suffix}")
    if save_path:
        plt.savefig(save_path)  # 保存图像
    else:
        plt.show()


def transform_roe_values(roe):
    """Convert numeric Ro/e values to symbolic enrichment categories.

    Parameters
    ----------
    roe : pandas.DataFrame
        Numeric Ro/e matrix.

    Returns
    -------
    pandas.DataFrame
        String-formatted matrix with symbols ``—``, ``+/-``, ``+``, ``++``,
        and ``+++``.
    """
    # Transform roe DataFrame to string format for annotation
    # Nature paper implementation thresholds: >1 (+++), 0.8<x≤1 (++), 0.2≤x≤0.8 (+), 0<x<0.2 (+/-), =0 (—)
    def _categorize_value(x):
        if x == 0:
            return "—"
        if 0 < x < 0.2:
            return "+/-"
        if 0.2 <= x <= 0.8:
            return "+"
        if 0.8 < x <= 1:
            return "++"
        return "+++"

    transformed_roe = roe.copy()
    # DataFrame.applymap was removed in pandas 3.0; use column-wise map instead.
    return transformed_roe.apply(lambda col: col.map(_categorize_value))


# roe(adata, sample_key='batch', cell_type_key='celltypist_cell_label_coarse')
# plot_heatmap(adata, display_numbers=True)
# batch_order = ['sample3', 'sample1', 'sample2']
# plot_heatmap(adata, batch_order=batch_order)
