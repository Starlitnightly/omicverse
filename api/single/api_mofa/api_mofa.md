
::: omicverse.single.pyMOFA
    handler: python
    selection:
        options:
        members:
            - __init__
            - mofa_preprocess
            - mofa_run
        show_root_heading: true
        show_source: true

::: omicverse.single.pyMOFAART
    handler: python
    selection:
        options:
        members:
            - __init__
            - get_factors
            - get_r2
            - plot_r2
            - get_cor
            - plot_cor
            - plot_factor
            - plot_weight_gene_d1
            - plot_weight_gene_d2
            - plot_weights
            - plot_top_feature_dotplot
            - plot_top_feature_heatmap
            - get_top_feature
        show_root_heading: true
        show_source: true

::: omicverse.single.GLUE_pair
    handler: python
    selection:
        options:
        members:
            - __init__
            - correlation
            - find_neighbor_cell
            - pair_omic
        show_root_heading: true
        show_source: true

::: omicverse.single.factor_exact
    handler: python
    selection:
        options:
        show_root_heading: true
        show_source: true

::: omicverse.single.factor_correlation
    handler: python
    selection:
        options:
        show_root_heading: true
        show_source: true

::: omicverse.single.get_weights
    handler: python
    selection:
        options:
        show_root_heading: true
        show_source: true

::: omicverse.single.glue_pair
    handler: python
    selection:
        options:
        show_root_heading: true
        show_source: true