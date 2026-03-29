import warnings

import omicverse as ov


def test_unique_curve_exports_are_exposed():
    assert callable(ov.pl.cpdb_curved_graph)
    assert callable(ov.pl.cpdb_curved_line)
    assert callable(ov.pl.cpdb_plot_curve_network)
    assert callable(ov.pl.flowsig_curved_graph)
    assert callable(ov.pl.flowsig_curved_line)
    assert callable(ov.pl.flowsig_plot_curve_network)


def test_generic_curve_alias_points_to_flowsig_behavior():
    assert ov.pl.flowsig_curved_graph is not ov.pl.cpdb_curved_graph
    assert ov.pl.flowsig_curved_line is not ov.pl.cpdb_curved_line
    assert ov.pl.flowsig_plot_curve_network is not ov.pl.cpdb_plot_curve_network


def test_generic_curve_aliases_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ov.pl.curved_line(0, 0, 1, 1)
        except Exception:
            # The backend may fail if optional deps like bezier are unavailable;
            # this test only cares that the deprecation warning is emitted first.
            pass

    assert caught
    assert "deprecated and ambiguous" in str(caught[0].message)
