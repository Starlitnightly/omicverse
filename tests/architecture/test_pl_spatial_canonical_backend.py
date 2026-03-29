import omicverse.pl._scatterplot_backend as scatter_backend


def test_scatterplot_backend_spatial_delegates_to_canonical(monkeypatch):
    calls = []

    def fake_spatial(*args, **kwargs):
        calls.append((args, kwargs))
        return "canonical"

    monkeypatch.setattr("omicverse.pl._spatial.spatial", fake_spatial)

    result = scatter_backend.spatial("adata", basis="spatial", show=False)

    assert result == "canonical"
    assert calls
    assert calls[0][0] == ("adata",)
    assert calls[0][1]["basis"] == "spatial"
    assert calls[0][1]["show"] is False
