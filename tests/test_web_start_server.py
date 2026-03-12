from tests.utils._web_test_support import import_web_module


start_server = import_web_module("omicverse_web.start_server")


def test_resolve_bind_host_ignores_unresolvable_cli_host(monkeypatch):
    monkeypatch.delenv("OV_WEB_HOST", raising=False)
    monkeypatch.delenv("HOST", raising=False)

    host, note = start_server._resolve_bind_host("arm64-apple-darwin20.0.0", False)

    assert host == "0.0.0.0"
    assert "could not be resolved" in (note or "")


def test_resolve_bind_host_prefers_valid_ov_web_host(monkeypatch):
    monkeypatch.setenv("OV_WEB_HOST", "127.0.0.1")
    monkeypatch.setenv("HOST", "arm64-apple-darwin20.0.0")

    host, note = start_server._resolve_bind_host(None, True)

    assert host == "127.0.0.1"
    assert note is None
