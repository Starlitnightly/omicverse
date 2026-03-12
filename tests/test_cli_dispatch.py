import types

from omicverse import cli


def test_web_subcommand_dispatches_to_web_launcher(monkeypatch):
    captured = {}

    def fake_web_main(argv=None):
        captured["argv"] = argv
        return 17

    monkeypatch.setitem(
        __import__("sys").modules,
        "omicverse_web.start_server",
        types.SimpleNamespace(main=fake_web_main),
    )

    rc = cli.main(["web", "--port", "5055", "--host", "127.0.0.1"])

    assert rc == 17
    assert captured["argv"] == ["--port", "5055", "--host", "127.0.0.1"]
