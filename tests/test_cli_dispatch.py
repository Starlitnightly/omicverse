import types

from omicverse import cli


def test_claw_subcommand_defaults_to_gateway_mode(monkeypatch):
    captured = {}

    def fake_jarvis_main(argv=None):
        captured["argv"] = argv
        return 23

    monkeypatch.setitem(
        __import__("sys").modules,
        "omicverse.jarvis.cli",
        types.SimpleNamespace(main=fake_jarvis_main),
    )

    rc = cli.main(["claw", "--web-port", "5055"])

    assert rc == 23
    assert captured["argv"] == ["--with-web", "--gateway-daemon", "--web-port", "5055"]


def test_claw_subcommand_keeps_one_shot_mode(monkeypatch):
    captured = {}

    def fake_jarvis_main(argv=None):
        captured["argv"] = argv
        return 29

    monkeypatch.setitem(
        __import__("sys").modules,
        "omicverse.jarvis.cli",
        types.SimpleNamespace(main=fake_jarvis_main),
    )

    rc = cli.main(["claw", "-q", "hello"])

    assert rc == 29
    assert captured["argv"] == ["-q", "hello"]


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
