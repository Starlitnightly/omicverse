from __future__ import annotations

import threading
import time

from omicverse_web.gateway.inprocess_channel_manager import InProcessChannelManager


def test_inprocess_manager_supports_all_channels(monkeypatch) -> None:
    started: list[tuple[str, object]] = []
    release = threading.Event()

    def _fake_run_channel(self, channel, cfg, stop_event) -> None:
        started.append((channel, self._sm))
        stop_event.wait(timeout=0.2)
        release.wait(timeout=0.2)

    monkeypatch.setattr(InProcessChannelManager, "_run_channel", _fake_run_channel)

    manager = InProcessChannelManager(session_manager=object())
    cfg = {
        "telegram": {"token": "tg-token", "allowed_users": []},
        "feishu": {"app_id": "fid", "app_secret": "fsec", "connection_mode": "websocket"},
        "qq": {"app_id": "qid", "client_secret": "qsec"},
        "imessage": {"cli_path": "imsg"},
    }

    results = manager.auto_start_configured(cfg)
    assert len(results) == 4
    assert all(item["ok"] for item in results)

    deadline = time.time() + 2.0
    while len(started) < 4 and time.time() < deadline:
        time.sleep(0.01)

    assert {channel for channel, _sm in started} == {"telegram", "feishu", "qq", "imessage"}
    assert {id(sm) for _channel, sm in started} == {id(manager._sm)}

    stop_results = manager.stop_all()
    assert all(item["ok"] for item in stop_results)
    release.set()
