from __future__ import annotations

from omicverse.jarvis.channels.telegram import _run_polling_with_restart


class _FakeApp:
    def __init__(self) -> None:
        self.kwargs = None

    def add_error_handler(self, _handler) -> None:
        return

    def run_polling(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_run_polling_disables_signal_registration_in_threads() -> None:
    fake_app = _FakeApp()

    _run_polling_with_restart(
        application_factory=lambda: fake_app,
        conflict_type=RuntimeError,
        max_attempts=1,
    )

    assert fake_app.kwargs is not None
    assert fake_app.kwargs["drop_pending_updates"] is True
    assert fake_app.kwargs["stop_signals"] is None
