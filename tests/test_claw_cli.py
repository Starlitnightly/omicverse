import io

from omicverse.claw import _StdoutRelay


def test_stdout_relay_mirrors_and_captures_complete_lines():
    mirror = io.StringIO()
    lines = []
    relay = _StdoutRelay(mirror_stream=mirror, emit_line=lines.append)

    relay.write("line one\nline two\n")
    relay.finalize()

    assert relay.getvalue() == "line one\nline two\n"
    assert mirror.getvalue() == "line one\nline two\n"
    assert lines == ["line one", "line two"]


def test_stdout_relay_flushes_partial_line_on_finalize():
    lines = []
    relay = _StdoutRelay(emit_line=lines.append)

    relay.write("partial")
    relay.write(" line\nnext")
    relay.finalize()

    assert relay.getvalue() == "partial line\nnext"
    assert lines == ["partial line", "next"]
