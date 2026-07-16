"""Structural guard for the OpenAI Responses WS memory-relay control flow.

The relay logic in ``handlers/openai.py`` is a three-phase state machine
(buffer → suppress → pass-through) whose intended behaviour is pinned by the
reference implementation in ``tests/test_ws_memory_relay.py``. The real handler
once diverged from it: the Phase-1 "flush + record metrics + reset" block was
dedented to Phase-1's own level, so it ran on EVERY event — memory-tool
suppression never took effect, per-response metrics were recorded per-event, and
the suppress / pass-through phases were unreachable dead code.

This reads the handler source (no import — the module pulls in the ML stack) and
asserts the flush/record/reset stays scoped to the ``response.completed`` branch
inside Phase 1, and that the suppress phase is still reachable.
"""

from __future__ import annotations

from pathlib import Path

_OPENAI = (
    Path(__file__).resolve().parents[1] / "headroom" / "proxy" / "handlers" / "openai.py"
)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def test_ws_relay_phase1_flush_scoped_to_response_completed():
    lines = _OPENAI.read_text(encoding="utf-8").splitlines()

    p1 = next(i for i, ln in enumerate(lines) if "Phase 1: Buffer until first output item" in ln)
    p2a = next(
        i for i, ln in enumerate(lines) if i > p1 and "Phase 2a: Suppress mode" in ln
    )
    region = lines[p1:p2a]

    if_not_decided = next(ln for ln in region if ln.strip() == "if not decided:")
    base = _indent(if_not_decided)

    # `_record_ws_response_metrics()` and `_reset()` in Phase 1 must be nested
    # deeper than `if not decided:` (i.e. inside the response.completed branch),
    # not run unconditionally on every event.
    scoped = [ln for ln in region if "_record_ws_response_metrics()" in ln or "_reset()" in ln]
    assert scoped, "expected the response.completed flush inside Phase 1"
    for ln in scoped:
        assert _indent(ln) > base + 4, f"Phase-1 flush ran unconditionally: {ln!r}"

    # The suppress phase must still be present/reachable after Phase 1.
    assert any(ln.strip() == "if suppress_response:" for ln in lines[p2a : p2a + 6])
