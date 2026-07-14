"""The proxy savings-ledger event must record the pre-compression original as
`before`, not the forwarded (post-compression) count.

`emit_request_outcome` passes `input_tokens=outcome.optimized_tokens` (the
forwarded count) to `record_request`. The durable ledger derives the reported
reduction percent as saved / before, so `before` must be the original input.
Passing the forwarded count understated `before` by `tokens_saved` and inflated
the percentage. This guards the call shape at the source level (importing the
module pulls in the ML stack, so we assert on the source rather than execute).
"""

from __future__ import annotations

import inspect

from headroom.proxy import prometheus_metrics


def test_record_savings_event_uses_original_input_as_before():
    src = inspect.getsource(prometheus_metrics.PrometheusMetrics.record_request)

    anchor = src.index("record_savings_event(")
    call = src[anchor : src.index(")", anchor + src[anchor:].index("source="))]

    # before = forwarded + saved (the reconstructed original); after = forwarded.
    assert "tokens_before=input_tokens + tokens_saved" in call
    assert "tokens_after=input_tokens," in call

    # The pre-fix shape (forwarded as before, forwarded-saved as after) is gone.
    assert "tokens_before=input_tokens," not in call
    assert "input_tokens - tokens_saved" not in call
