"""The URL-path model (Vertex rawPredict) must reach the backend via body["model"].

Issue #2363: Claude Code's Vertex rawPredict requests carry the model only in
the URL path (surfaced as ``model_override``) with no ``body["model"]``. The
handler resolved it correctly but only wrote it back when a string body model
already existed, so on the ``--backend litellm-vertex`` path
``LiteLLMBackend.send_message`` read ``body.get("model", <hardcoded default>)``
and silently sent every request as ``claude-3-5-sonnet-20241022``.
"""

from __future__ import annotations

from headroom.proxy.handlers.anthropic import _model_to_persist_in_body


def test_existing_string_body_model_still_sanitized_in_place() -> None:
    # Long-standing behavior: a display/ANSI-styled body model is rewritten to
    # its sanitized form regardless of backend.
    assert (
        _model_to_persist_in_body("claude-opus-4-8\x1b[1m", "claude-opus-4-8", has_backend=True)
        == "claude-opus-4-8"
    )
    assert (
        _model_to_persist_in_body("claude-opus-4-8\x1b[1m", "claude-opus-4-8", has_backend=False)
        == "claude-opus-4-8"
    )


def test_unchanged_string_body_model_is_not_rewritten() -> None:
    assert _model_to_persist_in_body("claude-opus-4-8", "claude-opus-4-8", has_backend=True) is None


def test_url_only_model_is_persisted_on_the_backend_path() -> None:
    # #2363: no body model + backend consumes the body -> persist the resolved
    # (model_override) model so the backend forwards it instead of a default.
    assert (
        _model_to_persist_in_body(None, "claude-opus-4@20250514", has_backend=True)
        == "claude-opus-4@20250514"
    )


def test_url_only_model_is_left_alone_on_a_native_passthrough() -> None:
    # No backend: the URL carries the model and the raw body is forwarded
    # upstream (Vertex), which would reject an unexpected body model.
    assert _model_to_persist_in_body(None, "claude-opus-4@20250514", has_backend=False) is None


def test_unknown_model_is_not_persisted() -> None:
    # Neither body nor URL supplied a model; do not stamp the "unknown" sentinel.
    assert _model_to_persist_in_body(None, "unknown", has_backend=True) is None


def test_non_string_resolved_model_is_left_alone() -> None:
    assert _model_to_persist_in_body(None, None, has_backend=True) is None
    assert _model_to_persist_in_body(None, 123, has_backend=True) is None
