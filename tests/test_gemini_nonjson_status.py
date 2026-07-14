"""A non-JSON Gemini upstream body must not be masked as a synthetic 502.

`response.json()` raises `json.JSONDecodeError` (a `ValueError`) on a non-JSON
error body; the token-extraction `except` must catch it so the handler falls
through and forwards the real upstream status/content, matching the all-non-text
sibling branch.
"""

from __future__ import annotations

import inspect

from headroom.proxy.handlers import gemini


def test_generate_content_token_extraction_catches_non_json_body():
    src = inspect.getsource(gemini.GeminiHandlerMixin.handle_gemini_generate_content)

    # Locate the token-extraction except that guards `response.json()` (right
    # after the cachedContentTokenCount read) and assert it catches the
    # JSON/ValueError family rather than only the KeyError/TypeError/Attribute
    # trio (which let a JSONDecodeError escape to the outer 502).
    anchor = src.index("cachedContentTokenCount")
    tail = src[anchor:]
    except_clause = tail[tail.index("except") : tail.index("except") + 120]

    assert "json.JSONDecodeError" in except_clause
    assert "ValueError" in except_clause
