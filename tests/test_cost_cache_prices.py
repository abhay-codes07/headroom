"""``_get_cache_prices`` must not bill cache reads at the full uncached rate.

LiteLLM omits ``cache_read_input_token_cost`` / ``cache_creation_input_token_cost``
for the long tail of its priced models (most Bedrock, Mistral, Fireworks and
OpenAI-compatible gateway models). The old ``.get(field, uncached)`` default
billed cache reads at the full uncached rate and cache writes with no premium,
so ``totals()`` (and the /stats figures it feeds) over-charged every cache-warm
request on those models. When a field is absent the tracker must fall back to
the provider cache economics the dashboard already uses (``_CACHE_ECONOMICS``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.proxy.cost import (
    _CACHE_ECONOMICS,
    CostTracker,
    _cache_economics_provider,
)


def _patch_litellm(monkeypatch: pytest.MonkeyPatch, model_cost: dict) -> None:
    monkeypatch.setattr(
        "headroom.proxy.cost._get_litellm_module",
        lambda: SimpleNamespace(model_cost=model_cost),
    )
    monkeypatch.setattr(
        "headroom.pricing.litellm_pricing.resolve_litellm_model",
        lambda model: model,
    )


class TestCacheEconomicsProvider:
    @pytest.mark.parametrize(
        ("litellm_provider", "expected"),
        [
            ("bedrock", "bedrock"),
            ("bedrock_converse", "bedrock"),
            ("openai", "openai"),
            ("azure", "openai"),
            ("azure_ai", "openai"),
            ("gemini", "gemini"),
            ("vertex_ai", "gemini"),
            ("anthropic", "anthropic"),
            ("mistral", "anthropic"),  # unknown -> anthropic default
            ("fireworks_ai", "anthropic"),
            (None, "anthropic"),
            ("", "anthropic"),
        ],
    )
    def test_maps_to_cache_economics_key(self, litellm_provider, expected) -> None:
        assert _cache_economics_provider(litellm_provider) == expected


class TestGetCachePrices:
    def test_uses_explicit_litellm_cache_fields_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_litellm(
            monkeypatch,
            {
                "m": {
                    "input_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": 1e-7,
                    "cache_creation_input_token_cost": 1.25e-6,
                    "litellm_provider": "anthropic",
                }
            },
        )
        assert CostTracker()._get_cache_prices("m") == (1e-7, 1.25e-6, 1e-6)

    def test_missing_cache_read_uses_provider_discount_not_full_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bedrock model with an input price but no cache fields: exactly the
        # 1600+ long-tail models the bug affects.
        _patch_litellm(
            monkeypatch,
            {"m": {"input_cost_per_token": 5e-7, "litellm_provider": "bedrock"}},
        )
        cache_read, cache_write, uncached = CostTracker()._get_cache_prices("m")
        assert uncached == 5e-7
        # Before the fix this was 5e-7 (the full uncached rate) — the bug.
        assert cache_read == pytest.approx(5e-7 * _CACHE_ECONOMICS["bedrock"]["read_multiplier"])
        assert cache_read < uncached
        assert cache_write == pytest.approx(5e-7 * _CACHE_ECONOMICS["bedrock"]["write_multiplier"])

    def test_openai_provider_uses_openai_multipliers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_litellm(
            monkeypatch,
            {"m": {"input_cost_per_token": 1e-6, "litellm_provider": "openai"}},
        )
        cache_read, cache_write, uncached = CostTracker()._get_cache_prices("m")
        # OpenAI: reads at 0.5x, writes at 1.0x (no write premium).
        assert cache_read == pytest.approx(1e-6 * 0.5)
        assert cache_write == pytest.approx(1e-6 * 1.0)

    def test_unknown_provider_defaults_to_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_litellm(
            monkeypatch,
            {"m": {"input_cost_per_token": 2e-6}},  # no litellm_provider
        )
        cache_read, cache_write, uncached = CostTracker()._get_cache_prices("m")
        assert cache_read == pytest.approx(2e-6 * _CACHE_ECONOMICS["anthropic"]["read_multiplier"])
        assert cache_write == pytest.approx(
            2e-6 * _CACHE_ECONOMICS["anthropic"]["write_multiplier"]
        )

    def test_only_the_missing_field_is_filled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Explicit read cost present, write cost absent: keep the real read,
        # infer only the write.
        _patch_litellm(
            monkeypatch,
            {
                "m": {
                    "input_cost_per_token": 1e-6,
                    "cache_read_input_token_cost": 3e-7,
                    "litellm_provider": "anthropic",
                }
            },
        )
        cache_read, cache_write, _ = CostTracker()._get_cache_prices("m")
        assert cache_read == 3e-7  # untouched real value
        assert cache_write == pytest.approx(
            1e-6 * _CACHE_ECONOMICS["anthropic"]["write_multiplier"]
        )

    def test_no_input_price_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_litellm(monkeypatch, {"m": {"litellm_provider": "anthropic"}})
        assert CostTracker()._get_cache_prices("m") is None
