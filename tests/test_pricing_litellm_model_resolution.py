from __future__ import annotations

from headroom.pricing.litellm_model_resolution import (
    MODEL_ALIASES,
    LiteLLMModelPrefixRule,
    pricing_lookup_candidates,
    resolution_candidates,
    resolve_litellm_model_name,
)


def test_prefix_rule_matches_case_insensitively() -> None:
    rule = LiteLLMModelPrefixRule("minimax-", "minimax/")

    assert rule.candidate_for("MiniMax-M3") == "minimax/MiniMax-M3"
    assert rule.candidate_for("gpt-4o") is None


def test_resolution_candidates_try_bare_then_matching_prefix_then_alias() -> None:
    assert resolution_candidates("gpt-4o") == ("gpt-4o", "openai/gpt-4o")
    assert resolution_candidates("MiniMax-M3") == ("MiniMax-M3", "minimax/MiniMax-M3")

    retired = "claude-3-5-sonnet-20241022"
    assert resolution_candidates(retired) == (
        retired,
        f"anthropic/{retired}",
        MODEL_ALIASES[retired],
    )


def test_pricing_lookup_candidates_include_provider_prefixes_and_aliases() -> None:
    candidates = pricing_lookup_candidates("claude-3-5-sonnet-20241022")

    assert candidates[0] == "claude-3-5-sonnet-20241022"
    assert "anthropic/claude-3-5-sonnet-20241022" in candidates
    assert "minimax/claude-3-5-sonnet-20241022" in candidates
    assert candidates[-1] == MODEL_ALIASES["claude-3-5-sonnet-20241022"]


def test_retired_claude_3_sonnet_aliases_to_sonnet_tier_not_haiku() -> None:
    """Retired claude-3-sonnet must map to a Sonnet-tier price, not Haiku.

    claude-3-sonnet-20240229 was a $3/$15-per-1M model; aliasing it to
    claude-3-haiku-20240307 ($0.25/$1.25) underpriced its cost/savings ~12x.
    """
    alias = MODEL_ALIASES["claude-3-sonnet-20240229"]

    assert "haiku" not in alias
    # Same-tier target as the other retired-Sonnet aliases.
    assert alias == MODEL_ALIASES["claude-3-5-sonnet-20241022"]
    assert alias == "claude-sonnet-4-20250514"


def test_resolve_litellm_model_name_returns_first_known_candidate() -> None:
    known = {"openai/gpt-4o"}

    assert resolve_litellm_model_name("gpt-4o", known.__contains__) == "openai/gpt-4o"


def test_resolve_litellm_model_name_returns_original_when_unknown() -> None:
    assert resolve_litellm_model_name("mystery-model", lambda _: False) == "mystery-model"


def test_vertex_version_suffix_stripped_and_vertex_prefix_tried() -> None:
    # #2515: Vertex model ids carry an @YYYYMMDD version suffix that LiteLLM's
    # cost DB does not key on, so pricing missed and cost read $0. Both the
    # version-stripped bare name and the vertex_ai/ prefixed key must be tried.
    candidates = resolution_candidates("claude-opus-4@20250514")
    assert candidates[0] == "claude-opus-4@20250514"
    assert "vertex_ai/claude-opus-4@20250514" in candidates
    assert "claude-opus-4" in candidates
    assert "anthropic/claude-opus-4" in candidates
    assert "vertex_ai/claude-opus-4" in candidates

    pricing = pricing_lookup_candidates("claude-haiku-4-5@20251001")
    assert pricing[0] == "claude-haiku-4-5@20251001"
    assert "claude-haiku-4-5" in pricing
    assert "vertex_ai/claude-haiku-4-5@20251001" in pricing
    assert "anthropic/claude-haiku-4-5" in pricing


def test_non_suffixed_models_keep_exact_candidate_lists() -> None:
    # The suffix handling must not perturb ordinary (non-Vertex) model names.
    assert resolution_candidates("gpt-4o") == ("gpt-4o", "openai/gpt-4o")
    assert resolution_candidates("claude-sonnet-4-6") == (
        "claude-sonnet-4-6",
        "anthropic/claude-sonnet-4-6",
    )


def test_vertex_suffixed_models_resolve_to_a_priced_key() -> None:
    # End-to-end: with a LiteLLM known-set that mirrors the real cost DB, each
    # versioned Vertex model resolves to a real (priced) key instead of falling
    # through to the unpriced verbatim name.
    known = {
        "claude-haiku-4-5",
        "vertex_ai/claude-opus-4@20250514",
        "claude-opus-4-1",
        "claude-sonnet-4-6",
    }
    assert resolve_litellm_model_name("claude-haiku-4-5@20251001", known.__contains__) == (
        "claude-haiku-4-5"
    )
    assert resolve_litellm_model_name("claude-opus-4@20250514", known.__contains__) == (
        "vertex_ai/claude-opus-4@20250514"
    )
    assert resolve_litellm_model_name("claude-opus-4-1@20250805", known.__contains__) == (
        "claude-opus-4-1"
    )
    # No suffix, already priced — unchanged.
    assert resolve_litellm_model_name("claude-sonnet-4-6", known.__contains__) == (
        "claude-sonnet-4-6"
    )
