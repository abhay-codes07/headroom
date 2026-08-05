"""Pure LiteLLM model-name resolution rules."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# Vertex AI model ids carry a date-stamped version suffix
# (``claude-opus-4@20250514``) that LiteLLM's cost DB does not key on, so a
# verbatim lookup misses and cost reads $0 (#2515). Stripping it yields the
# bare Anthropic key that IS priced.
_VERSION_SUFFIX_RE = re.compile(r"@\d{8}$")

# Vertex-routed Claude/Gemini keys live under this LiteLLM provider prefix, both
# with the version suffix (``vertex_ai/claude-opus-4@20250514``) and, for some
# models, only as the bare Anthropic key after stripping.
_VERTEX_PREFIX = "vertex_ai/"


def _strip_version_suffix(model: str) -> str:
    """Drop a trailing Vertex ``@YYYYMMDD`` model-version suffix (#2515).

    Returns ``model`` unchanged when there is no such suffix, so non-Vertex
    model names keep their existing candidate lists exactly.
    """
    return _VERSION_SUFFIX_RE.sub("", model)


@dataclass(frozen=True, slots=True)
class LiteLLMModelPrefixRule:
    """Case-insensitive bare-model prefix mapping to a LiteLLM provider key."""

    model_prefix: str
    provider_prefix: str

    def candidate_for(self, model: str) -> str | None:
        if model.lower().startswith(self.model_prefix):
            return f"{self.provider_prefix}{model}"
        return None


# Aliases for models removed from LiteLLM's cost database (retired/renamed).
# Maps old model name -> current LiteLLM key that has equivalent pricing.
MODEL_ALIASES: dict[str, str] = {
    # Claude 3.5 Sonnet retired Feb 2026, pricing same as claude-sonnet-4-20250514
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-20250514",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4-20250514",
    # Claude 3 Sonnet retired. It was a Sonnet-tier model ($3/$15 per 1M
    # in/out) — same price as claude-sonnet-4-20250514 — so alias it there.
    # The old target, claude-3-haiku-20240307 ($0.25/$1.25), is a different
    # (Haiku) tier and underpriced every cost/savings figure ~12x.
    "claude-3-sonnet-20240229": "claude-sonnet-4-20250514",
}


MODEL_PREFIX_RULES: tuple[LiteLLMModelPrefixRule, ...] = (
    LiteLLMModelPrefixRule("claude-", "anthropic/"),
    LiteLLMModelPrefixRule("gpt-", "openai/"),
    LiteLLMModelPrefixRule("o1-", "openai/"),
    LiteLLMModelPrefixRule("o3-", "openai/"),
    LiteLLMModelPrefixRule("o4-", "openai/"),
    LiteLLMModelPrefixRule("gemini-", "google/"),
    LiteLLMModelPrefixRule("minimax-", "minimax/"),
    LiteLLMModelPrefixRule("deepseek-", "deepseek/"),
)


PRICE_LOOKUP_PROVIDER_PREFIXES: tuple[str, ...] = (
    "openai/",
    "anthropic/",
    "google/",
    "mistral/",
    "deepseek/",
    "minimax/",
)


def _prefix_rule_candidates(name: str) -> list[str]:
    return [
        candidate
        for rule in MODEL_PREFIX_RULES
        for candidate in (rule.candidate_for(name),)
        if candidate is not None
    ]


def resolution_candidates(model: str) -> tuple[str, ...]:
    """Return ordered LiteLLM keys to try for cost-per-token resolution."""
    candidates = [model]
    bare = _strip_version_suffix(model)
    if bare != model:
        # Vertex keeps the suffix on its own key; try it right after the exact
        # name, then fall back to the version-stripped bare name (#2515).
        candidates.append(f"{_VERTEX_PREFIX}{model}")
    candidates.extend(_prefix_rule_candidates(model))
    if bare != model:
        candidates.append(bare)
        candidates.extend(_prefix_rule_candidates(bare))
        candidates.append(f"{_VERTEX_PREFIX}{bare}")
    alias = MODEL_ALIASES.get(model) or MODEL_ALIASES.get(bare)
    if alias:
        candidates.append(alias)
    return tuple(dict.fromkeys(candidates))


def unwrapped_model_forms(model: str) -> tuple[str, ...]:
    """Progressively drop leading gateway segments: ``a/b/c`` -> ``b/c``, ``c``.

    A gateway-routed name wraps the real model id, and ``litellm.model_cost`` keys
    the *unwrapped* form -- e.g. it has ``anthropic.claude-3-5-sonnet-20241022-v2:0``
    but not ``bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0``. Deriving the
    forms by splitting on ``/`` avoids maintaining yet another list of gateway
    prefixes, and costs nothing when wrong: each candidate is an exact dict
    lookup, so a bad guess simply misses.
    """
    parts = model.split("/")
    return tuple("/".join(parts[i:]) for i in range(1, len(parts)))


def pricing_lookup_candidates(model: str) -> tuple[str, ...]:
    """Return ordered LiteLLM model_cost keys to try for pricing lookup."""
    candidates = [model]
    bare = _strip_version_suffix(model)
    if bare != model:
        candidates.append(f"{_VERTEX_PREFIX}{model}")
    candidates.extend(f"{prefix}{model}" for prefix in PRICE_LOOKUP_PROVIDER_PREFIXES)
    # Unwrapped forms come after the prefixed ones so existing precedence is
    # unchanged for names that already resolved.
    candidates.extend(unwrapped_model_forms(model))
    if bare != model:
        candidates.append(bare)
        candidates.extend(f"{prefix}{bare}" for prefix in PRICE_LOOKUP_PROVIDER_PREFIXES)
        candidates.extend(unwrapped_model_forms(bare))
        candidates.append(f"{_VERTEX_PREFIX}{bare}")
    candidates.extend(
        alias for candidate in tuple(candidates) if (alias := MODEL_ALIASES.get(candidate))
    )
    return tuple(dict.fromkeys(candidates))


def resolve_litellm_model_name(
    model: str,
    is_known_model: Callable[[str], bool],
) -> str:
    """Resolve ``model`` to the first candidate accepted by LiteLLM."""
    for candidate in resolution_candidates(model):
        if is_known_model(candidate):
            return candidate
    return model
