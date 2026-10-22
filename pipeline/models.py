"""Chat model factory and the structured-output retry seam.

Provider SDK imports are lazy: a machine that only installs the OpenAI package
can still use this project, and a missing optional package fails only when the
corresponding provider is actually requested.
"""

from __future__ import annotations

import os
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from pipeline.errors import PipelineError, PayloadIncompleteError, StructuredOutputError

SchemaT = TypeVar("SchemaT", bound=BaseModel)

SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "anthropic")

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-latest"


def _resolve_provider(provider: str | None) -> str:
    """Resolve the provider with explicit argument > env var > default."""

    raw = provider or os.environ.get("LLM_PROVIDER") or "openai"
    return raw.strip().lower()


def _resolve_temperature() -> float:
    """Parse LLM_TEMPERATURE defensively and keep it in a safe range.

    A bad environment value must never crash the pipeline, so invalid input
    silently falls back to the deterministic default of 0.0.
    """

    raw = os.environ.get("LLM_TEMPERATURE")
    if raw is None or not raw.strip():
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return min(max(value, 0.0), 1.0)


def _missing_key_error(provider: str, env_var: str) -> PipelineError:
    return PipelineError(
        f"Missing {env_var} for provider '{provider}'.",
        detail=(
            "Copy .env.example to .env and set a valid key, "
            "or export the variable in your shell."
        ),
    )


def _wrap_import_error(provider: str, package: str, exc: ImportError) -> PipelineError:
    return PipelineError(
        f"Provider '{provider}' requires the '{package}' package.",
        detail=f"Install it with: uv pip install {package} (original error: {exc})",
    )


def get_model(provider: str | None = None, model_name: str | None = None) -> BaseChatModel:
    """Build a chat model for the requested provider.

    Resolution order for the provider is: explicit argument, then
    `LLM_PROVIDER`, then `"openai"`. The optional `model_name` argument lets
    the CLI override the model without mutating the environment.
    """

    resolved = _resolve_provider(provider)
    temperature = _resolve_temperature()

    if resolved == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise _missing_key_error("openai", "OPENAI_API_KEY")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise _wrap_import_error("openai", "langchain-openai", exc) from exc

        return ChatOpenAI(
            model=model_name or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            temperature=temperature,
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )

    if resolved == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise _missing_key_error("anthropic", "ANTHROPIC_API_KEY")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise _wrap_import_error("anthropic", "langchain-anthropic", exc) from exc

        return ChatAnthropic(
            model=model_name or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
            temperature=temperature,
            max_tokens=1024,
            anthropic_api_key=api_key,
        )

    supported = ", ".join(SUPPORTED_PROVIDERS)
    raise PipelineError(
        f"Unknown provider '{resolved}'. Supported providers: {supported}.",
        detail="Set LLM_PROVIDER to one of the supported names or pass --provider.",
    )


def build_structured_model(
    model: BaseChatModel,
    schema: type[BaseModel],
    max_inner_attempts: int = 1,
) -> Any:
    """Attach structured output and a runnable-level retry policy.

    `include_raw=True` is essential: the envelope keeps the raw `AIMessage`
    alongside the parsed object and the parsing error, which is the only way to
    inspect `finish_reason` and detect truncated payloads.

    `max_inner_attempts` defaults to 1. The assignment requires `.with_retry()`
    to be present on the chain, and it is — but the attempt budget is owned by
    the outer loop in `process_text`. Retrying at both levels multiplies the
    real request count (`attempts**2`), so the runnable-level policy must stay
    at a single attempt to keep the observable budget honest.

    This function is a module-level seam so tests can monkeypatch
    `pipeline.chain.build_structured_model` without constructing a real model
    or touching the network.
    """

    structured = model.with_structured_output(schema, include_raw=True)
    return structured.with_retry(
        stop_after_attempt=max(1, max_inner_attempts),
        wait_exponential_jitter=True,
        retry_if_exception_type=(StructuredOutputError, PayloadIncompleteError),
    )
