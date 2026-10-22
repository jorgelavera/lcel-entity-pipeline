"""LCEL extraction chain with observable retries and finish_reason checks.

The chain is intentionally small:

    prompt | model.with_structured_output(TechnicalEntities, include_raw=True)

The surrounding orchestration adds validation, finish_reason inspection, and
explicit retry accounting. See `process_text` for the full flow.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pipeline.errors import (
    EmptyInputError,
    PayloadIncompleteError,
    PipelineError,
    StructuredOutputError,
)
from pipeline.models import build_structured_model, get_model
from pipeline.prompts import build_extraction_prompt
from pipeline.schemas import SCHEMA_HINT, TechnicalEntities

logger = logging.getLogger(__name__)

TOKEN_LIMIT_REASONS: frozenset[str] = frozenset({"length", "max_tokens"})


def _extract_finish_reason(raw: Any) -> str | None:
    """Read `finish_reason` defensively from the raw message metadata.

    Providers expose it in different places: `finish_reason` for OpenAI,
    `stop_reason` for Anthropic, and sometimes nested under a
    `response_metadata` key. It may also be absent entirely.
    """

    metadata = getattr(raw, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None

    value = metadata.get("finish_reason") or metadata.get("stop_reason")

    nested = metadata.get("response_metadata")
    if value is None and isinstance(nested, dict):
        value = nested.get("finish_reason") or nested.get("stop_reason")

    return value if isinstance(value, str) else None


def _extract_partial_text(raw: Any) -> str | None:
    """Best-effort extraction of the partial content for diagnostics."""

    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        joined = "".join(parts)
        return joined or None
    return None


def _resolve_provider_name(provider: str | None) -> str:
    raw = provider or os.environ.get("LLM_PROVIDER") or "openai"
    return raw.strip().lower() or "openai"


def _describe_model(model: Any) -> str:
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


async def process_text(
    text: str,
    *,
    provider: str | None = None,
    model: Any = None,
    schema: type[TechnicalEntities] = TechnicalEntities,
    max_attempts: int = 3,
    logger: logging.Logger | None = None,
) -> TechnicalEntities:
    """Extract validated technical entities from raw text.

    Equivalent LCEL composition (the exact form required by the assignment):

        prompt = build_extraction_prompt()
        structured = model.with_structured_output(TechnicalEntities, include_raw=True)
        chain = prompt | structured

    Here the structured runnable is built through
    `pipeline.models.build_structured_model` so the retry policy is attached
    and tests can replace it with a fake.

    Two retry layers coexist on purpose:

    1. `.with_retry(...)` inside the structured runnable is the LCEL-level
       policy. It retries errors raised while the runnable executes.
    2. The explicit loop below provides observable, testable retry accounting
       (`retry n/max: reason` warnings). It is also the only layer that can
       retry failures detected AFTER the runnable returns, because
       `finish_reason` inspection and payload-shape validation happen outside
       the runnable.

    The outer loop never exceeds `max_attempts`.
    """

    active_logger = logger or globals()["logger"]

    if not isinstance(text, str) or not text.strip():
        raise EmptyInputError()

    attempts = max(1, int(max_attempts))
    provider_name = _resolve_provider_name(provider)

    active_logger.info(
        "Starting technical entity extraction (provider=%s, text_length=%d, max_attempts=%d)",
        provider_name,
        len(text),
        attempts,
    )

    active_model = model if model is not None else get_model(provider)
    active_logger.info(
        "Using provider=%s model=%s temperature=%s",
        provider_name,
        _describe_model(active_model),
        getattr(active_model, "temperature", "default"),
    )

    prompt = build_extraction_prompt()

    # Resolved through the module global ON PURPOSE: tests monkeypatch
    # `pipeline.chain.build_structured_model`, and an import-time local alias
    # would defeat that seam.
    #
    # The runnable gets `max_inner_attempts=1` on purpose. The retry policy
    # still exists (`.with_retry(...)` is applied inside the builder, as the
    # assignment requires), but it must NOT multiply with the outer loop:
    # nesting `attempts` outer iterations around a runnable that itself retries
    # `attempts` times yields attempts**2 API calls. With the default of 3 that
    # is 9 real requests for a user who asked for 3 — a paid, rate-limited
    # surprise. The outer loop is the single owner of the attempt budget, and
    # it also covers the post-validation failures (finish_reason inspection,
    # payload shape) that happen after the runnable has already returned.
    structured_runnable = build_structured_model(active_model, schema, max_inner_attempts=1)
    chain = prompt | structured_runnable

    last_error: PipelineError | None = None

    for attempt in range(1, attempts + 1):
        try:
            result = await chain.ainvoke({"text": text, "schema_hint": SCHEMA_HINT})

            if not isinstance(result, dict):
                raise StructuredOutputError(
                    "Structured output returned an unexpected envelope.",
                    detail=f"Expected a dict with raw/parsed/parsing_error, got {type(result).__name__}.",
                )

            raw = result.get("raw")
            parsed = result.get("parsed")
            parsing_error = result.get("parsing_error")

            finish_reason = _extract_finish_reason(raw)
            active_logger.debug("Resolved finish_reason=%s", finish_reason)

            # Detect truncation BEFORE touching `parsed`. A truncated payload
            # is incomplete by definition, even if a partial object parsed.
            if finish_reason in TOKEN_LIMIT_REASONS:
                raise PayloadIncompleteError(
                    "The model stopped at a token limit before finishing the payload.",
                    finish_reason=finish_reason,
                    partial_text=_extract_partial_text(raw),
                )

            if parsed is None or parsing_error is not None:
                raise StructuredOutputError(
                    "The model did not return a valid structured object.",
                    parsing_error=parsing_error if isinstance(parsing_error, Exception) else None,
                    detail=(
                        f"parsed_is_none={parsed is None}, "
                        f"parsing_error={parsing_error!r}"
                    ),
                )

            technologies = getattr(parsed, "tecnologias", [])
            criticality = getattr(parsed, "nivel_de_criticidad", None)
            active_logger.info(
                "Validation succeeded: technologies=%d criticality=%s",
                len(technologies) if isinstance(technologies, list) else 0,
                getattr(criticality, "value", criticality),
            )

            return parsed  # type: ignore[return-value]

        except (PayloadIncompleteError, StructuredOutputError) as exc:
            last_error = exc
            if attempt >= attempts:
                if isinstance(exc, PayloadIncompleteError):
                    active_logger.error(
                        "Incomplete payload detected after %d attempt(s) "
                        "(finish_reason=%s): %s",
                        attempts,
                        exc.finish_reason,
                        exc,
                    )
                else:
                    active_logger.error(
                        "Structured output failure after %d attempt(s) "
                        "(parsing_error=%r): %s",
                        attempts,
                        exc.parsing_error,
                        exc,
                    )
                raise
            active_logger.warning(
                "Retry %d/%d: %s",
                attempt,
                attempts,
                exc,
            )

    # Defensive: the loop always returns or raises.
    raise last_error or StructuredOutputError("Structured extraction failed.")


async def process_text_safe(
    text: str,
    **kwargs: Any,
) -> tuple[TechnicalEntities | None, PipelineError | None]:
    """Run `process_text` without raising controlled `PipelineError`s.

    Returns `(result, None)` on success or `(None, error)` on a controlled
    failure. Unexpected exceptions are wrapped into `PipelineError` so callers
    always receive a predictable tuple.
    """

    active_logger = kwargs.get("logger") or logger
    try:
        result = await process_text(text, **kwargs)
        return result, None
    except PipelineError as exc:
        return None, exc
    except Exception as exc:  # pragma: no cover - last-resort safety net
        active_logger.exception("Unexpected failure while processing text.")
        return None, PipelineError(
            "Unexpected failure while processing text.",
            detail=repr(exc),
        )


def describe_result(result: TechnicalEntities) -> str:
    """Render a readable multi-line summary for CLI output."""

    technologies = ", ".join(result.tecnologias)
    criticality = getattr(result.nivel_de_criticidad, "value", result.nivel_de_criticidad)
    return (
        "Technical entities\n"
        f"  Technologies : {technologies}\n"
        f"  Criticality  : {criticality}\n"
        f"  Summary      : {result.resumen_tecnico}"
    )
