"""Offline tests for the LCEL chain, using fake structured runnables."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

import pipeline.chain as chain_module
from pipeline.errors import (
    EmptyInputError,
    PayloadIncompleteError,
    StructuredOutputError,
)
from pipeline.schemas import TechnicalEntities

VALID_SUMMARY = "Resumen técnico de prueba con longitud suficiente."


class FakeStructuredRunnable(Runnable):
    """Deterministic stand-in for `model.with_structured_output(...).with_retry(...)`.

    Each outcome is either an envelope dict or an exception instance. Calls are
    counted so retry accounting can be asserted precisely.
    """

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # noqa: A002
        return self._next()

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:  # noqa: A002
        return self._next()

    def _next(self) -> Any:
        self.calls += 1
        index = min(self.calls - 1, len(self.outcomes) - 1)
        outcome = self.outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ExplodingParsed:
    """Fails the test if the chain touches parsed data before checking finish_reason."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"Parsed payload was accessed too early: {name}")


def make_entities() -> TechnicalEntities:
    return TechnicalEntities(
        tecnologias=["FastAPI", "Redis", "PostgreSQL"],
        nivel_de_criticidad="alta",
        resumen_tecnico=VALID_SUMMARY,
    )


def make_envelope(
    parsed: Any,
    *,
    finish_reason: str | None = "stop",
    parsing_error: Exception | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason
    raw = AIMessage(content='{"tecnologias": []}', response_metadata=metadata)
    return {"raw": raw, "parsed": parsed, "parsing_error": parsing_error}


def install_fake(monkeypatch: pytest.MonkeyPatch, fake: FakeStructuredRunnable) -> dict[str, int]:
    """Replace the module-level builder seam used by `process_text`."""

    calls = {"count": 0, "max_inner_attempts": []}

    def builder(model: Any, schema: Any, *, max_inner_attempts: int = 1) -> FakeStructuredRunnable:
        calls["count"] += 1
        calls["max_inner_attempts"].append(max_inner_attempts)
        return fake

    monkeypatch.setattr(chain_module, "build_structured_model", builder)
    return calls


async def test_happy_path_returns_validated_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeStructuredRunnable([make_envelope(make_entities())])
    install_fake(monkeypatch, fake)

    result = await chain_module.process_text("FastAPI with Redis.", model=object())

    assert isinstance(result, TechnicalEntities)
    assert result.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert fake.calls == 1


async def test_token_limit_is_detected_before_touching_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeStructuredRunnable(
        [make_envelope(ExplodingParsed(), finish_reason="length")]
    )
    install_fake(monkeypatch, fake)

    with pytest.raises(PayloadIncompleteError) as exc_info:
        await chain_module.process_text("FastAPI with Redis.", model=object(), max_attempts=1)

    assert exc_info.value.finish_reason == "length"
    assert fake.calls == 1


async def test_missing_parsed_with_parsing_error_raises_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsing_error = ValueError("malformed JSON")
    fake = FakeStructuredRunnable([make_envelope(None, parsing_error=parsing_error)])
    install_fake(monkeypatch, fake)

    with pytest.raises(StructuredOutputError) as exc_info:
        await chain_module.process_text("FastAPI with Redis.", model=object(), max_attempts=1)

    assert exc_info.value.parsing_error is parsing_error


async def test_retry_recovers_and_emits_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeStructuredRunnable(
        [
            StructuredOutputError("bad payload"),
            StructuredOutputError("bad payload again"),
            make_envelope(make_entities()),
        ]
    )
    install_fake(monkeypatch, fake)
    caplog.set_level(logging.WARNING, logger="pipeline.chain")

    result = await chain_module.process_text("FastAPI with Redis.", model=object(), max_attempts=3)

    assert isinstance(result, TechnicalEntities)
    assert fake.calls == 3
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("Retry 1/3" in message for message in warnings)
    assert any("Retry 2/3" in message for message in warnings)


async def test_retry_exhaustion_raises_after_exact_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeStructuredRunnable([StructuredOutputError("always bad")])
    install_fake(monkeypatch, fake)

    with pytest.raises(StructuredOutputError):
        await chain_module.process_text("FastAPI with Redis.", model=object(), max_attempts=3)

    assert fake.calls == 3


async def test_blank_input_raises_before_building_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeStructuredRunnable([make_envelope(make_entities())])
    calls = install_fake(monkeypatch, fake)

    with pytest.raises(EmptyInputError):
        await chain_module.process_text("   \n\t ", model=object())

    assert calls["count"] == 0
    assert fake.calls == 0


async def test_safe_variant_returns_error_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeStructuredRunnable([StructuredOutputError("bad payload")])
    install_fake(monkeypatch, fake)

    result, error = await chain_module.process_text_safe(
        "FastAPI with Redis.",
        model=object(),
        max_attempts=1,
    )

    assert result is None
    assert isinstance(error, StructuredOutputError)


async def test_inner_runnable_never_owns_more_than_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inner retry budget must stay at 1 to avoid attempts**2 API calls.

    Regression guard: the outer loop in `process_text` owns the attempt
    budget. If the inner runnable is also built with `max_attempts`, a caller
    asking for 3 attempts would trigger 9 real requests.
    """

    fake = FakeStructuredRunnable([make_envelope(make_entities())])
    calls = install_fake(monkeypatch, fake)

    await chain_module.process_text("FastAPI with Redis.", model=object(), max_attempts=5)

    assert calls["max_inner_attempts"] == [1], (
        "inner runnable must be built with a single attempt; got "
        f"{calls['max_inner_attempts']}"
    )
    assert fake.calls == 1


async def test_outer_loop_catches_runnable_level_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceptions raised inside the runnable must still be retried and bounded.

    The outer loop is the only layer that can account for these because the
    inner runnable is deliberately built with a single attempt.
    """

    fake = FakeStructuredRunnable(
        [StructuredOutputError("transient failure"), make_envelope(make_entities())]
    )
    install_fake(monkeypatch, fake)

    result = await chain_module.process_text("FastAPI with Redis.", model=object(), max_attempts=3)

    assert isinstance(result, TechnicalEntities)
    assert fake.calls == 2, "one failure plus one success, bounded by max_attempts"


async def test_total_calls_never_exceed_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The observable budget is exactly `max_attempts`, never attempts**2."""

    fake = FakeStructuredRunnable([StructuredOutputError("always bad")])
    install_fake(monkeypatch, fake)

    with pytest.raises(StructuredOutputError):
        await chain_module.process_text("FastAPI with Redis.", model=object(), max_attempts=4)

    assert fake.calls == 4
