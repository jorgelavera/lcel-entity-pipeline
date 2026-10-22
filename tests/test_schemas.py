"""Contract tests for the Pydantic schema."""

import pytest
from pydantic import ValidationError

from pipeline.schemas import Criticidad, TechnicalEntities

VALID_SUMMARY = "Resumen técnico de prueba con longitud suficiente."


def test_valid_model_accepts_expected_payload() -> None:
    model = TechnicalEntities(
        tecnologias=["FastAPI", "Redis", "PostgreSQL"],
        nivel_de_criticidad="alta",
        resumen_tecnico=VALID_SUMMARY,
    )

    assert model.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert model.nivel_de_criticidad is Criticidad.ALTA
    assert model.resumen_tecnico == VALID_SUMMARY


def test_empty_technologies_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalEntities(
            tecnologias=[],
            nivel_de_criticidad="media",
            resumen_tecnico=VALID_SUMMARY,
        )


def test_empty_string_technology_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalEntities(
            tecnologias=["FastAPI", "   "],
            nivel_de_criticidad="media",
            resumen_tecnico=VALID_SUMMARY,
        )


def test_duplicates_are_removed_case_insensitively_preserving_order() -> None:
    model = TechnicalEntities(
        tecnologias=["FastAPI", "fastapi", " Redis ", "REDIS", "PostgreSQL"],
        nivel_de_criticidad="media",
        resumen_tecnico=VALID_SUMMARY,
    )

    assert model.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]


def test_invalid_criticality_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalEntities(
            tecnologias=["FastAPI"],
            nivel_de_criticidad="critica",
            resumen_tecnico=VALID_SUMMARY,
        )


def test_too_short_summary_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalEntities(
            tecnologias=["FastAPI"],
            nivel_de_criticidad="baja",
            resumen_tecnico="Demasiado corto.",
        )


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        TechnicalEntities(
            tecnologias=["FastAPI"],
            nivel_de_criticidad="baja",
            resumen_tecnico=VALID_SUMMARY,
            proveedor="openai",
        )


def test_whitespace_only_summary_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalEntities(
            tecnologias=["FastAPI"],
            nivel_de_criticidad="baja",
            resumen_tecnico=" " * 40,
        )
