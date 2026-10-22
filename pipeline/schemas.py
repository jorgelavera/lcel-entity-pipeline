"""Pydantic contracts for the technical entity extraction pipeline.

The field names intentionally stay in Spanish because the assignment mandates
them. Everything else (classes, helpers, comments) is in English.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Criticidad(str, Enum):
    """Allowed criticality levels, constrained by the assignment."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class TechnicalEntities(BaseModel):
    """Validated representation of the technical entities found in a text.

    `extra="forbid"` is deliberate: if the model hallucinates extra keys the
    pipeline should fail loudly instead of silently accepting an off-contract
    payload. `str_strip_whitespace` removes accidental padding before the
    validators below run.
    """

    tecnologias: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of technology names mentioned or implied in the text. "
            "Must contain at least one non-empty item."
        ),
    )
    nivel_de_criticidad: Criticidad = Field(
        ...,
        description="Criticality level of the situation described: baja, media or alta.",
    )
    resumen_tecnico: str = Field(
        ...,
        min_length=20,
        max_length=600,
        description="A concise technical summary in Spanish, 20-600 characters.",
    )

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("tecnologias")
    @classmethod
    def normalize_technologies(cls, value: list[str]) -> list[str]:
        """Strip, reject blanks, and de-duplicate case-insensitively.

        De-duplication preserves the first-seen order and the original casing,
        which keeps the output readable while still removing repeated mentions
        of the same technology (e.g. "FastAPI" and "fastapi").
        """

        cleaned: list[str] = []
        seen: set[str] = set()

        for item in value:
            if not isinstance(item, str):
                raise ValueError("Each technology entry must be a string.")
            stripped = item.strip()
            if not stripped:
                raise ValueError("Technology entries must not be empty or whitespace-only.")
            key = stripped.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(stripped)

        if not cleaned:
            raise ValueError("The 'tecnologias' list must contain at least one technology.")

        return cleaned

    @field_validator("resumen_tecnico")
    @classmethod
    def reject_blank_summary(cls, value: str) -> str:
        """Reject summaries that only contain whitespace."""

        if not value.strip():
            raise ValueError("The technical summary must not be empty or whitespace-only.")
        return value


# Short human-readable contract used by the prompt. This is intentionally NOT
# a JSON schema dump: the LLM only needs a compact description of the fields.
SCHEMA_HINT: str = (
    'Return a JSON object with exactly three fields: '
    '"tecnologias" (list of technology names, at least one), '
    '"nivel_de_criticidad" (one of "baja", "media", "alta"), and '
    '"resumen_tecnico" (a 20-600 character technical summary in Spanish).'
)
