"""Modular prompt construction for the extraction pipeline.

The prompt is a `ChatPromptTemplate`, never an f-string. LangChain templates
own the variable interpolation, which keeps the prompt testable and reusable
and prevents user text from being spliced into the instructions at runtime.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# System instructions: role, contract, and guardrails against hallucination.
# NOTE: `ChatPromptTemplate` treats every `{...}` as a variable and every
# `{{...}}` as a literal brace. The JSON example below is escaped on purpose;
# otherwise LangChain would try to resolve `"tecnologias"` as an input variable
# and formatting would fail with a KeyError.
SYSTEM_TEMPLATE: str = """You are a senior software architect who extracts structured technical entities from raw text.

You must return exactly these three fields:
- tecnologias: a non-empty list of technology names mentioned or clearly implied by the text.
- nivel_de_criticidad: one of "baja", "media", or "alta".
- resumen_tecnico: a concise technical summary in Spanish between 20 and 600 characters.

Rules:
1. Extract ONLY what is supported by the text. Never invent technologies that are not present or clearly implied.
2. The criticality must be justified by evidence in the text. Explicit signals such as "outage", "data loss", "500 errors", "connection pool exhausted", or "production down" justify "alta". Minor or cosmetic issues justify "baja".
3. If the text is ambiguous or contains too little technical evidence, choose "media" and say so in the summary.
4. Returning an empty list is NOT allowed. If a technology is explicitly named, include it. If none is named, do not fabricate one; explain the limitation in the summary instead.

The expected shape is:
{{
  "tecnologias": ["FastAPI", "Redis"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "Short technical summary in Spanish."
}}"""

HUMAN_TEMPLATE: str = """Output contract:
{schema_hint}

Text to analyze:
{text}

Return the structured object now."""


def build_extraction_prompt() -> ChatPromptTemplate:
    """Build a fresh, deterministic extraction prompt.

    Pure function: no I/O, no environment reads, no mutable global state. A
    new template is returned on every call so callers cannot accidentally
    mutate a shared instance.
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            ("human", HUMAN_TEMPLATE),
        ]
    )
