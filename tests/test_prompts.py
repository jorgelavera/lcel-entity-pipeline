"""Tests for the modular ChatPromptTemplate."""

from langchain_core.prompts import ChatPromptTemplate

from pipeline.prompts import build_extraction_prompt
from pipeline.schemas import SCHEMA_HINT


def test_prompt_is_a_chat_prompt_template_with_expected_variables() -> None:
    prompt = build_extraction_prompt()

    assert isinstance(prompt, ChatPromptTemplate)
    assert "text" in prompt.input_variables
    assert "schema_hint" in prompt.input_variables


def test_prompt_formatting_does_not_raise_on_literal_braces() -> None:
    prompt = build_extraction_prompt()

    messages = prompt.format_messages(
        text="FastAPI y Redis con errores 500.",
        schema_hint=SCHEMA_HINT,
    )

    assert len(messages) == 2
    assert "FastAPI y Redis con errores 500." in messages[1].content
    assert SCHEMA_HINT in messages[1].content


def test_system_message_mentions_all_three_field_names() -> None:
    prompt = build_extraction_prompt()

    system_content = prompt.format_messages(
        text="Texto de ejemplo.",
        schema_hint=SCHEMA_HINT,
    )[0].content

    assert "tecnologias" in system_content
    assert "nivel_de_criticidad" in system_content
    assert "resumen_tecnico" in system_content
