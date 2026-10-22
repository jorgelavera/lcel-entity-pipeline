# Technical Entity Extraction Pipeline

A validated extraction pipeline built with LangChain LCEL. It reads a raw
architecture description or error log and returns a Pydantic object with
`tecnologias`, `nivel_de_criticidad`, and `resumen_tecnico`. This is a course
deliverable (pre-entrega 2): clarity, focused modules, and explicit types.

## What it is

The pipeline turns unstructured text into a contract-safe object. A modular
`ChatPromptTemplate` defines the rules, the model produces structured output
against a Pydantic schema, validators reject weak payloads, and a retry layer
recovers from malformed responses. Spanish field names are mandated; all
identifiers, comments, and docs are English.

## How it works

```text
ChatPromptTemplate
        |
        v
prompt | model.with_structured_output(Schema, include_raw=True) | .with_retry(...)
        |
        v
finish_reason inspection -> parsed validation -> TechnicalEntities
```

The code composes `prompt | structured_runnable`, where the runnable is built
by `pipeline.models.build_structured_model`:

```python
structured = model.with_structured_output(TechnicalEntities, include_raw=True)
chain = build_extraction_prompt() | structured.with_retry(stop_after_attempt=max_attempts)
result = await chain.ainvoke({"text": text, "schema_hint": SCHEMA_HINT})
```

`include_raw=True` is what makes resilience possible: the envelope keeps the
raw `AIMessage`, and its `response_metadata` carries `finish_reason` (`"stop"`,
`"length"`, `"max_tokens"`). A value of `length` or `max_tokens` means the JSON
was truncated before it could be validated.

## The output contract

| Field | Type | Rules |
| --- | --- | --- |
| `tecnologias` | `list[str]` | At least one item. Items are stripped, blanks rejected, and duplicates removed case-insensitively while preserving first-seen order and original casing. |
| `nivel_de_criticidad` | `Criticidad` | One of `baja`, `media`, `alta`. |
| `resumen_tecnico` | `str` | 20 to 600 characters, not whitespace-only. |

Extra fields are forbidden. The assignment's expected output shape:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API en FastAPI usa Redis como caché y PostgreSQL para persistencia; el agotamiento del pool de conexiones provoca errores 500 y riesgo de pérdida de datos."
}
```

## Requirements

Python 3.12, `uv` (recommended) or `pip`, and an API key for OpenAI or Anthropic.

## Setup

With `uv`:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
Copy-Item .env.example .env
```

With plain `pip`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Environment variables

| Variable | Required | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | No | `openai` or `anthropic`. Defaults to `openai`. |
| `OPENAI_API_KEY` | For OpenAI | OpenAI API key. |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o-mini`. |
| `OPENAI_BASE_URL` | No | Optional OpenAI-compatible gateway. |
| `ANTHROPIC_API_KEY` | For Anthropic | Anthropic API key. |
| `ANTHROPIC_MODEL` | No | Defaults to `claude-3-5-haiku-latest`. |
| `LLM_TEMPERATURE` | No | Defaults to `0`. Invalid values fall back to `0`. |

## Usage

As a library:

```python
import asyncio
from pipeline.chain import process_text
result = asyncio.run(process_text("FastAPI con Redis y PostgreSQL sufrió errores 500."))
print(result.nivel_de_criticidad, result.tecnologias)
```

As a CLI with OpenAI:

```powershell
.\.venv\Scripts\python.exe main.py --provider openai --text "La API en FastAPI se cayó por un outage de PostgreSQL."
```

As a CLI with Anthropic:

```powershell
.\.venv\Scripts\python.exe main.py --provider anthropic --model claude-3-5-haiku-latest
```

Run `main.py --help` for all options, or set `--log-level DEBUG --max-attempts 3`.
Without credentials the CLI prints a controlled configuration error and exits
successfully, so the project is safe to demo before setup.

## Resilience model

| Error | When it is raised | Retryable |
| --- | --- | --- |
| `PayloadIncompleteError` | `finish_reason` is `length` or `max_tokens`. The payload was truncated. | Yes |
| `StructuredOutputError` | `parsed` is `None` and/or `parsing_error` is present after the attempts. | Yes |
| `EmptyInputError` | The input is blank or whitespace-only. | No |
| `PipelineError` | Configuration problems or unexpected wrapped failures. | No |

Retries exist at two levels:

1. The structured runnable uses `.with_retry(...)`, satisfying the
   LCEL-level policy and retrying errors raised inside the runnable.
2. `process_text` also runs an explicit attempt loop. This produces observable
   `retry n/max: reason` warnings and covers failures detected after the
   runnable returns, such as `finish_reason` inspection and envelope validation.

The outer loop never exceeds `max_attempts`; the layers are independent, so the
count is per-layer when both are active. Logging covers start, provider/model,
`finish_reason` at DEBUG, retry warnings, success, and terminal errors.

## Stress test

`main.py` runs a realistic sample and a deliberately ambiguous one:

- `SAMPLE_ARCHITECTURE_TEXT`: FastAPI, Redis, and PostgreSQL with an exhausted
  pool, intermittent 500s, and potential data loss, justifying `alta`.
- `SAMPLE_AMBIGUOUS_TEXT`: vague, exercising the `media` path through
  `process_text_safe`, so a controlled failure prints instead of crashing.

## Testing

All tests are offline and use fake structured runnables.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Covers schema validation, prompts, retry accounting, token-limit detection,
blank input, and the safe variant.

## Design decisions

- **ChatPromptTemplate over f-strings.** LangChain owns interpolation, keeping
  prompts testable and preventing accidental splicing of untrusted text.
- **`include_raw=True`.** It is the only way to read `finish_reason`;
  structured output alone hides truncation.
- **Two retry layers.** `.with_retry` covers the runnable; the explicit loop
  adds accounting and covers post-runnable checks.
- **Strict validation.** `extra="forbid"` and custom validators fail
  off-contract output fast.
- **Lazy imports.** Missing optional packages break only their provider, with an
  install hint.

## Known limitations

- Detection needs `finish_reason`/`stop_reason`; otherwise truncation surfaces
  as a parsing error.
- Both retry layers can multiply attempts; keep `max_attempts` modest.
- No streaming or memory; extraction is single-turn.
- Spanish summaries are required; changing that means updating schema, prompt,
  and tests together.
- Tests verify orchestration, not live provider parsing.
