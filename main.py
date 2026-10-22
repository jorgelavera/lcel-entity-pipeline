"""Command-line entry point for the technical entity extraction pipeline.

The CLI is a thin adapter: all extraction logic lives in `pipeline.chain`.
It is designed to fail gracefully when credentials are missing, because the
most common first run happens before a `.env` file exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import traceback
from typing import Sequence

from dotenv import load_dotenv

from pipeline.chain import describe_result, process_text, process_text_safe
from pipeline.errors import PipelineError
from pipeline.models import get_model

SAMPLE_ARCHITECTURE_TEXT = """La plataforma de pedidos está construida con una API REST en FastAPI
que atiende a unos 4.000 usuarios concurrentes. Las sesiones y los catálogos
más consultados se cachean en Redis, mientras que los pedidos, pagos y el
historial de envíos se persisten en PostgreSQL. Durante el último pico de
ventas observamos que el pool de conexiones de PostgreSQL se agota, lo que
provoca errores 500 intermitentes, timeouts en la pasarela de pagos y caídas
parciales del servicio. El equipo sospecha además que algunas transacciones
quedaron en estado inconsistente y existe riesgo de pérdida de datos en los
pedidos que no llegaron a confirmarse. La API está detrás de Nginx y se
despliega en contenedores Docker sobre Kubernetes."""

SAMPLE_AMBIGUOUS_TEXT = """Hola, el sistema anda más o menos. A veces responde lento
y no sabemos bien por qué, pero no hay mayores detalles por ahora."""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""

    parser = argparse.ArgumentParser(
        description="Extract validated technical entities from text using LangChain LCEL.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic"),
        default=None,
        help="LLM provider. Defaults to LLM_PROVIDER or openai.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the provider's default model name.",
    )
    parser.add_argument(
        "--text",
        default=SAMPLE_ARCHITECTURE_TEXT,
        help="Text to analyze. Defaults to the built-in architecture sample.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING"),
        default="INFO",
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of extraction attempts (default: 3).",
    )
    return parser


def _print_error_block(error: PipelineError, provider: str | None) -> None:
    """Print a controlled, readable error block instead of a traceback."""

    print("\n" + "=" * 62)
    print("CONTROLLED PIPELINE ERROR")
    print("=" * 62)
    print(f"Provider : {provider or 'auto'}")
    print(f"Type     : {type(error).__name__}")
    print(f"Message  : {error.message}")
    if error.detail:
        print(f"Detail   : {error.detail}")
    print("Hint     : Check .env (copy .env.example) and your provider credentials.")
    print("=" * 62)


def _print_json(result: object) -> None:
    """Print a Pydantic model as pretty JSON."""

    payload = result.model_dump()  # type: ignore[attr-defined]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


async def main(argv: Sequence[str] | None = None) -> int:
    """Run both sample texts and return a process exit code."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    print("Technical Entity Extraction Pipeline")
    print("LangChain LCEL + structured output + validation")
    print("-" * 62)

    try:
        active_model = get_model(args.provider, model_name=args.model)

        result = await process_text(
            args.text,
            provider=args.provider,
            model=active_model,
            max_attempts=args.max_attempts,
        )

        print("\n[1] Validated result for the architecture sample")
        _print_json(result)
        print()
        print(describe_result(result))

        print("\n[2] Stress test with the ambiguous sample")
        safe_result, safe_error = await process_text_safe(
            SAMPLE_AMBIGUOUS_TEXT,
            provider=args.provider,
            model=active_model,
            max_attempts=args.max_attempts,
        )
        if safe_error is not None:
            _print_error_block(safe_error, args.provider)
        elif safe_result is not None:
            _print_json(safe_result)
            print()
            print(describe_result(safe_result))

    except PipelineError as error:
        _print_error_block(error, args.provider)
        return 0

    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    load_dotenv()

    try:
        exit_code = asyncio.run(main())
    except Exception:
        traceback.print_exc()
        exit_code = 1

    raise SystemExit(exit_code)
