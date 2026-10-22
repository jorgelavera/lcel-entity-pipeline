"""Domain errors for the extraction pipeline.

Every error states when it is raised and whether it is retryable, so callers
can decide between retrying, fixing configuration, or surfacing a controlled
message to the user.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for every controlled failure in the pipeline.

    Raised for configuration problems (missing API key, unknown provider,
    missing optional package) and for any unexpected exception wrapped by
    `process_text_safe`. Not retryable by itself: the caller must fix the
    configuration or the input.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} ({self.detail})"
        return self.message


class PayloadIncompleteError(PipelineError):
    """Raised when the model stopped because it hit a token limit.

    Detected when `finish_reason` is `"length"` (OpenAI) or `"max_tokens"`
    (Anthropic). This means the JSON object was truncated, so it must be
    detected BEFORE any attempt to transform the parsed payload.

    Retryable: yes, because a new attempt may produce a shorter, complete
    object. The pipeline retries it and logs every attempt.
    """

    def __init__(
        self,
        message: str,
        *,
        finish_reason: str,
        partial_text: str | None = None,
        detail: str | None = None,
    ) -> None:
        self.finish_reason = finish_reason
        self.partial_text = partial_text
        super().__init__(message, detail=detail)


class StructuredOutputError(PipelineError):
    """Raised when structured output could not be parsed after all retries.

    Triggered when `parsed` is `None` and/or `parsing_error` is not `None`.
    The underlying parsing exception is carried in `parsing_error` for
    diagnostics.

    Retryable: yes, because malformed or incomplete JSON is often transient
    (temperature, truncation, provider hiccups). The pipeline retries it up to
    `max_attempts` and logs a warning per retry.
    """

    def __init__(
        self,
        message: str,
        *,
        parsing_error: Exception | None = None,
        detail: str | None = None,
    ) -> None:
        self.parsing_error = parsing_error
        super().__init__(message, detail=detail)


class EmptyInputError(PipelineError):
    """Raised when the user provides blank or whitespace-only text.

    Not retryable: retrying the same empty input can never succeed. The caller
    must provide actual content.
    """

    def __init__(self, message: str = "Input text must not be empty or whitespace-only.") -> None:
        super().__init__(message)
