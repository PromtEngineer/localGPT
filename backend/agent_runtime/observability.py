from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_SECRET_KEYS = {
    "authorization",
    "api_key",
    "token",
    "access_token",
    "password",
    "secret",
    "connection_string",
    "url_with_credentials",
}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in _SECRET_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    if isinstance(value, str):
        cleaned = re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*",
            "Bearer [REDACTED]",
            value,
        )
        return re.sub(
            r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@",
            r"\1[REDACTED]@",
            cleaned,
        )
    return value


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id.get(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = redact(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    if os.getenv("LOCALGPT_STRUCTURED_LOGS", "true").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return
    root = logging.getLogger()
    if any(isinstance(handler.formatter, JsonLogFormatter) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    root.handlers = [handler]
    root.setLevel(os.getenv("LOCALGPT_LOG_LEVEL", "INFO").upper())


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    logger = logging.getLogger("localgpt.telemetry")
    started = time.monotonic()
    otel_span = None
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("localgpt")
        otel_span = tracer.start_span(name, attributes=redact(attributes))
    except ImportError:
        pass
    logger.info("span.started", extra={"fields": {"name": name, **attributes}})
    try:
        yield
    except Exception as exc:
        if otel_span is not None:
            otel_span.record_exception(exc)
        logger.exception(
            "span.failed", extra={"fields": {"name": name, **attributes}}
        )
        raise
    finally:
        duration_ms = (time.monotonic() - started) * 1000
        if otel_span is not None:
            otel_span.set_attribute("duration_ms", duration_ms)
            otel_span.end()
        logger.info(
            "span.completed",
            extra={"fields": {"name": name, "duration_ms": duration_ms, **attributes}},
        )
