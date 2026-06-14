"""
Structured Logging System for LocalGPT

Provides JSON-formatted logging with structured data, correlation IDs,
performance metrics, and comprehensive observability features.
"""

import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

# Thread-local storage for correlation IDs
_local = threading.local()


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with a Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StructuredLogger:
    """
    Structured logger that outputs JSON-formatted logs with consistent fields.
    Supports correlation IDs, performance timing, and structured data.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add JSON handler
        json_handler = JSONLogHandler()
        json_handler.setLevel(level)
        self.logger.addHandler(json_handler)
        self.logger.propagate = False  # Don't propagate to root logger

    def _log(self, level: int, event: str, **kwargs):
        """Internal logging method with structured data"""
        # Add standard fields
        log_data = {
            "timestamp": _utc_timestamp(),
            "level": logging.getLevelName(level),
            "logger": self.name,
            "event": event,
            "correlation_id": getattr(_local, "correlation_id", None),
            "thread_id": threading.get_ident(),
        }

        # Add custom fields
        log_data.update(kwargs)

        # Remove None values for cleaner logs
        log_data = {k: v for k, v in log_data.items() if v is not None}

        self.logger.log(level, json.dumps(log_data, default=str))

    def debug(self, event: str, **kwargs):
        """Log debug event"""
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs):
        """Log info event"""
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs):
        """Log warning event"""
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs):
        """Log error event"""
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs):
        """Log critical event"""
        self._log(logging.CRITICAL, event, **kwargs)

    def performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metric"""
        self._log(
            logging.INFO,
            "performance",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            **kwargs,
        )

    def request_start(self, method: str, path: str, **kwargs):
        """Log request start"""
        self._log(logging.INFO, "request_start", method=method, path=path, **kwargs)

    def request_end(
        self, method: str, path: str, status_code: int, duration_ms: float, **kwargs
    ):
        """Log request end"""
        level = (
            logging.INFO
            if status_code < 400
            else logging.WARNING if status_code < 500 else logging.ERROR
        )
        self._log(
            level,
            "request_end",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            **kwargs,
        )

    def indexing_start(self, index_id: str, file_count: int, **kwargs):
        """Log indexing operation start"""
        self._log(
            logging.INFO,
            "indexing_start",
            index_id=index_id,
            file_count=file_count,
            **kwargs,
        )

    def indexing_progress(
        self,
        index_id: str,
        processed: int,
        total: int,
        current_file: Optional[str] = None,
        **kwargs,
    ):
        """Log indexing progress"""
        self._log(
            logging.INFO,
            "indexing_progress",
            index_id=index_id,
            processed=processed,
            total=total,
            current_file=current_file,
            **kwargs,
        )

    def indexing_complete(
        self,
        index_id: str,
        total_files: int,
        total_chunks: int,
        duration_ms: float,
        **kwargs,
    ):
        """Log indexing completion"""
        self._log(
            logging.INFO,
            "indexing_complete",
            index_id=index_id,
            total_files=total_files,
            total_chunks=total_chunks,
            duration_ms=round(duration_ms, 2),
            **kwargs,
        )

    def query_start(
        self, query_length: int, session_id: Optional[str] = None, **kwargs
    ):
        """Log query start"""
        self._log(
            logging.INFO,
            "query_start",
            query_length=query_length,
            session_id=session_id,
            **kwargs,
        )

    def query_end(
        self,
        query_length: int,
        result_count: int,
        duration_ms: float,
        cache_hit: bool = False,
        **kwargs,
    ):
        """Log query end"""
        self._log(
            logging.INFO,
            "query_end",
            query_length=query_length,
            result_count=result_count,
            duration_ms=round(duration_ms, 2),
            cache_hit=cache_hit,
            **kwargs,
        )

    def cache_hit(self, cache_type: str, key: str, **kwargs):
        """Log cache hit"""
        self._log(logging.DEBUG, "cache_hit", cache_type=cache_type, key=key, **kwargs)

    def cache_miss(self, cache_type: str, key: str, **kwargs):
        """Log cache miss"""
        self._log(logging.DEBUG, "cache_miss", cache_type=cache_type, key=key, **kwargs)

    def error_with_context(self, error: Exception, operation: str, **kwargs):
        """Log error with full context"""
        self._log(
            logging.ERROR,
            "error",
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs,
        )


class JSONLogHandler(logging.Handler):
    """Custom log handler that outputs structured JSON"""

    def __init__(self, stream=None):
        super().__init__()
        self.stream = stream or sys.stderr

    def emit(self, record):
        """Emit a JSON-formatted log record"""
        try:
            # If the message is already JSON, use it directly
            if record.getMessage().startswith("{") and record.getMessage().endswith(
                "}"
            ):
                log_entry = record.getMessage()
            else:
                # Create structured log entry
                log_entry = json.dumps(
                    {
                        "timestamp": datetime.fromtimestamp(
                            record.created, timezone.utc
                        )
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno,
                        "process_id": record.process,
                        "thread_id": record.thread,
                    },
                    default=str,
                )

            self.stream.write(log_entry + "\n")
            self.stream.flush()

        except Exception:
            # Fallback to basic logging if JSON formatting fails
            self.stream.write(f"LOG_ERROR: {record.getMessage()}\n")
            self.stream.flush()


class LogContext:
    """Context manager for setting correlation IDs and log context"""

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.previous_id = getattr(_local, "correlation_id", None)

    def __enter__(self):
        _local.correlation_id = self.correlation_id
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _local.correlation_id = self.previous_id


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, logger: StructuredLogger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            self.logger.performance(self.operation, duration_ms, **self.context)


# Global logger instances
system_logger = StructuredLogger("localgpt.system")
indexing_logger = StructuredLogger("localgpt.indexing")
query_logger = StructuredLogger("localgpt.query")
api_logger = StructuredLogger("localgpt.api")
_structured_loggers = [system_logger, indexing_logger, query_logger, api_logger]


_TIMINGS_ENV = "LOCALGPT_TIMINGS"


def timings_enabled() -> bool:
    """Per-request RAG stage timing is opt-in via the LOCALGPT_TIMINGS env flag.

    Off by default so the chat hot path is byte-for-byte unchanged unless an
    operator explicitly turns instrumentation on.
    """
    return os.getenv(_TIMINGS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


class StageTimings:
    """Observes a chat request's event_callback stream to record per-stage
    wall-clock timings, time-to-first-token, and total latency.

    It is a passive observer: it never alters the events it sees, so attaching
    it is behavior-neutral. Stage spans are derived from the pipeline's existing
    ``*_started`` / ``*_done`` events; repeated rounds (agentic / decomposition)
    accumulate into the same stage bucket. TTFT is the first ``token`` event
    measured from request start, so it includes retrieve + rerank + prefill.
    """

    _STARTED = {
        "retrieval_started": "retrieval",
        "rerank_started": "rerank",
        "context_expand_started": "context_expand",
        "prune_started": "prune",
        "generation_started": "generation",
    }
    _DONE = {
        "retrieval_done": "retrieval",
        "rerank_done": "rerank",
        "context_expand_done": "context_expand",
        "prune_done": "prune",
        "generation_done": "generation",
    }

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._open: Dict[str, float] = {}
        self.stages_ms: Dict[str, float] = {}
        self.ttft_ms: Optional[float] = None

    def observe(self, event_type: str, _payload: Any = None) -> None:
        """Record one event. Safe to call for every event in the stream."""
        now = time.perf_counter()
        name = self._STARTED.get(event_type)
        if name is not None:
            self._open[name] = now
            return
        name = self._DONE.get(event_type)
        if name is not None:
            start = self._open.pop(name, None)
            if start is not None:
                self.stages_ms[name] = (
                    self.stages_ms.get(name, 0.0) + (now - start) * 1000.0
                )
            return
        if event_type == "token" and self.ttft_ms is None:
            self.ttft_ms = (now - self._t0) * 1000.0

    def as_dict(self) -> Dict[str, Any]:
        """Snapshot for attaching to a chat response. Always carries 'total'."""
        stages = {name: round(ms, 1) for name, ms in self.stages_ms.items()}
        stages["total"] = round((time.perf_counter() - self._t0) * 1000.0, 1)
        out: Dict[str, Any] = {"timings_ms": stages}
        if self.ttft_ms is not None:
            out["ttft_ms"] = round(self.ttft_ms, 1)
        return out

    def log(self, **context: Any) -> None:
        """Emit one structured 'rag_timings' line via the query logger."""
        snapshot = self.as_dict()
        fields = {f"{name}_ms": ms for name, ms in snapshot["timings_ms"].items()}
        query_logger.info(
            "rag_timings", ttft_ms=snapshot.get("ttft_ms"), **fields, **context
        )


def set_log_level(level: Union[str, int]):
    """Set log level for all structured loggers"""
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    for logger in _structured_loggers:
        logger.logger.setLevel(level)
        for handler in logger.logger.handlers:
            handler.setLevel(level)


def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure global logging settings"""
    set_log_level(log_level)
    level = getattr(logging, log_level.upper())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = JSONLogHandler()
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    structured_handlers = [console_handler]
    if log_file:
        file_handler = JSONLogHandler(open(log_file, "a", encoding="utf-8"))
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        structured_handlers.append(file_handler)

    for structured_logger in _structured_loggers:
        structured_logger.logger.handlers.clear()
        structured_logger.logger.setLevel(level)
        for handler in structured_handlers:
            structured_logger.logger.addHandler(handler)

    system_logger.info("logging_configured", log_level=log_level, log_file=log_file)


# Legacy compatibility functions
def log_query(query: str, sub_queries: Optional[list] = None) -> None:
    """Legacy function for backward compatibility"""
    context: Dict[str, Any] = {"query": query}
    if sub_queries:
        context["sub_queries"] = sub_queries
    query_logger.info("user_query", **context)


def log_retrieval_results(results: list, k: int) -> None:
    """Legacy function for backward compatibility"""
    if not results:
        query_logger.info("retrieval_empty")
        return

    query_logger.info(
        "retrieval_results",
        result_count=len(results),
        top_k=k,
        top_scores=[r.get("score", 0) for r in results[:k]],
    )
