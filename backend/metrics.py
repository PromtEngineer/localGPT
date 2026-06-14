"""In-memory request metrics with Prometheus text-format export."""

import threading
from collections import defaultdict
from typing import Dict, List


class _Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._active_requests: int = 0
        self._index_job_queue_depth: int = 0

    # ---- mutation helpers ----

    def record_request(self, endpoint: str, latency_ms: float) -> None:
        with self._lock:
            self._request_count[endpoint] += 1
            bucket = self._latencies[endpoint]
            bucket.append(latency_ms)
            if len(bucket) > 1000:
                # Keep only the most recent 1000 samples to bound memory
                self._latencies[endpoint] = bucket[-1000:]

    def inc_active(self) -> None:
        with self._lock:
            self._active_requests += 1

    def dec_active(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._index_job_queue_depth = depth

    # ---- read helpers ----

    def _percentile(self, values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * pct / 100)
        return round(sorted_vals[min(idx, len(sorted_vals) - 1)], 2)

    def snapshot(self) -> dict:
        with self._lock:
            snap: dict = {
                "active_requests": self._active_requests,
                "index_job_queue_depth": self._index_job_queue_depth,
                "endpoints": {},
            }
            for ep, count in self._request_count.items():
                lats = self._latencies.get(ep, [])
                snap["endpoints"][ep] = {
                    "request_count": count,
                    "latency_p50_ms": self._percentile(lats, 50),
                    "latency_p95_ms": self._percentile(lats, 95),
                    "latency_p99_ms": self._percentile(lats, 99),
                }
            return snap

    def prometheus_text(self) -> str:
        """Render metrics in Prometheus exposition format."""
        lines: List[str] = []
        snap = self.snapshot()

        lines.append("# HELP http_active_requests Currently in-flight HTTP requests")
        lines.append("# TYPE http_active_requests gauge")
        lines.append(f"http_active_requests {snap['active_requests']}")

        lines.append("# HELP index_job_queue_depth Number of queued index jobs")
        lines.append("# TYPE index_job_queue_depth gauge")
        lines.append(f"index_job_queue_depth {snap['index_job_queue_depth']}")

        lines.append("# HELP http_requests_total Total request count per endpoint")
        lines.append("# TYPE http_requests_total counter")
        for ep, stats in snap["endpoints"].items():
            safe = ep.replace('"', '\\"')
            lines.append(
                f'http_requests_total{{endpoint="{safe}"}} {stats["request_count"]}'
            )

        for pct_label, pct_key in [
            ("0.5", "latency_p50_ms"),
            ("0.95", "latency_p95_ms"),
            ("0.99", "latency_p99_ms"),
        ]:
            lines.append(
                "# HELP http_request_latency_ms Request latency quantiles (ms)"
            )
            lines.append("# TYPE http_request_latency_ms summary")
            for ep, stats in snap["endpoints"].items():
                safe = ep.replace('"', '\\"')
                lines.append(
                    f'http_request_latency_ms{{endpoint="{safe}",quantile="{pct_label}"}} {stats[pct_key]}'
                )

        return "\n".join(lines) + "\n"


# Module-level singleton
metrics = _Metrics()
