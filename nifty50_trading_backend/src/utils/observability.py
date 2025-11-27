from __future__ import annotations

from fastapi import APIRouter, Response
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Attempt to import Prometheus client; fall back to stubbed endpoints if unavailable
PROM_AVAILABLE = True
try:
    from prometheus_client import (
        Counter,
        Histogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
except Exception:  # pragma: no cover - only hit when dependency is missing
    PROM_AVAILABLE = False
    Counter = None  # type: ignore[assignment]
    Histogram = None  # type: ignore[assignment]
    CollectorRegistry = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"  # default text/plain


if PROM_AVAILABLE:
    # Create a custom registry to avoid global state pollution in tests
    _registry = CollectorRegistry()
    REQUEST_COUNTER = Counter(
        "service_requests_total",
        "Total number of service requests",
        ["endpoint"],
        registry=_registry,
    )
    REQUEST_LATENCY = Histogram(
        "service_request_latency_seconds",
        "Latency of service requests in seconds",
        ["endpoint"],
        registry=_registry,
    )
else:
    _registry = None
    REQUEST_COUNTER = None  # type: ignore[assignment]
    REQUEST_LATENCY = None  # type: ignore[assignment]


# PUBLIC_INTERFACE
def get_metrics_router() -> APIRouter:
    """Return an APIRouter exposing service metrics in JSON and Prometheus formats.

    If the prometheus_client package is not installed, the JSON endpoint still works with
    minimal information and the Prometheus endpoint returns a 501 status with a message.
    """
    router = APIRouter()

    @router.get("/", summary="Service metrics (JSON)")
    def json_metrics() -> dict:
        """Return a simple JSON snapshot of key metrics."""
        if PROM_AVAILABLE:
            labels = list(REQUEST_COUNTER._labelnames)  # type: ignore[attr-defined]
        else:
            labels = []
        return {
            "counters": {
                "service_requests_total": {
                    "description": "Total number of service requests",
                    "labels": labels,
                }
            },
            "prometheus_enabled": PROM_AVAILABLE,
            "notes": "Use /metrics/prom for Prometheus exposition format when enabled.",
        }

    @router.get("/prom", summary="Prometheus metrics")
    def prom_metrics() -> Response:
        """Expose all metrics in Prometheus exposition format or 501 if unavailable."""
        if not PROM_AVAILABLE or _registry is None or generate_latest is None:
            return Response(
                content=b"prometheus_client not installed; metrics not available.",
                media_type=CONTENT_TYPE_LATEST,
                status_code=501,
            )
        output = generate_latest(_registry)
        return Response(content=output, media_type=CONTENT_TYPE_LATEST)

    return router


# Instantiate a router for the app to mount
metrics_router = get_metrics_router()
