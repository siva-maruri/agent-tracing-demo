"""Tracer provider with telemetry-scrubber in front of whatever exporter is used."""

from __future__ import annotations

import os

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from telemetry_scrubber import Scrubber, Tokenizer
from telemetry_scrubber.otel import ScrubbingSpanExporter


def default_scrubber() -> Scrubber:
    # Tokenize when a key is configured (so emails stay joinable), otherwise redact.
    tokenizer = Tokenizer.from_env() if os.environ.get("SCRUBBER_TOKEN_KEY") else None
    return Scrubber(tokenizer=tokenizer)


def tracer_provider(
    *exporters: SpanExporter,
    service_name: str = "support-agent",
    scrubber: Scrubber | None = None,
    batch: bool = True,
) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    scrubber = scrubber or default_scrubber()
    processor = BatchSpanProcessor if batch else SimpleSpanProcessor
    for exporter in exporters:
        provider.add_span_processor(processor(ScrubbingSpanExporter(exporter, scrubber)))
    return provider


def meter_provider(*readers: MetricReader, service_name: str = "support-agent") -> MeterProvider:
    # Metrics carry model names and token counts only, never message content, so they
    # don't go through the scrubber.
    return MeterProvider(
        metric_readers=list(readers), resource=Resource.create({"service.name": service_name})
    )
