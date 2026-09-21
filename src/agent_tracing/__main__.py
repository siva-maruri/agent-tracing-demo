"""Run the demo conversation and print the trace it produced.

python -m agent_tracing                       # scripted model, prints the span tree
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 python -m agent_tracing   # also ship it
AZURE_OPENAI_ENDPOINT=... python -m agent_tracing --azure-deployment gpt-4o-mini
"""

from __future__ import annotations

import argparse
import os

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from .agent import build_agent
from .models import AzureOpenAIModel, demo_script
from .telemetry import tracer_provider
from .tools import TOOLS

QUESTION = "Customer says order 1042 was charged twice. What happened, and is a refund on the way?"


def print_tree(spans) -> None:
    children: dict = {}
    for s in spans:
        children.setdefault(s.parent.span_id if s.parent else None, []).append(s)

    def walk(parent_id, depth):
        for s in sorted(children.get(parent_id, []), key=lambda s: s.start_time):
            ms = (s.end_time - s.start_time) / 1e6
            attrs = s.attributes
            extra = []
            if "gen_ai.usage.input_tokens" in attrs:
                tokens_in, tokens_out = (
                    attrs["gen_ai.usage.input_tokens"],
                    attrs["gen_ai.usage.output_tokens"],
                )
                extra.append(f"tokens in/out {tokens_in}/{tokens_out}")
            if "gen_ai.response.finish_reasons" in attrs:
                extra.append(f"finish {attrs['gen_ai.response.finish_reasons'][0]}")
            if not s.status.is_ok:
                extra.append(f"ERROR {attrs.get('error.type')}")
            print(f"{'  ' * depth}{s.name:<34} {ms:7.1f} ms  {'  '.join(extra)}")
            walk(s.context.span_id, depth + 1)

    walk(None, 0)


def main() -> None:
    p = argparse.ArgumentParser(prog="agent_tracing")
    p.add_argument("question", nargs="?", default=QUESTION)
    p.add_argument("--azure-deployment", help="use Azure OpenAI instead of the scripted model")
    args = p.parse_args()

    memory = InMemorySpanExporter()
    exporters = [memory]
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporters.append(OTLPSpanExporter())
    provider = tracer_provider(*exporters, batch=False)

    model = AzureOpenAIModel(args.azure_deployment, TOOLS) if args.azure_deployment else demo_script()
    answer = build_agent(model, TOOLS, tracer_provider=provider)(args.question)
    provider.shutdown()

    print(answer, end="\n\n")
    print_tree(memory.get_finished_spans())


if __name__ == "__main__":
    main()
