"""Span helpers that follow the OpenTelemetry GenAI semantic conventions.

Spans:  invoke_agent {agent}  ->  chat {model}  /  execute_tool {tool}
Message content (prompts, completions, tool arguments and results) is opt-in through
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, the same switch the official
instrumentations use. It's off by default because it's where PII ends up.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as G
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer

CAPTURE_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def capture_content() -> bool:
    return os.environ.get(CAPTURE_ENV, "").lower() == "true"


def to_semconv_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """LangChain messages -> the gen_ai.input.messages / output.messages JSON shape."""
    out = []
    for m in messages:
        parts: list[dict[str, Any]] = []
        if isinstance(m, ToolMessage):
            parts.append({"type": "tool_call_response", "id": m.tool_call_id, "response": m.content})
        elif m.content:
            parts.append({"type": "text", "content": m.content})
        for call in getattr(m, "tool_calls", None) or []:
            parts.append(
                {"type": "tool_call", "id": call["id"], "name": call["name"], "arguments": call["args"]}
            )
        out.append({"role": _ROLES.get(m.type, m.type), "parts": parts})
    return out


def _fail(span: Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.set_attribute(ERROR_TYPE, type(exc).__qualname__)


@contextmanager
def agent_span(tracer: Tracer, agent: str, conversation_id: str, provider: str) -> Iterator[Span]:
    with tracer.start_as_current_span(
        f"invoke_agent {agent}",
        kind=SpanKind.INTERNAL,
        record_exception=False,
        set_status_on_exception=False,
        attributes={
            G.GEN_AI_OPERATION_NAME: G.GenAiOperationNameValues.INVOKE_AGENT.value,
            G.GEN_AI_AGENT_NAME: agent,
            G.GEN_AI_CONVERSATION_ID: conversation_id,
            G.GEN_AI_PROVIDER_NAME: provider,
        },
    ) as span:
        try:
            yield span
        except Exception as exc:
            _fail(span, exc)
            raise


@contextmanager
def chat_span(tracer: Tracer, model: str, provider: str, messages: Sequence[BaseMessage]) -> Iterator[Span]:
    # The model name goes in the span name, the prompt never does: span names have to stay
    # low-cardinality and they're indexed and shown everywhere.
    with tracer.start_as_current_span(
        f"chat {model}",
        kind=SpanKind.CLIENT,
        record_exception=False,
        set_status_on_exception=False,
        attributes={
            G.GEN_AI_OPERATION_NAME: G.GenAiOperationNameValues.CHAT.value,
            G.GEN_AI_PROVIDER_NAME: provider,
            G.GEN_AI_REQUEST_MODEL: model,
        },
    ) as span:
        if capture_content():
            span.set_attribute(G.GEN_AI_INPUT_MESSAGES, json.dumps(to_semconv_messages(messages)))
        try:
            yield span
        except Exception as exc:
            _fail(span, exc)
            raise


def record_response(span: Span, requested_model: str, reply: AIMessage) -> tuple[int, int]:
    meta = reply.response_metadata or {}
    finish = meta.get("finish_reason") or ("tool_calls" if reply.tool_calls else "stop")
    usage = reply.usage_metadata or {}
    input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    span.set_attribute(G.GEN_AI_RESPONSE_MODEL, meta.get("model_name", requested_model))
    span.set_attribute(G.GEN_AI_RESPONSE_FINISH_REASONS, [finish])
    span.set_attribute(G.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    span.set_attribute(G.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    if capture_content():
        out = to_semconv_messages([reply])
        out[0]["finish_reason"] = finish
        span.set_attribute(G.GEN_AI_OUTPUT_MESSAGES, json.dumps(out))
    return input_tokens, output_tokens


@contextmanager
def tool_span(tracer: Tracer, call: dict[str, Any]) -> Iterator[Span]:
    with tracer.start_as_current_span(
        f"execute_tool {call['name']}",
        kind=SpanKind.INTERNAL,
        record_exception=False,
        set_status_on_exception=False,
        attributes={
            G.GEN_AI_OPERATION_NAME: G.GenAiOperationNameValues.EXECUTE_TOOL.value,
            G.GEN_AI_TOOL_NAME: call["name"],
            G.GEN_AI_TOOL_CALL_ID: call["id"],
            G.GEN_AI_TOOL_TYPE: "function",
        },
    ) as span:
        if capture_content():
            span.set_attribute(G.GEN_AI_TOOL_CALL_ARGUMENTS, json.dumps(call["args"]))
        try:
            yield span
        except Exception as exc:
            _fail(span, exc)
            raise


def record_tool_result(span: Span, result: Any) -> None:
    if capture_content():
        span.set_attribute(G.GEN_AI_TOOL_CALL_RESULT, json.dumps(result, default=str))
