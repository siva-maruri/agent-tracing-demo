import json

import pytest
from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from telemetry_scrubber import Scrubber, Tokenizer

from agent_tracing import TOOLS, ScriptedModel, build_agent, demo_script, tracer_provider
from agent_tracing.genai import CAPTURE_ENV

QUESTION = "Customer says order 1042 was charged twice. What happened?"


def run(model, capture=False, monkeypatch=None, tokenize=False):
    if monkeypatch is not None:
        monkeypatch.setenv(CAPTURE_ENV, "true" if capture else "false")
    memory = InMemorySpanExporter()
    scrubber = Scrubber(tokenizer=Tokenizer(b"k" * 32) if tokenize else None)
    provider = tracer_provider(memory, scrubber=scrubber, batch=False)
    answer = build_agent(model, TOOLS, tracer_provider=provider)(QUESTION, conversation_id="conv-1")
    return answer, {s.name: s for s in memory.get_finished_spans()}, memory.get_finished_spans()


def test_span_tree_follows_semconv(monkeypatch):
    answer, _, spans = run(demo_script(), monkeypatch=monkeypatch)
    assert "rf_77310" in answer

    (agent,) = [s for s in spans if s.name == "invoke_agent support-agent"]
    chats = [s for s in spans if s.name == "chat gpt-4o-mini"]
    tools = sorted(s.name for s in spans if s.name.startswith("execute_tool"))
    assert len(chats) == 3
    assert tools == ["execute_tool lookup_order", "execute_tool refund_status"]

    # Everything hangs off the agent span in one trace, even across LangGraph nodes.
    for s in spans:
        assert s.context.trace_id == agent.context.trace_id
        if s is not agent:
            assert s.parent.span_id == agent.context.span_id

    assert agent.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent.attributes["gen_ai.conversation.id"] == "conv-1"
    assert agent.attributes["gen_ai.usage.input_tokens"] == 182 + 268 + 331
    assert agent.attributes["gen_ai.usage.output_tokens"] == 21 + 19 + 47


def test_chat_span_attributes(monkeypatch):
    _, _, spans = run(demo_script(), monkeypatch=monkeypatch)
    first, *_, last = sorted((s for s in spans if s.name.startswith("chat")), key=lambda s: s.start_time)
    assert first.kind == SpanKind.CLIENT
    assert first.attributes["gen_ai.operation.name"] == "chat"
    assert first.attributes["gen_ai.provider.name"] == "openai"
    assert first.attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert first.attributes["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert first.attributes["gen_ai.usage.input_tokens"] == 182
    assert first.attributes["gen_ai.response.finish_reasons"] == ("tool_calls",)
    assert last.attributes["gen_ai.response.finish_reasons"] == ("stop",)


def test_tool_span_attributes(monkeypatch):
    _, by_name, _ = run(demo_script(), monkeypatch=monkeypatch)
    span = by_name["execute_tool lookup_order"]
    assert span.attributes["gen_ai.tool.name"] == "lookup_order"
    assert span.attributes["gen_ai.tool.call.id"] == "call_1"
    assert span.attributes["gen_ai.tool.type"] == "function"


def test_content_not_captured_by_default(monkeypatch):
    _, _, spans = run(demo_script(), monkeypatch=monkeypatch)
    for s in spans:
        assert not any(
            k in s.attributes
            for k in (
                "gen_ai.input.messages",
                "gen_ai.output.messages",
                "gen_ai.tool.call.arguments",
                "gen_ai.tool.call.result",
            )
        )


def test_captured_content_is_scrubbed(monkeypatch):
    _, by_name, spans = run(demo_script(), capture=True, monkeypatch=monkeypatch, tokenize=True)

    exported = json.dumps([dict(s.attributes) for s in spans])
    assert "jane.doe@contoso.com" not in exported

    result = json.loads(by_name["execute_tool lookup_order"].attributes["gen_ai.tool.call.result"])
    assert result["customer_email"].startswith("tok_v1_")
    assert result["amount"] == 59.9

    # The last chat call saw both tool results; the email is tokenized there too, and the
    # token matches the one on the tool span, so the two can still be correlated.
    last_chat = max((s for s in spans if s.name.startswith("chat")), key=lambda s: s.start_time)
    inputs = last_chat.attributes["gen_ai.input.messages"]
    assert result["customer_email"] in inputs
    roles = [m["role"] for m in json.loads(inputs)]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]


def test_failing_tool_is_recorded_and_agent_recovers(monkeypatch):
    model = ScriptedModel(
        [
            AIMessage(
                content="", tool_calls=[{"id": "c1", "name": "lookup_order", "args": {"order_id": "9999"}}]
            ),
            AIMessage(
                content="", tool_calls=[{"id": "c2", "name": "cancel_order", "args": {"order_id": "9999"}}]
            ),
            AIMessage(content="I couldn't find order 9999."),
        ]
    )
    answer, by_name, _ = run(model, monkeypatch=monkeypatch)
    assert answer == "I couldn't find order 9999."

    missing = by_name["execute_tool lookup_order"]
    assert missing.status.status_code == StatusCode.ERROR
    assert missing.attributes["error.type"] == "NotFound"
    assert missing.events[0].name == "exception"

    unknown = by_name["execute_tool cancel_order"]
    assert unknown.attributes["error.type"] == "LookupError"
    assert by_name["invoke_agent support-agent"].status.status_code == StatusCode.UNSET


def test_model_failure_marks_chat_and_agent_spans(monkeypatch):
    monkeypatch.setenv(CAPTURE_ENV, "false")
    memory = InMemorySpanExporter()
    provider = tracer_provider(memory, scrubber=Scrubber(), batch=False)
    agent = build_agent(ScriptedModel([]), TOOLS, tracer_provider=provider)  # no replies at all

    with pytest.raises(RuntimeError):
        agent(QUESTION)

    spans = {s.name: s for s in memory.get_finished_spans()}
    for name in ("chat gpt-4o-mini", "invoke_agent support-agent"):
        assert spans[name].status.status_code == StatusCode.ERROR
        assert spans[name].attributes["error.type"] == "RuntimeError"


def test_azure_model_builds_without_network(monkeypatch):
    pytest.importorskip("langchain_openai")
    pytest.importorskip("azure.identity")
    from agent_tracing import AzureOpenAIModel

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    model = AzureOpenAIModel("gpt-4o-mini", TOOLS)
    assert model.provider_name == "azure.ai.openai"
    assert model.model_name == "gpt-4o-mini"
