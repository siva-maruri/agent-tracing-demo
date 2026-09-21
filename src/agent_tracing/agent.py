"""A small LangGraph support agent, traced end to end."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as G
from opentelemetry.trace import TracerProvider

from . import genai

SYSTEM_PROMPT = (
    "You are a support assistant. Check the order and any refund with the tools "
    "before you answer, and keep the answer to two or three sentences."
)


def build_agent(
    model,
    tools: Sequence,
    name: str = "support-agent",
    tracer_provider: TracerProvider | None = None,
) -> Callable[..., str]:
    tracer = (tracer_provider or trace.get_tracer_provider()).get_tracer("agent_tracing")
    by_name = {t.name: t for t in tools}
    usage = {"input": 0, "output": 0}

    def call_model(state: MessagesState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        with genai.chat_span(tracer, model.model_name, model.provider_name, messages) as span:
            reply = model.invoke(messages)
            i, o = genai.record_response(span, model.model_name, reply)
        usage["input"] += i
        usage["output"] += o
        return {"messages": [reply]}

    def call_tools(state: MessagesState) -> dict:
        results = []
        for call in state["messages"][-1].tool_calls:
            try:
                with genai.tool_span(tracer, call) as span:
                    tool = by_name.get(call["name"])
                    if tool is None:
                        raise LookupError(f"unknown tool {call['name']}")
                    output = tool.invoke(call["args"])
                    genai.record_tool_result(span, output)
                content = json.dumps(output, default=str)
            except Exception as exc:
                # The span already carries the error. Hand it back to the model so it can
                # recover instead of failing the whole run.
                content = json.dumps({"error": str(exc)})
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results}

    def route(state: MessagesState) -> str:
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, ["tools", END])
    graph.add_edge("tools", "model")
    app = graph.compile()

    def run(question: str, conversation_id: str | None = None) -> str:
        usage["input"] = usage["output"] = 0
        with genai.agent_span(tracer, name, conversation_id or uuid.uuid4().hex, model.provider_name) as span:
            state = app.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": 12})
            span.set_attribute(G.GEN_AI_USAGE_INPUT_TOKENS, usage["input"])
            span.set_attribute(G.GEN_AI_USAGE_OUTPUT_TOKENS, usage["output"])
        return state["messages"][-1].content

    return run
