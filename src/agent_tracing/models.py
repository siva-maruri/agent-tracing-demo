"""Chat models the agent can run on.

Anything with .invoke(messages) -> AIMessage plus model_name / provider_name works.
ScriptedModel replays canned replies so the demo and the tests run offline.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

from langchain_core.messages import AIMessage, BaseMessage


class ScriptedModel:
    provider_name = "openai"

    def __init__(self, replies: Iterable[AIMessage], model_name: str = "gpt-4o-mini"):
        self._replies = list(replies)
        self.model_name = model_name

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        if not self._replies:
            raise RuntimeError("scripted model ran out of replies")
        return self._replies.pop(0)


class AzureOpenAIModel:
    """Azure OpenAI through Entra ID, no API key. Needs the [azure] extra and
    AZURE_OPENAI_ENDPOINT; the caller needs Cognitive Services OpenAI User on the resource."""

    provider_name = "azure.ai.openai"

    def __init__(self, deployment: str, tools: Sequence, api_version: str = "2024-10-21"):
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from langchain_openai import AzureChatOpenAI

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        self.model_name = deployment
        self._llm = AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=api_version,
            azure_ad_token_provider=token_provider,
            temperature=0,
        ).bind_tools(list(tools))

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        return self._llm.invoke(list(messages))


def demo_script() -> ScriptedModel:
    """The conversation used by `python -m agent_tracing` and the tests."""

    def usage(i: int, o: int) -> dict:
        return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}

    return ScriptedModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "lookup_order", "args": {"order_id": "1042"}}],
                usage_metadata=usage(182, 21),
                response_metadata={"finish_reason": "tool_calls", "model_name": "gpt-4o-mini-2024-07-18"},
            ),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_2", "name": "refund_status", "args": {"order_id": "1042"}}],
                usage_metadata=usage(268, 19),
                response_metadata={"finish_reason": "tool_calls", "model_name": "gpt-4o-mini-2024-07-18"},
            ),
            AIMessage(
                content=(
                    "Order 1042 was charged twice for $59.90. A refund for the duplicate charge "
                    "(rf_77310) was issued on 2026-09-18 and is still processing, so it should "
                    "reach the card within 3-5 business days."
                ),
                usage_metadata=usage(331, 47),
                response_metadata={"finish_reason": "stop", "model_name": "gpt-4o-mini-2024-07-18"},
            ),
        ]
    )
