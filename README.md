# agent-tracing-demo

A small LangGraph support agent traced with the OpenTelemetry
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), with prompts,
completions and tool results scrubbed by
[telemetry-scrubber](https://github.com/siva-maruri/telemetry-scrubber) before they leave
the process. The traces can go straight into
[otel-azure-pipeline](https://github.com/siva-maruri/otel-azure-pipeline).

The agent answers "was this order double charged, and is a refund coming?" by calling two tools.
The point isn't the agent. It's what an agent run looks like as a trace, and keeping customer
data out of it.

## Run it

```bash
pip install -e ".[otlp]"
python -m agent_tracing
```

It runs offline against a scripted model and prints the answer, the trace and the metrics it produced:

```
Order 1042 was charged twice for $59.90. A refund for the duplicate charge (rf_77310) was issued on 2026-09-18 and is still processing, so it should reach the card within 3-5 business days.

invoke_agent support-agent             8.6 ms  tokens in/out 781/87
  chat gpt-4o-mini                       0.0 ms  tokens in/out 182/21  finish tool_calls
  execute_tool lookup_order              0.5 ms
  chat gpt-4o-mini                       0.0 ms  tokens in/out 268/19  finish tool_calls
  execute_tool refund_status             0.4 ms
  chat gpt-4o-mini                       0.0 ms  tokens in/out 331/47  finish stop

gen_ai.client.operation.duration             count 3  sum 0.000522814 s
gen_ai.client.token.usage input              count 3  sum 781 {token}
gen_ai.client.token.usage output             count 3  sum 87 {token}
```

Add `--stream` to stream the model responses; the spans and token counts come out the same,
plus a `gen_ai.client.operation.time_to_first_chunk` histogram. Set
`OTEL_EXPORTER_OTLP_ENDPOINT` to also send everything to a Collector. For a real model:

```bash
pip install -e ".[azure,otlp]"
export AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
python -m agent_tracing --azure-deployment gpt-4o-mini
```

That uses Entra ID through `DefaultAzureCredential`, no API key. The identity needs
Cognitive Services OpenAI User on the resource.

## What gets recorded

| Span | Kind | Key attributes |
|---|---|---|
| `invoke_agent {agent}` | INTERNAL | `gen_ai.agent.name`, `gen_ai.conversation.id`, total token usage |
| `chat {model}` | CLIENT | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.*`, `gen_ai.response.finish_reasons` |
| `execute_tool {tool}` | INTERNAL | `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.type` |

Attribute names come from `opentelemetry-semantic-conventions` rather than being typed in, so
a rename upstream shows up as an import error instead of silently wrong telemetry.

It also records the two GenAI client metrics, `gen_ai.client.token.usage` (split by
`gen_ai.token.type`) and `gen_ai.client.operation.duration`, with the bucket boundaries the
conventions recommend. Spans answer "what happened in this run"; the metrics answer "what
is this model costing per day" without anyone aggregating spans. Failed calls still record
a duration, tagged with `error.type`, and no tokens. When streaming, time to first chunk is
recorded too, which is the number users actually feel.

Failures set the span status, `error.type` and an exception event. A failing tool doesn't
fail the run: the error goes back to the model as the tool result, and the span keeps the
record.

## Message content

Prompts, completions, tool arguments and tool results are **not** recorded unless
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`, the same switch the official
GenAI instrumentations use. When it's on, they go into `gen_ai.input.messages`,
`gen_ai.output.messages`, `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result`, and
everything passes through telemetry-scrubber on export. With `SCRUBBER_TOKEN_KEY` set, the
customer email in the order lookup comes out tokenized:

```json
{"order_id": "1042", "customer_email": "tok_v1_b341209daace81b8", "amount": 59.9, "currency": "USD", "charge_count": 2, "status": "shipped"}
```

The same email gets the same token in the chat span that later reads that tool result, so the
two can still be lined up in a query without anyone seeing the address.

## Design notes

**Manual spans instead of a LangChain callback handler.** Each span is opened where the work
happens, so it's obvious what's covered and the span tree matches the code. It also keeps
the tracing independent of LangChain's callback internals, which change often.

**Context crosses LangGraph nodes.** The agent span is opened around `graph.invoke`, and every
chat and tool span ends up as its child in the same trace. There's a test for that, because
it's the first thing to break if graph execution moves to another thread without context.

**Streaming keeps the same trace shape.** The chat span covers the whole stream, and the
chunks are merged back into one message before anything is recorded, so a streamed run and a
non-streamed run produce the same spans and token counts. Usage arrives on the last chunk
(`stream_usage=True` for Azure OpenAI), which is why it's read after the merge rather than
per chunk.

**Prompts never go in span names.** Span names are `chat {model}` and `execute_tool {tool}`.
Names are indexed and shown everywhere, so they need to stay low-cardinality and free of user
input.

## Tests

`pytest` covers the span tree and parent links, the semconv attributes on each span type,
content being off by default, scrubbing and tokenization when it's on, tool and model
failures, token and duration metrics (including the error path), streaming (same trace, plus
time to first chunk), and the Azure model wiring (constructed without a network call).

## Compatibility

The GenAI semantic conventions are still experimental upstream and have renamed attributes
before (`gen_ai.system` became `gen_ai.provider.name`). Every attribute and metric name here
comes from the `opentelemetry-semantic-conventions` package, so a rename shows up as an import
error in CI rather than as silently mislabelled telemetry. Built against 0.65b0.
