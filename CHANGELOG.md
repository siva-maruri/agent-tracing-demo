# Changelog

## 0.3.0

- Streaming (`--stream`, `build_agent(..., stream=True)`): same spans and token counts as a
  non-streamed run, plus `gen_ai.client.operation.time_to_first_chunk`. Azure OpenAI asks
  for usage on the final chunk.

## 0.2.0

- `gen_ai.client.token.usage` and `gen_ai.client.operation.duration` metrics with the
  semconv bucket boundaries. Failed calls record duration with `error.type`.
- The CLI prints a metrics summary and exports metrics over OTLP when an endpoint is set.

## 0.1.0

- LangGraph support agent with `invoke_agent` / `chat` / `execute_tool` spans following the
  GenAI semantic conventions, opt-in message content, scrubbing through telemetry-scrubber.
