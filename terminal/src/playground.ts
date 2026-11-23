import { SseParser } from "./sources/live";

export type ChatPiece =
  | { kind: "text"; text: string }
  | { kind: "done"; finishReason: string | null; promptTokens: number; completionTokens: number };

/** The first model `GET /v1/models` lists. */
export async function servedModel(api: string): Promise<string> {
  const response = await fetch(`${api}/v1/models`);
  if (!response.ok) throw new Error(`${api}/v1/models: HTTP ${response.status}`);
  const { data } = (await response.json()) as { data: { id: string }[] };
  if (!data[0]) throw new Error(`${api} serves no model`);
  return data[0].id;
}

/**
 * One greedy, streamed chat completion (C1): each content piece as it comes, then the usage.
 * Throws on a refusal, an error event, or a stream that ends without `[DONE]`.
 */
export async function* streamChat(
  api: string,
  model: string,
  prompt: string,
  maxTokens: number,
  signal: AbortSignal,
): AsyncGenerator<ChatPiece> {
  const response = await fetch(`${api}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature: 0,
      stream: true,
      stream_options: { include_usage: true },
    }),
    signal,
  });
  if (!response.ok || response.body === null) {
    const payload = (await response.json().catch(() => null)) as { error?: { message?: string } } | null;
    throw new Error(`${response.status}: ${payload?.error?.message ?? response.statusText}`);
  }
  const parser = new SseParser();
  let finishReason: string | null = null;
  for await (const chunk of response.body.pipeThrough(new TextDecoderStream())) {
    for (const data of parser.push(chunk)) {
      if (data === "[DONE]") return;
      const event = JSON.parse(data) as Chunk;
      if (event.error) throw new Error(event.error.message ?? JSON.stringify(event.error));
      for (const choice of event.choices ?? []) {
        if (choice.delta?.content) yield { kind: "text", text: choice.delta.content };
        if (choice.finish_reason) finishReason = choice.finish_reason;
      }
      if (event.usage) {
        const { prompt_tokens: promptTokens, completion_tokens: completionTokens } = event.usage;
        yield { kind: "done", finishReason, promptTokens, completionTokens };
      }
    }
  }
  throw new Error("the stream ended without [DONE]");
}

interface Chunk {
  error?: { message?: string };
  choices?: { delta?: { content?: string }; finish_reason?: string | null }[];
  usage?: { prompt_tokens: number; completion_tokens: number };
}
