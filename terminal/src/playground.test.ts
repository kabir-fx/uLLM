import { createServer, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { servedModel, streamChat, type ChatPiece } from "./playground";

let server: Server | undefined;
afterEach(() => {
  server?.closeAllConnections();
  return new Promise<void>((done) => (server ? server.close(() => done()) : done()));
});

const bodies: unknown[] = [];

function serve(reply: (url: string, res: ServerResponse) => void): Promise<string> {
  bodies.length = 0;
  server = createServer((req, res) => {
    let body = "";
    req.on("data", (c: Buffer) => (body += c.toString()));
    req.on("end", () => {
      if (body) bodies.push(JSON.parse(body));
      reply(req.url!, res);
    });
  });
  return new Promise((resolve) =>
    server!.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as AddressInfo).port}`)),
  );
}

const chunk = (delta: object, finish: string | null = null) =>
  `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;

async function collect(api: string, signal = new AbortController().signal): Promise<ChatPiece[]> {
  const pieces: ChatPiece[] = [];
  for await (const piece of streamChat(api, "m", "hi", 16, signal)) pieces.push(piece);
  return pieces;
}

describe("streamChat", () => {
  it("sends one streamed greedy request and yields each piece, then the usage", async () => {
    const api = await serve((_, res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(chunk({ role: "assistant" }));
      res.write(chunk({ content: "Hel" }));
      res.write(chunk({ content: "lo" }));
      res.write(chunk({}, "length"));
      res.write(`data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 9, completion_tokens: 2, total_tokens: 11 } })}\n\n`);
      res.end("data: [DONE]\n\n");
    });
    expect(await collect(api)).toEqual([
      { kind: "text", text: "Hel" },
      { kind: "text", text: "lo" },
      { kind: "done", finishReason: "length", promptTokens: 9, completionTokens: 2 },
    ]);
    expect(bodies).toEqual([
      {
        model: "m",
        messages: [{ role: "user", content: "hi" }],
        max_tokens: 16,
        temperature: 0,
        stream: true,
        stream_options: { include_usage: true },
      },
    ]);
  });

  it("throws the server's message on a refusal", async () => {
    const api = await serve((_, res) => {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "the prompt is too long", type: "invalid_request_error" } }));
    });
    await expect(collect(api)).rejects.toThrow("400: the prompt is too long");
  });

  it("throws an error event sent mid-stream", async () => {
    const api = await serve((_, res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(chunk({ content: "a" }));
      res.end(`data: ${JSON.stringify({ error: { message: "the engine failed this request" } })}\n\n`);
    });
    await expect(collect(api)).rejects.toThrow("the engine failed this request");
  });

  it("throws when the stream ends without [DONE]", async () => {
    const api = await serve((_, res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(chunk({ content: "a" }));
    });
    await expect(collect(api)).rejects.toThrow("the stream ended without [DONE]");
  });

  it("stops reading when aborted, and the server sees the client leave", async () => {
    let closed!: () => void;
    const left = new Promise<void>((resolve) => (closed = resolve));
    const api = await serve((_, res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(chunk({ content: "a" }));
      res.on("close", closed);
    });
    const abort = new AbortController();
    const pieces = streamChat(api, "m", "hi", 16, abort.signal);
    expect((await pieces.next()).value).toEqual({ kind: "text", text: "a" });
    abort.abort();
    await expect(pieces.next()).rejects.toThrow();
    await left;
  });
});

describe("servedModel", () => {
  it("returns the first model the server lists", async () => {
    const api = await serve((url, res) => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(url === "/v1/models" ? { object: "list", data: [{ id: "fake" }, { id: "b" }] } : {}));
    });
    expect(await servedModel(api)).toBe("fake");
  });
});
