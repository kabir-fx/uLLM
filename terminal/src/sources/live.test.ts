import { createServer, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import type { Frame } from "../contracts/telemetry";
import type { SourceStatus } from "../store";
import * as ev from "../testing/events";
import { liveSource, SseParser } from "./live";

describe("SseParser", () => {
  it("joins data lines split across chunks and skips comments", () => {
    const parser = new SseParser();
    expect(parser.push(': keepalive\n\ndata: {"a"')).toEqual([]);
    expect(parser.push(":1}\r\n\r\ndata: x\n")).toEqual(['{"a":1}']);
    expect(parser.push("data: y\n\n")).toEqual(["x\ny"]);
  });

  it("reads a CR LF pair split across two chunks as one line end", () => {
    const parser = new SseParser();
    expect(parser.push("data: a\r")).toEqual([]);
    expect(parser.push("\ndata: b\r\n\r\n")).toEqual(["a\nb"]);
  });

  it("dispatches nothing for an event without data", () => {
    expect(new SseParser().push("event: ping\n\n")).toEqual([]);
  });
});

let server: Server | undefined;
afterEach(() => {
  server?.closeAllConnections();
  return new Promise<void>((done) => (server ? server.close(() => done()) : done()));
});

function serve(handler: (res: ServerResponse) => void): Promise<string> {
  server = createServer((_, res) => handler(res));
  return new Promise((resolve) =>
    server!.listen(0, "127.0.0.1", () => resolve(`http://127.0.0.1:${(server!.address() as AddressInfo).port}/telemetry/stream`)),
  );
}

function run(url: string) {
  const frames: Frame[] = [];
  const statuses: SourceStatus[] = [];
  const ended = new Promise<void>((resolve) => {
    liveSource(url).start({
      frame: (f) => frames.push(f),
      status: (s) => {
        statuses.push(s);
        if (s.kind === "ended" || s.kind === "error") resolve();
      },
    });
  });
  return { frames, statuses, ended };
}

describe("liveSource", () => {
  it("sends every frame of an SSE stream, then ends", async () => {
    const [f0, f1] = ev.frames([[ev.hello()], [ev.arrived(5, "a")]]);
    const url = await serve((res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(": keepalive\n\n");
      const line = `data: ${JSON.stringify(f1)}\n\n`;
      res.write(`data: ${JSON.stringify(f0)}\n\n${line.slice(0, 20)}`);
      setTimeout(() => res.end(line.slice(20)), 20);
    });
    const out = run(url);
    await out.ended;
    expect(out.frames).toEqual([f0, f1]);
    expect(out.statuses.map((s) => s.kind)).toEqual(["connecting", "open", "ended"]);
  });

  it("reports an HTTP error", async () => {
    const url = await serve((res) => res.writeHead(503).end());
    const out = run(url);
    await out.ended;
    expect(out.statuses.at(-1)).toEqual({ kind: "error", message: `${url}: HTTP 503` });
  });

  it("reports a frame from another protocol version", async () => {
    const url = await serve((res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end('data: {"v":2}\n\n');
    });
    const out = run(url);
    await out.ended;
    expect(out.statuses.at(-1)?.kind).toBe("error");
    expect(out.frames).toEqual([]);
  });

  it("sends nothing and reports nothing after stop", async () => {
    let push: ((s: string) => void) | undefined;
    const url = await serve((res) => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.flushHeaders();
      push = (s) => res.write(s);
    });
    const frames: Frame[] = [];
    const statuses: SourceStatus["kind"][] = [];
    let opened!: () => void;
    const open = new Promise<void>((resolve) => (opened = resolve));
    const stop = liveSource(url).start({
      frame: (f) => frames.push(f),
      status: (s) => {
        statuses.push(s.kind);
        if (s.kind === "open") opened();
      },
    });
    await open;
    stop();
    push?.(`data: ${JSON.stringify(ev.frames([[ev.hello()]])[0])}\n\n`);
    await new Promise((r) => setTimeout(r, 50));
    expect(frames).toEqual([]);
    expect(statuses).toEqual(["connecting", "open"]);
  });
});
