import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { fixtureBytes as fixture } from "./testing/fixtures";
import { parseFrame, parseTrace } from "./trace";

describe("parseTrace", () => {
  it("reads every frame of a gzip trace, hello first", async () => {
    const frames = await parseTrace(fixture("steady-chat.jsonl.gz"));
    expect(frames).toHaveLength(625);
    expect(frames[0]?.events[0]?.type).toBe("hello");
    expect(frames.map((f) => f.seq)).toEqual(frames.map((_, i) => i));
  });

  it("reads a trace that arrives already decompressed", async () => {
    const plain = new Uint8Array(gunzipSync(fixture("steady-chat.jsonl.gz")));
    const frames = await parseTrace(plain);
    expect(frames).toHaveLength(625);
  });

  it("names the line of a frame from another protocol version", async () => {
    const text = '{"v":1,"replica":"r1","seq":0,"t":1,"dropped":0,"events":[]}\n{"v":2}\n';
    await expect(parseTrace(new TextEncoder().encode(text))).rejects.toThrow(/line 2.*version 2/);
  });
});

describe("parseFrame", () => {
  it("rejects text that is not a frame object", () => {
    expect(() => parseFrame("[1, 2]")).toThrow(/not a telemetry frame/);
  });

  it("rejects a version 1 frame without its fields", () => {
    expect(() => parseFrame('{"v":1,"seq":1,"t":5}')).toThrow(/not a telemetry frame/);
  });
});
