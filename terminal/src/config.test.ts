import { describe, expect, it } from "vitest";
import { sourceFromQuery } from "./config";

describe("sourceFromQuery", () => {
  it("replays the steady-chat fixture by default", () => {
    expect(sourceFromQuery("", "/").label).toBe("/telemetry/steady-chat.jsonl.gz");
  });

  it("replays the trace that ?trace names", () => {
    expect(sourceFromQuery("?trace=/runs/a.jsonl.gz", "/").label).toBe("/runs/a.jsonl.gz");
  });

  it("reads the stream that ?live names", () => {
    const url = "http://127.0.0.1:8000/telemetry/stream";
    expect(sourceFromQuery(`?live=${encodeURIComponent(url)}`, "/").label).toBe(url);
  });

  it("rejects a speed that is not a positive number", () => {
    expect(() => sourceFromQuery("?speed=0", "/")).toThrow(/speed/);
    expect(() => sourceFromQuery("?speed=fast", "/")).toThrow(/speed/);
  });
});
