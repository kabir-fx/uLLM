import { describe, expect, it } from "vitest";
import { sourceFromQuery } from "./config";
import { RUNS } from "./runs";

const stream = "http://127.0.0.1:8000/telemetry/stream";
const live = `?live=${encodeURIComponent(stream)}`;

describe("sourceFromQuery", () => {
  it("replays the first bundled run by default", () => {
    const mode = sourceFromQuery("", "/");
    expect(mode.kind).toBe("replay");
    if (mode.kind !== "replay") return;
    expect(mode.run).toBe(RUNS[0]);
    expect(mode.source.label).toBe(`/${RUNS[0]!.trace}`);
  });

  it("replays the run that ?run names, under the build's base", () => {
    const mode = sourceFromQuery("?run=steady-chat", "./");
    expect(mode.kind === "replay" && mode.run?.id).toBe("steady-chat");
    expect(mode.source.label).toBe("./telemetry/steady-chat.jsonl.gz");
  });

  it("rejects a run it does not bundle", () => {
    expect(() => sourceFromQuery("?run=nope", "/")).toThrow(/nope/);
  });

  it("replays the trace that ?trace names, as no bundled run", () => {
    const mode = sourceFromQuery("?trace=/runs/a.jsonl.gz", "/");
    expect(mode.kind === "replay" && mode.run).toBeNull();
    expect(mode.source.label).toBe("/runs/a.jsonl.gz");
  });

  it("starts a replay at ?speed", () => {
    const mode = sourceFromQuery("?speed=4", "/");
    expect(mode.kind === "replay" && mode.source.speed).toBe(4);
  });

  it("rejects a speed that is not a positive number", () => {
    expect(() => sourceFromQuery("?speed=0", "/")).toThrow(/speed/);
    expect(() => sourceFromQuery("?speed=fast", "/")).toThrow(/speed/);
  });

  it("reads the stream that ?live names, with chat to its origin and load to the default control process", () => {
    const mode = sourceFromQuery(live, "/");
    expect(mode.kind).toBe("live");
    if (mode.kind !== "live") return;
    expect(mode.source.label).toBe(stream);
    expect(mode.controls).toEqual({ api: "http://127.0.0.1:8000", control: "http://127.0.0.1:8090" });
  });

  it("takes the control process from ?control", () => {
    const mode = sourceFromQuery(`${live}&control=${encodeURIComponent("http://localhost:9000")}`, "/");
    expect(mode.kind === "live" && mode.controls.control).toBe("http://localhost:9000");
  });

  describe("in the hosted build", () => {
    it("plays only bundled runs, whatever ?live and ?trace say", () => {
      for (const query of [live, "?trace=https://example.com/x.jsonl.gz", `${live}&run=steady-chat`]) {
        const mode = sourceFromQuery(query, "./", true);
        expect(mode.kind).toBe("replay");
        expect(mode.kind === "replay" && mode.run).not.toBeNull();
        expect(mode.source.label.startsWith("./")).toBe(true);
      }
    });

    it("still picks a run and a speed", () => {
      const mode = sourceFromQuery("?run=prefix-pressure&speed=2", "./", true);
      expect(mode.kind === "replay" && mode.run?.id).toBe("prefix-pressure");
      expect(mode.kind === "replay" && mode.source.speed).toBe(2);
    });
  });
});
