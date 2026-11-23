import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { RUNS } from "./runs";

const repo = (path: string) => new URL(`../../${path}`, import.meta.url);
const served = (trace: string) => new URL(`../fixtures/${trace}`, import.meta.url);

describe("RUNS", () => {
  it("has unique ids", () => {
    expect(new Set(RUNS.map((r) => r.id)).size).toBe(RUNS.length);
  });

  it("serves every run's trace from the build", async () => {
    for (const run of RUNS) await expect(readFile(served(run.trace))).resolves.toBeInstanceOf(Buffer);
  });

  it("bundles an exact copy of each real recording, labelled from its session's env.json", async () => {
    const real = RUNS.filter((r) => r.recording !== null);
    expect(real.length).toBeGreaterThan(0);
    for (const { trace, recording } of real) {
      expect(await readFile(served(trace))).toEqual(await readFile(repo(recording!.from)));
      const env = JSON.parse(await readFile(repo(recording!.from.replace(/[^/]+$/, "env.json")), "utf8"));
      expect(recording).toMatchObject({ chip: env.chip, memoryGb: env.memory_gb, commit: env.commit, dirty: env.dirty });
    }
  });
});
