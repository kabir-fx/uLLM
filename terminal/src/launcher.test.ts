import { describe, expect, it } from "vitest";
import type { LoadStatus } from "./contracts/control";
import { launchSpec, statusLine, type LaunchForm } from "./launcher";

const FORM: LaunchForm = { workload: "a", arrival: "closed", concurrency: 8, ratePerS: 2, maxTokens: 128, requests: 64 };

describe("launchSpec", () => {
  it("sends the concurrency for a closed loop and the rate for an open one", () => {
    expect(launchSpec(FORM)).toEqual({
      workload: "a",
      arrival: { kind: "closed", concurrency: 8 },
      max_tokens: 128,
      requests: 64,
    });
    expect(launchSpec({ ...FORM, arrival: "open" }).arrival).toEqual({ kind: "open", rate_per_s: 2 });
  });
});

describe("statusLine", () => {
  const spec = launchSpec(FORM);
  const status = (s: Partial<LoadStatus>): LoadStatus => ({ state: "idle", spec: null, ok: 0, errors: 0, first_error: null, ...s });

  it("says what runs, and the totals once it ends", () => {
    expect(statusLine(status({}))).toBe("idle");
    expect(statusLine(status({ state: "running", spec }))).toBe("running: 64 of a, closed loop of 8, max_tokens 128");
    expect(statusLine(status({ state: "done", spec, ok: 60, errors: 4, first_error: "HTTP 503: full" }))).toBe(
      "done: 60 ok, 4 errors, first: HTTP 503: full",
    );
    expect(statusLine(status({ state: "stopped", spec: launchSpec({ ...FORM, arrival: "open" }) }))).toBe(
      "stopped: 64 of a, open loop at 2/s, max_tokens 128",
    );
    expect(statusLine(status({ state: "failed", spec, first_error: "ConnectError: refused" }))).toBe(
      "failed: ConnectError: refused",
    );
  });
});
