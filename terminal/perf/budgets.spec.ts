import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Browser, type Page } from "@playwright/test";
import { fixtureFrames } from "../src/testing/fixtures";
import { repeatTrace } from "../src/testing/repeat";

// The T7 budgets (decision 0027). `make terminal-perf` sets PERF_OUT and commits what lands there.
const HOSTED = "http://127.0.0.1:4183/";
const FPS_MIN = 58;
const FRAME_P99_MAX_MS = 25;
// The TTFT and TPOT rings hold 4,096 samples and the stress trace adds 218 a repeat, so every
// ring is full after 19 repeats; growth past that is a leak.
const BASE_REPEATS = 20;
const LONG_REPEATS = 40;
const LOADS = 3;
const GROWTH_MAX = 1.05;
/** Nothing listens here: Playwright answers both from the test. */
const STREAM = "http://127.0.0.1:9999/telemetry/stream";
const CONTROL = "http://127.0.0.1:9998";
const INTERACTIVE_MAX_MS = 1000;
const COLD_LOADS = 5;

const result: Record<string, unknown> = {};

test.describe.configure({ mode: "serial" });

test.afterAll(async ({ browser }) => {
  const out = process.env.PERF_OUT;
  if (!out) return;
  const report = { chromium: browser.version(), viewport: test.info().project.use.viewport, ...result };
  writeFileSync(join(out, "result.json"), JSON.stringify(report, null, 2) + "\n");
});

test("60 fps over the stress trace, with the finished tab open", async ({ page }) => {
  await page.goto("/?trace=telemetry/stress.jsonl.gz");
  await page.locator('[role="tab"][data-tab="finished"]').click();
  await page.evaluate(() => {
    const stamps: number[] = [];
    (window as unknown as { stamps: number[] }).stamps = stamps;
    requestAnimationFrame(function tick(t) {
      stamps.push(t);
      requestAnimationFrame(tick);
    });
  });
  await expect(page.locator(".status")).toHaveText("ended", { timeout: 120_000 });
  const stamps = await page.evaluate(() => (window as unknown as { stamps: number[] }).stamps);
  const finishedRows = Number(await page.locator('[data-tab="finished"] .dim').textContent());

  const gaps = stamps.slice(1).map((t, i) => t - stamps[i]!).sort((a, b) => a - b);
  const fps = gaps.length / ((stamps.at(-1)! - stamps[0]!) / 1000);
  const p99 = gaps[Math.floor(gaps.length * 0.99)]!;
  result.fps = {
    frames: stamps.length,
    mean_fps: round(fps),
    p50_ms: round(gaps[gaps.length >> 1]!),
    p99_ms: round(p99),
    max_ms: round(gaps.at(-1)!),
    finished_rows: finishedRows,
    budget: { mean_fps_min: FPS_MIN, p99_ms_max: FRAME_P99_MAX_MS },
    pass: fps >= FPS_MIN && p99 <= FRAME_P99_MAX_MS,
  };
  expect(finishedRows).toBe(200);
  expect(fps).toBeGreaterThanOrEqual(FPS_MIN);
  expect(p99).toBeLessThanOrEqual(FRAME_P99_MAX_MS);
});

test("flat memory: a live session holds the same heap after 40 repeats of the stress trace as after 20", async ({ browser }) => {
  const stress = await fixtureFrames("stress.jsonl.gz");
  const samples = async (repeats: number) => {
    const sse = repeatTrace(stress, repeats).map((f) => `data: ${JSON.stringify(f)}\n\n`).join("");
    const out: { heap: number; nodes: number }[] = [];
    for (let i = 0; i < LOADS; i++) out.push(await liveMemoryAfter(browser, sse));
    return out;
  };
  const base = await samples(BASE_REPEATS);
  const long = await samples(LONG_REPEATS);

  const heapGrowth = median(long.map((s) => s.heap)) / median(base.map((s) => s.heap));
  const nodeGrowth = median(long.map((s) => s.nodes)) / median(base.map((s) => s.nodes));
  result.memory = {
    repeats: { base: BASE_REPEATS, long: LONG_REPEATS },
    heap_bytes: { base: base.map((s) => s.heap), long: long.map((s) => s.heap) },
    dom_nodes: { base: base.map((s) => s.nodes), long: long.map((s) => s.nodes) },
    heap_long_over_base: round(heapGrowth),
    nodes_long_over_base: round(nodeGrowth),
    budget: { growth_max: GROWTH_MAX },
    pass: heapGrowth <= GROWTH_MAX && nodeGrowth <= GROWTH_MAX,
  };
  expect(heapGrowth).toBeLessThanOrEqual(GROWTH_MAX);
  expect(nodeGrowth).toBeLessThanOrEqual(GROWTH_MAX);
});

/**
 * JS heap and DOM nodes after a forced GC, once a live stream has delivered `sse` in full.
 * Live mode keeps no copy of the frames, unlike a replay, so a leak of frames shows.
 */
async function liveMemoryAfter(browser: Browser, sse: string): Promise<{ heap: number; nodes: number }> {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.route(STREAM, (route) =>
    route.fulfill({ body: sse, headers: { "content-type": "text/event-stream", "access-control-allow-origin": "*" } }),
  );
  await page.route(`${CONTROL}/**`, (route) => route.abort());
  const cdp = await context.newCDPSession(page);
  await page.goto(`/?live=${encodeURIComponent(STREAM)}&control=${encodeURIComponent(CONTROL)}`);
  await expect(page.locator(".status")).toHaveText("ended", { timeout: 60_000 });
  await cdp.send("HeapProfiler.collectGarbage");
  const heap = (await cdp.send("Runtime.getHeapUsage")).usedSize;
  const nodes = (await cdp.send("Memory.getDOMCounters")).nodes;
  await context.close();
  return { heap, nodes };
}

test("the hosted replay is interactive within 1 s of navigation", async ({ browser }) => {
  const times: number[] = [];
  for (let i = 0; i < COLD_LOADS; i++) {
    const context = await browser.newContext();
    const page = await context.newPage();
    times.push(await interactiveAfter(page));
    await context.close();
  }
  const sorted = [...times].sort((a, b) => a - b);
  const median = sorted[sorted.length >> 1]!;
  result.interactive = {
    cold_loads_ms: times.map(round),
    median_ms: round(median),
    budget: { median_ms_max: INTERACTIVE_MAX_MS },
    pass: median <= INTERACTIVE_MAX_MS,
  };
  expect(median).toBeLessThanOrEqual(INTERACTIVE_MAX_MS);
});

/** Ms from navigation start until the replay panel shows the trace's duration. */
async function interactiveAfter(page: Page): Promise<number> {
  await page.goto(HOSTED, { waitUntil: "commit" });
  const ready = await page.waitForFunction(
    () => {
      const line = document.querySelector(".replay .line")?.textContent ?? "";
      return /\/ (?!0:00\.0$)/.test(line) ? performance.now() : null;
    },
    null,
    { polling: "raf" },
  );
  return (await ready.jsonValue()) as number;
}

function median(values: number[]): number {
  return [...values].sort((a, b) => a - b)[values.length >> 1]!;
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
