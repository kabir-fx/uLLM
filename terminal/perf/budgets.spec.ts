import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page } from "@playwright/test";
import { fixtureFrames } from "../src/testing/fixtures";
import { repeatTrace } from "../src/testing/repeat";

// The T7 budgets (decision 0027). `make terminal-perf` sets PERF_OUT and commits what lands there.
const HOSTED = "http://127.0.0.1:4173/";
const FPS_MIN = 58;
const FRAME_P99_MAX_MS = 25;
const REPEATS = 10;
const GROWTH_MAX = 1.1;
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

test("flat memory over the stress trace streamed 10 times with no reset", async ({ page }) => {
  const frames = repeatTrace(await fixtureFrames("stress.jsonl.gz"), REPEATS);
  const body = frames.map((f) => JSON.stringify(f)).join("\n");
  await page.route("**/long.jsonl", (route) => route.fulfill({ body, contentType: "application/x-ndjson" }));
  const cdp = await page.context().newCDPSession(page);

  await page.goto("/?trace=long.jsonl&speed=16");
  await expect(page.locator(".status")).toHaveText("open");
  // V8 shrinks a parsed frame's strings once the store has read them, so the heap falls while
  // a trace plays for the first time. Delivering every frame first takes that out.
  const position = page.getByLabel("position");
  await page.keyboard.press("Space");
  await position.fill((await position.getAttribute("max"))!);
  await position.fill("0");
  await page.locator(".replay button").click();
  await expect(page.locator(".status")).toHaveText("open");
  const period = (frames.at(-1)!.t - frames[0]!.t) / REPEATS;
  const heap: number[] = [];
  const nodes: number[] = [];
  for (let k = 1; k <= REPEATS; k++) {
    await expect.poll(async () => Number(await position.inputValue()), { timeout: 60_000 }).toBeGreaterThanOrEqual(k * period - 1);
    await cdp.send("HeapProfiler.collectGarbage");
    heap.push((await cdp.send("Runtime.getHeapUsage")).usedSize);
    nodes.push((await cdp.send("Memory.getDOMCounters")).nodes);
  }

  const heapGrowth = heap.at(-1)! / heap[1]!;
  const nodeGrowth = nodes.at(-1)! / nodes[1]!;
  result.memory = {
    repeats: REPEATS,
    speed: 16,
    heap_bytes: heap,
    dom_nodes: nodes,
    heap_last_over_second: round(heapGrowth),
    nodes_last_over_second: round(nodeGrowth),
    budget: { growth_max: GROWTH_MAX },
    pass: heapGrowth <= GROWTH_MAX && nodeGrowth <= GROWTH_MAX,
  };
  expect(heapGrowth).toBeLessThanOrEqual(GROWTH_MAX);
  expect(nodeGrowth).toBeLessThanOrEqual(GROWTH_MAX);
});

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

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
