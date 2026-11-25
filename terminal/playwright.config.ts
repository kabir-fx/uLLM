import { defineConfig, devices } from "@playwright/test";

// Decision 0027: `smoke` drives the hosted build and runs in `make check-ts`; `perf` checks
// the T7 budgets on the local build, which accepts `?trace=`, and runs only in `make terminal-perf`.
// `demo` records the README's GIF from the hosted build, only in `make demo-gif` (decision 0030).
// Both servers serve builds that the Makefile made first, on ports apart from `vite preview`'s 4173.
const HOSTED = "http://127.0.0.1:4183";
const LOCAL = "http://127.0.0.1:4184";

export default defineConfig({
  forbidOnly: true,
  reporter: "list",
  use: { ...devices["Desktop Chrome"], viewport: { width: 1600, height: 1000 } },
  projects: [
    { name: "smoke", testDir: "e2e", use: { baseURL: HOSTED } },
    { name: "perf", testDir: "perf", use: { baseURL: LOCAL }, timeout: 300_000 },
    { name: "demo", testDir: "demo", use: { baseURL: HOSTED } },
  ],
  webServer: [
    { command: "vite preview --outDir dist-hosted --port 4183", url: HOSTED, reuseExistingServer: false },
    { command: "vite preview --port 4184", url: LOCAL, reuseExistingServer: false },
  ],
});
