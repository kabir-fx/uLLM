import { expect, test } from "@playwright/test";

// Decision 0030: `make demo-gif` sets DEMO_OUT and turns this video into docs/assets/demo.gif.
const SIZE = { width: 1600, height: 1000 };

test("record the hosted replay of the default run", async ({ browser }, info) => {
  const out = process.env.DEMO_OUT;
  if (!out) throw new Error("DEMO_OUT names the video to write; run `make demo-gif`");
  const context = await browser.newContext({ viewport: SIZE, recordVideo: { dir: info.outputDir, size: SIZE } });
  const page = await context.newPage();
  await page.goto("/");
  await expect(page.locator(".badge-replay")).toHaveText("REPLAY");
  await expect(page.locator("tr[data-row]").first()).toBeVisible();
  await page.waitForTimeout(15_000);
  const video = page.video()!;
  await context.close();
  await video.saveAs(out);
});
