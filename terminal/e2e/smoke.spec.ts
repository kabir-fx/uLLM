import { expect, test } from "@playwright/test";

test("the hosted replay opens, plays and draws in a clean browser", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });

  await page.goto("/");

  await expect(page.locator(".badge-replay")).toHaveText("REPLAY");
  await expect(page.locator(".header")).toContainText("Apple M4");
  await expect(page.locator(".status")).toHaveText("open");
  await expect(page.locator("tr[data-row]").first()).toBeVisible();
  await expect(page.locator('[data-stat="tps"] .stat-value')).not.toHaveText("—");

  const clock = page.locator(".replay .line");
  const before = await clock.textContent();
  await expect(clock).not.toHaveText(before!);

  // Labels and grid lines are drawn with no data; only token bars and strip cells use these colours.
  for (const panel of ["step chart", "batch strip"]) {
    const canvas = page.getByLabel(panel).locator("canvas");
    const tokenPixels = () =>
      canvas.evaluate((canvas: HTMLCanvasElement) => {
        const css = getComputedStyle(canvas);
        const rgb = (name: string) => parseInt(css.getPropertyValue(name).trim().slice(1), 16);
        const colours = new Set([rgb("--prefill"), rgb("--decode")]);
        const { data } = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);
        let n = 0;
        for (let i = 0; i < data.length; i += 4) if (colours.has((data[i]! << 16) | (data[i + 1]! << 8) | data[i + 2]!)) n++;
        return n;
      });
    await expect.poll(tokenPixels, { message: `prefill and decode pixels in the ${panel}` }).toBeGreaterThan(500);
  }
  expect(errors).toEqual([]);
});
