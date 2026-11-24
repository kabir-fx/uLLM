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

  for (const panel of ["step chart", "batch strip"]) {
    const canvas = page.getByLabel(panel).locator("canvas");
    const painted = () => canvas.evaluate((canvas: HTMLCanvasElement) => {
      const { data } = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);
      let n = 0;
      for (let i = 3; i < data.length; i += 4) if (data[i]! > 0) n++;
      return n / (canvas.width * canvas.height);
    });
    await expect.poll(painted, { message: `share of the ${panel} drawn` }).toBeGreaterThan(0.01);
  }
  expect(errors).toEqual([]);
});
