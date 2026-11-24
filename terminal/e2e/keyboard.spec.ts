import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".status")).toHaveText("open");
});

test("Space pauses and plays the replay", async ({ page }) => {
  await page.keyboard.press("Space");
  await expect(page.locator(".status")).toHaveText("paused");
  await expect(page.locator(".replay button")).toHaveText("play");
  await page.keyboard.press("Space");
  await expect(page.locator(".status")).toHaveText("open");
  await expect(page.locator(".replay button")).toHaveText("pause");
});

test("the arrow keys seek 5 s of trace time", async ({ page }) => {
  await page.keyboard.press("Space");
  const position = page.getByLabel("position");
  const at = async () => Number(await position.inputValue());
  const start = await at();
  await page.keyboard.press("ArrowRight");
  await expect.poll(at).toBeCloseTo(start + 5000, 0);
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("ArrowLeft");
  await expect.poll(at).toBeCloseTo(Math.max(0, start - 5000), 0);
});

test("Space on a focused button presses only that button", async ({ page }) => {
  await page.locator(".replay button").focus();
  await page.keyboard.press("Space");
  await expect(page.locator(".status")).toHaveText("paused");
});
