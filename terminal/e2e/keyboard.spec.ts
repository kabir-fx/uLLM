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

test("keys aimed at a tab or a select keep their own meaning and leave the replay alone", async ({ page }) => {
  await page.keyboard.press("Space");
  const position = page.getByLabel("position");
  const at = await position.inputValue();
  await page.locator('[role="tab"][data-tab="running"]').focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator('[role="tab"][aria-selected="true"]')).toHaveAttribute("data-tab", "waiting");
  await page.getByLabel("speed").focus();
  await page.keyboard.press("Space");
  await expect(page.locator(".status")).toHaveText("paused");
  expect(await position.inputValue()).toBe(at);
});

test("the requests tabs follow the WAI-ARIA tabs pattern", async ({ page }) => {
  const selected = page.locator('[role="tab"][aria-selected="true"]');
  await page.locator('[role="tab"][data-tab="running"]').focus();
  await page.keyboard.press("ArrowRight");
  await expect(selected).toHaveAttribute("data-tab", "waiting");
  await expect(selected).toBeFocused();
  await page.keyboard.press("End");
  await expect(selected).toHaveAttribute("data-tab", "preempted");
  await page.keyboard.press("ArrowRight");
  await expect(selected).toHaveAttribute("data-tab", "running");
  await page.keyboard.press("ArrowLeft");
  await expect(selected).toHaveAttribute("data-tab", "preempted");
  await page.keyboard.press("Home");
  await expect(selected).toHaveAttribute("data-tab", "running");
  await expect(page.locator('[role="tab"][tabindex="0"]')).toHaveCount(1);
});

test("Tab reaches every control once, and each shows a focus ring", async ({ page }) => {
  const reached: string[] = [];
  for (let i = 0; i < 20; i++) {
    await page.keyboard.press("Tab");
    const focused = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body) return null;
      const style = getComputedStyle(el);
      return { name: el.getAttribute("aria-label") ?? el.getAttribute("name") ?? el.textContent!.trim(), ring: style.outlineStyle !== "none" && parseFloat(style.outlineWidth) >= 2 };
    });
    if (focused === null || reached.includes(focused.name)) break;
    expect(focused.ring, `focus ring on ${focused.name}`).toBe(true);
    reached.push(focused.name);
  }
  expect(reached).toEqual([expect.stringMatching(/^running/), "run", "position", "pause", "speed"]);
});
