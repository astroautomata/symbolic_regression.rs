import { test, expect } from "@playwright/test";

test("loads real WASM + worker and can run a tiny search", async ({ page }) => {
  page.on("pageerror", (err) => {
    throw err;
  });

  await page.goto("/");

  // Wait for WASM init (options become available and enable parsing).
  await expect(page.getByTestId("parse-preview")).toBeEnabled();

  // Parse the default CSV so plots can use local parsed data too.
  await page.getByRole("button", { name: "Data" }).click();
  await page.getByTestId("parse-preview").click();

  // Shrink search budget so CI stays fast.
  await page.getByRole("button", { name: "Configure" }).click();

  // Expand Advanced hyperparameters section to access fields inside.
  await page.getByText("Advanced hyperparameters").click();

  await page.getByTestId("opt-populations").fill("1");
  await page.getByTestId("opt-population-size").fill("16");
  await page.getByTestId("opt-ncycles").fill("20");

  // Run search.
  await page.getByRole("button", { name: "Run" }).click();

  // Shrink iterations right before initialize (options apply at init time).
  await page.getByTestId("opt-niterations").fill("1");

  await page.getByTestId("search-init").click();
  await expect(page.getByTestId("search-status")).toHaveText("ready");

  await page.getByTestId("search-start").click();
  await expect(page.getByTestId("search-status")).toHaveText(/done|paused|running/);

  // Wait for at least one solution and click it to trigger evaluation.
  const table = page.getByTestId("solutions-table");
  await expect(table).toBeVisible();

  // If the search is still running, give it a moment to publish the first front_update.
  await page.waitForTimeout(1500);

  const firstRow = table.locator("tbody tr").first();
  await expect(firstRow).toBeVisible();
  await firstRow.click();

  // Expect evaluation to populate selected equation (and remove "no metrics" state).
  await expect(page.getByTestId("selected-equation")).toBeVisible();
  await expect(page.getByTestId("no-metrics")).toHaveCount(0);
});

test("guesses seed the real search, and an invalid guess fails setup naming it", async ({ page }) => {
  page.on("pageerror", (err) => {
    throw err;
  });

  await page.goto("/");
  await expect(page.getByTestId("parse-preview")).toBeEnabled();

  await page.getByRole("button", { name: "Configure" }).click();
  await page.getByText("Advanced hyperparameters").click();
  await page.getByTestId("opt-populations").fill("1");
  await page.getByTestId("opt-population-size").fill("16");
  await page.getByTestId("opt-ncycles").fill("20");
  await page.getByRole("textbox", { name: "guesses", exact: true }).fill("1.0*cos(x0) + x1*x1 - 1.0\nz * 2.0");
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByTestId("search-init").click();
  await expect(page.getByTestId("search-status")).toHaveText("error");
  await expect(page.getByText("invalid guess `z * 2.0`: `z` is not a variable", { exact: false })).toBeVisible();

  await page.getByRole("button", { name: "Configure" }).click();
  await page.getByRole("textbox", { name: "guesses", exact: true }).fill("1.0*cos(x0) + x1*x1 - 1.0");
  await page.getByRole("button", { name: "Run" }).click();
  await page.getByTestId("opt-niterations").fill("1");
  await page.getByTestId("search-init").click();
  await expect(page.getByTestId("search-status")).toHaveText("ready");
  await page.getByTestId("search-start").click();
  await expect(page.getByTestId("search-status")).toHaveText("done", { timeout: 60_000 });
  // The default data is y = 2*cos(x0) + x1^2 - 2, so the guess with optimized constants fits exactly.
  await expect(page.getByTestId("solutions-table").locator("tbody tr").first()).toContainText("cos(x0)");
});
