import { expect, test } from "@playwright/test";
import path from "node:path";

// Full path: Browser -> Frontend -> API -> RAG -> Gemini -> Response -> Frontend.
// Uses the real backend and the real LLM (see playwright.config.ts) — this is
// a deliberate, occasional live check, not a fast unit test.

const FORMULA_PDF = path.resolve(__dirname, "../../data/formula_sample.pdf");

test("upload a document, ask a question, and ask a follow-up", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Knowledge Assistant" })).toBeVisible();
  await page.screenshot({ path: "e2e/screenshots/01-initial.png", fullPage: true });

  await page.locator('input[type="file"]').setInputFiles(FORMULA_PDF);
  await expect(page.getByText("Indexed")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/chunks/)).toBeVisible();
  await page.screenshot({ path: "e2e/screenshots/02-indexed.png", fullPage: true });

  const input = page.getByPlaceholder("Ask a question about your document…");
  await expect(input).toBeEnabled();
  await input.fill("What is the compound interest formula?");
  await input.press("Enter");

  await expect(page.getByText("Retrieving evidence and writing the answer…")).toBeVisible();
  const firstAnswer = page.locator(".message-assistant").first();
  await expect(firstAnswer).toBeVisible({ timeout: 30_000 });
  await expect(firstAnswer).toContainText(/\[S\d+\]/, { timeout: 30_000 });
  await page.screenshot({ path: "e2e/screenshots/03-first-answer.png", fullPage: true });

  await page.getByText("Sources and retrieval details").first().click();
  await expect(page.getByText(/Evidence level:/)).toBeVisible();

  await input.fill("Explain it in more detail.");
  await input.press("Enter");

  const assistantMessages = page.locator(".message-assistant");
  await expect(assistantMessages).toHaveCount(2, { timeout: 30_000 });
  const secondAnswer = assistantMessages.nth(1);
  await secondAnswer.getByText("Sources and retrieval details").click();
  await expect(secondAnswer.getByText("Follow-up detected")).toBeVisible();
  await page.screenshot({ path: "e2e/screenshots/04-follow-up.png", fullPage: true });

  const seriousErrors = consoleErrors.filter((e) => !e.includes("Download the React DevTools"));
  expect(seriousErrors, `console errors:\n${seriousErrors.join("\n")}`).toHaveLength(0);
});

test("rejects an unsupported file type with a clear error banner", async ({ page }) => {
  await page.goto("/");
  const badFile = path.resolve(__dirname, "bad-upload.txt");
  await page.locator('input[type="file"]').setInputFiles(badFile);
  await expect(page.locator(".error-banner")).toContainText("Unsupported file type");
});

test("declines without fabricating an answer when evidence is insufficient", async ({ page }) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles(FORMULA_PDF);
  await expect(page.getByText("Indexed")).toBeVisible({ timeout: 30_000 });

  const input = page.getByPlaceholder("Ask a question about your document…");
  await input.fill("What is the parental leave policy?");
  await input.press("Enter");

  const answer = page.locator(".message-assistant").first();
  await expect(answer).toContainText("couldn't find enough information", { timeout: 30_000 });
  // No low-confidence caveat on a declined answer — refusal already says it plainly.
  await expect(page.locator(".low-confidence-caption")).toHaveCount(0);

  await answer.getByText("Sources and retrieval details").click();
  await expect(answer.getByText("Evidence level:")).toContainText("none");
});

test("shows a clear message when the backend is unreachable, and recovers", async ({ page }) => {
  await page.route("**/health", (route) => route.abort("connectionrefused"));
  await page.goto("/");
  await expect(page.getByText(/Can't reach the API backend/)).toBeVisible();

  await page.unroute("**/health");
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByRole("heading", { name: "Knowledge Assistant" })).toBeVisible();
});
