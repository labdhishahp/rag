import { defineConfig } from "@playwright/test";
import path from "node:path";

// Live end-to-end verification: Browser -> Next.js frontend -> FastAPI
// backend -> RAG core -> Gemini -> response -> frontend. Starts BOTH
// services itself. Requires the project-root .env to have a working
// GEMINI_API_KEY — this hits the real model, not a fake, so run it
// deliberately (`npm run test:e2e`), not on every save.
const REPO_ROOT = path.resolve(__dirname, "..");

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3000",
    screenshot: "on",
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `${path.join(REPO_ROOT, ".venv/bin/uvicorn")} api.main:app --port 8000`,
      cwd: path.join(REPO_ROOT, "backend"),
      port: 8000,
      reuseExistingServer: true,
      timeout: 60_000,
      stdout: "pipe",
      stderr: "pipe",
    },
    {
      command: "npm run dev",
      cwd: __dirname,
      port: 3000,
      reuseExistingServer: true,
      timeout: 60_000,
      stdout: "pipe",
      stderr: "pipe",
    },
  ],
});
