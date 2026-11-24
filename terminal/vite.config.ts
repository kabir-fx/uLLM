import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Rule 14: local by default. `--mode hosted` is the public replay build (decision 0026):
// a relative base, so it works under any path, and its own output directory.
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  publicDir: "fixtures",
  base: mode === "hosted" ? "./" : "/",
  build: { outDir: mode === "hosted" ? "dist-hosted" : "dist" },
  server: { host: "127.0.0.1", port: 5173, strictPort: true },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true },
  test: { environment: "node", include: ["src/**/*.test.{ts,tsx}"] },
}));
