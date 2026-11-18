import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Rule 14: local by default.
export default defineConfig({
  plugins: [react()],
  publicDir: "fixtures",
  server: { host: "127.0.0.1", port: 5173, strictPort: true },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true },
  test: { environment: "node", include: ["src/**/*.test.{ts,tsx}"] },
});
