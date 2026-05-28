import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiHost = process.env.MODEL_TESTER_API_HOST || "127.0.0.1";
const apiPort = process.env.MODEL_TESTER_API_PORT || "8000";
const apiTarget = `http://${apiHost}:${apiPort}`;

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": apiTarget,
      "/models": apiTarget,
      "/analyze-image": apiTarget,
      "/analyze-image-pipelines": apiTarget,
      "/analyze-video": apiTarget,
      "/result": apiTarget
    }
  }
});
