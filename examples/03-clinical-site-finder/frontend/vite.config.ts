import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` serves the app on :5174 and forwards /api to the FastAPI backend
export default defineConfig({
  plugins: [react()],
  server: { port: 5174, proxy: { "/api": "http://localhost:8001" } },
});
