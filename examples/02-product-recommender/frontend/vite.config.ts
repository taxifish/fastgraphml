import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` serves the app on :5173 and forwards /api to the FastAPI backend
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
