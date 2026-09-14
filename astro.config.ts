import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "astro/config";

// https://astro.build/config
export default defineConfig({
  site: "https://tandat8896.github.io",
  vite: {
    plugins: [tailwindcss()],
  },
  output: "static",
});
