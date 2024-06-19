import { defineConfig } from 'astro/config';
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkSlug from "remark-slug";
import tailwind from "@astrojs/tailwind";

// https://astro.build/config
export default defineConfig({
  site: "https://pride7.github.io",
  base: "/",
  markdown: {
    remarkPlugins: [remarkMath, remarkSlug],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    tailwind({
      nesting: true,
    }),
  ],
});