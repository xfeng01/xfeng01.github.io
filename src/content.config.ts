import { z, defineCollection } from "astro:content";

import { glob, file } from "astro/loaders";

const posts = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
  schema: z.object({
    title: z.string(),
    authors: z.array(z.string()).optional().default([]),
    published: z.date(),
    description: z.string(),
    tags: z.array(z.string()).optional().default([]),
    category: z.string().optional().default(""),
    draft: z.boolean().default(false),
  }),
});

export const collections = {posts}