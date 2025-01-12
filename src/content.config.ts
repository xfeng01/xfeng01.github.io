import { z, defineCollection } from "astro:content";

import { glob, file } from "astro/loaders";

const posts = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
  schema: z.object({
    title: z.string(),
    published: z.date(),
    description: z.string(),
    tags: z.array(z.string()),
    category: z.string(),
    draft: z.boolean().default(false),
  }),
});

const papers = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/papers" }),
  schema: z.object({
    title: z.string(),
    published: z.date(),
    tags: z.array(z.string()).optional().default([]),
    authors: z.array(z.string()).optional().default([]),
    category: z.string().optional().default(""),
    draft: z.boolean().optional().default(false),
  }),
});

export const collections = {posts, papers}