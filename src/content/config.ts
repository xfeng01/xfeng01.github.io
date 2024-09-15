// Import utilities from `astro:content`
import { z, defineCollection } from "astro:content";
// Define a `type` and `schema` for each collection
const postsCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    published: z.date(),
    description: z.string(),
    tags: z.array(z.string()),
    category: z.string(),
    draft: z.boolean().default(false),
  }),
});

const paperCollection = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    published: z.date(),
    description: z.string().optional().default(""),
    tags: z.array(z.string()).optional().default([]),
    category: z.string().optional().default(""),
    draft: z.boolean().optional().default(false),
  }),
});

// Export a single `collections` object to register your collection(s)
export const collections = {
  posts: postsCollection,
  paper: paperCollection,
};
