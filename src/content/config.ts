import { defineCollection, z } from 'astro:content';

const aboutCollection = defineCollection({
  type: 'content',
  schema: z.object({
    // You can define frontmatter properties here if needed
    // For example: title: z.string()
  }),
});

export const collections = {
  'about': aboutCollection,
};
