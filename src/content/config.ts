import { defineCollection, z } from 'astro:content';

const aboutCollection = defineCollection({
  type: 'content',
  schema: z.object({}),
});

const projectsCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    authors: z.array(z.string()),
    pub: z.string(),
    date: z.date(),
    description: z.string().optional(),
    image: z.string().optional(),
  paper: z.string().optional(),
    code: z.string().optional(),
    poster: z.string().optional(),
    blog: z.string().optional(),
  }),
});

export const collections = {
  'about': aboutCollection,
  'projects': projectsCollection,
};
