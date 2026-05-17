import type { CollectionEntry } from 'astro:content';

type BlogPost = CollectionEntry<'blog'>;
type BlogLang = 'en' | 'zh';

export const getBlogPostLang = (post: BlogPost): BlogLang => post.data.lang ?? 'en';

export const getBlogPostSlug = (post: BlogPost) => post.data.translationKey ?? post.slug;

export const getBlogPostHref = (post: BlogPost) => {
  const slug = getBlogPostSlug(post);

  return getBlogPostLang(post) === 'zh' ? `/blog/zh/${slug}/` : `/blog/${slug}/`;
};
