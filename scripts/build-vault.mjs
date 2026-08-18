#!/usr/bin/env node
/**
 * 私密空间的构建器。
 *
 * 把 private/ 下的明文 Markdown 渲染成 HTML（和博客走同一套 remark/rehype 管线，
 * 所以数学公式、代码、表格的样式完全一致），再用口令派生出的密钥整体加密，
 * 输出到 public/vault.json。
 *
 * 明文和口令都不进仓库，只有密文进。CI 不需要知道任何密钥。
 *
 * 用法：
 *   npm run vault                     交互式输入口令
 *   VAULT_PASSWORD=xxx npm run vault  从环境变量读（适合脚本里用）
 *   npm run vault -- --no-compress    关掉 gzip（浏览器太老时的退路）
 */

import { createMarkdownProcessor, parseFrontmatter } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import katex from 'katex';
import { webcrypto as crypto } from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SOURCE_DIR = path.join(ROOT, 'private');
const OUTPUT_FILE = path.join(ROOT, 'public', 'vault.json');

// PBKDF2 的迭代次数。密文是公开可下载的，口令强度是唯一防线，
// 所以这里取浏览器上还能接受的高档位（解密时约 0.3–1 秒）。
const PBKDF2_ITERATIONS = 310000;
const compress = !process.argv.includes('--no-compress');

const fail = (message) => {
  console.error(`\n✗ ${message}\n`);
  process.exit(1);
};

const escapeHtml = (value) =>
  value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

// 标题和摘要里的 $...$ 在构建时就渲染成 KaTeX，
// 和 src/components/MathText.astro 的做法保持一致。
const renderInlineMath = (source) =>
  String(source)
    .split(/(\$[^$]+\$)/g)
    .map((part) =>
      part.startsWith('$') && part.endsWith('$') && part.length > 2
        ? katex.renderToString(part.slice(1, -1), {
            displayMode: false,
            throwOnError: false,
            strict: false,
          })
        : escapeHtml(part),
    )
    .join('');

// 把标题里的行内公式、链接、强调标记洗掉，只留可读文本。
// 逻辑对齐 src/components/BlogArticle.astro 里的 cleanTocText。
const cleanTocText = (text) => {
  const mathExpressions = [];
  const withPlaceholders = text.replace(/\$([^$]+)\$/g, (_, expression) => {
    const normalized = String(expression)
      .replace(/\\([a-zA-Z]+)/g, '$1')
      .replace(/\s+/g, ' ')
      .trim();
    mathExpressions.push(normalized);
    return `@@MATH${mathExpressions.length - 1}@@`;
  });

  return withPlaceholders
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*`]/g, '')
    .replace(/(^|[\s([{])_([^_]+)_($|[\s)\]}.,:;!?])/g, '$1$2$3')
    .replace(/@@MATH(\d+)@@/g, (_, index) => mathExpressions[Number(index)] ?? '')
    .replace(/\s+/g, ' ')
    .trim();
};

// 用原始 markdown 的标题文本（洗干净）配上渲染器生成的 slug，两边顺序一一对应。
const buildToc = (markdown, renderedHeadings) => {
  const rendered = renderedHeadings.filter((h) => h.depth === 2 || h.depth === 3);
  const raw = Array.from(markdown.matchAll(/^(#{2,3})\s+(.*)$/gm), (match) => ({
    depth: match[1].length,
    text: cleanTocText(match[2]),
  }));

  return raw
    .map((item, index) => ({ ...item, slug: rendered[index]?.slug ?? "" }))
    .filter((item) => item.slug);
};

async function collectMarkdownFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectMarkdownFiles(full)));
    } else if (entry.isFile() && /\.mdx?$/.test(entry.name) && !entry.name.startsWith('_')) {
      files.push(full);
    }
  }

  return files.sort();
}

async function promptPassword(label) {
  if (!process.stdin.isTTY) {
    fail('当前不是交互式终端，请改用 VAULT_PASSWORD 环境变量传口令。');
  }

  process.stdout.write(label);

  return new Promise((resolve) => {
    const stdin = process.stdin;
    let buffer = '';

    stdin.setRawMode(true);
    stdin.resume();
    stdin.setEncoding('utf8');

    const onData = (chunk) => {
      for (const char of chunk) {
        const code = char.charCodeAt(0);

        // 回车 / 换行 / Ctrl-D：输入结束
        if (code === 13 || code === 10 || code === 4) {
          stdin.setRawMode(false);
          stdin.pause();
          stdin.removeListener('data', onData);
          process.stdout.write('\n');
          resolve(buffer);
          return;
        }

        // Ctrl-C：放弃
        if (code === 3) {
          stdin.setRawMode(false);
          process.stdout.write('\n');
          process.exit(130);
        }

        // Backspace / Delete
        if (code === 8 || code === 127) {
          buffer = buffer.slice(0, -1);
        } else if (code >= 32) {
          buffer += char;
        }
      }
    };

    stdin.on('data', onData);
  });
}

async function resolvePassword() {
  const fromEnv = process.env.VAULT_PASSWORD;
  if (fromEnv) {
    if (fromEnv.length < 12) fail('VAULT_PASSWORD 太短了，至少 12 个字符。');
    return fromEnv;
  }

  const password = await promptPassword('口令: ');
  if (password.length < 12) fail('口令太短了，至少 12 个字符（建议 4 个以上随机单词）。');

  const confirmation = await promptPassword('再输一次: ');
  if (password !== confirmation) fail('两次输入不一致。');

  return password;
}

async function encrypt(plaintext, password) {
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));

  const baseKey = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    'PBKDF2',
    false,
    ['deriveKey'],
  );

  const key = await crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt, iterations: PBKDF2_ITERATIONS, hash: 'SHA-256' },
    baseKey,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt'],
  );

  const body = compress ? gzipSync(plaintext, { level: 9 }) : plaintext;
  const ciphertext = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, body);

  return {
    v: 1,
    kdf: { name: 'PBKDF2', hash: 'SHA-256', iterations: PBKDF2_ITERATIONS },
    compression: compress ? 'gzip' : 'none',
    salt: Buffer.from(salt).toString('base64'),
    iv: Buffer.from(iv).toString('base64'),
    data: Buffer.from(ciphertext).toString('base64'),
  };
}

async function main() {
  if (!existsSync(SOURCE_DIR)) fail(`找不到 ${path.relative(ROOT, SOURCE_DIR)}/ 目录。`);

  const files = await collectMarkdownFiles(SOURCE_DIR);
  if (files.length === 0) fail('private/ 里没有 .md 文件，没什么可加密的。');

  const processor = await createMarkdownProcessor({
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  });

  const notes = [];

  for (const file of files) {
    const relative = path.relative(SOURCE_DIR, file).replace(/\\/g, '/');
    const raw = await readFile(file, 'utf8');
    const { frontmatter, content } = parseFrontmatter(raw);
    const rendered = await processor.render(content);
    const title = frontmatter.title ?? path.basename(relative).replace(/\.mdx?$/, '');
    const description = frontmatter.description ?? '';
    const slug = relative.replace(/\.mdx?$/, '');
    // 语言和译文分组的语义完全对齐 src/utils/blog.ts：
    // lang 缺省是 en，translationKey 缺省退回文件名。
    const lang = frontmatter.lang === 'zh' ? 'zh' : 'en';
    const translationKey = frontmatter.translationKey ?? slug;

    notes.push({
      slug,
      lang,
      translationKey,
      title,
      // 页面是用 innerHTML 渲染的，所以这里把安全的 HTML 一并备好，
      // 前端不需要再做转义或数学渲染。
      titleHtml: renderInlineMath(title),
      date: frontmatter.date ? new Date(frontmatter.date).toISOString() : null,
      tags: Array.isArray(frontmatter.tags) ? frontmatter.tags.map(String) : [],
      description,
      descriptionHtml: renderInlineMath(description),
      html: rendered.code,
      toc: buildToc(content, rendered.metadata.headings ?? []),
    });

    console.log(`  · ${relative}`);
  }

  notes.sort((a, b) => {
    if (a.date && b.date) return b.date.localeCompare(a.date);
    if (a.date) return -1;
    if (b.date) return 1;
    return a.title.localeCompare(b.title);
  });

  const password = await resolvePassword();
  const plaintext = Buffer.from(JSON.stringify({ notes }), 'utf8');
  const envelope = await encrypt(plaintext, password);
  const serialized = JSON.stringify(envelope);

  await writeFile(OUTPUT_FILE, serialized, 'utf8');

  const sizeKb = (Buffer.byteLength(serialized) / 1024).toFixed(1);
  console.log(`\n✓ 已加密 ${notes.length} 篇 → public/vault.json (${sizeKb} KB)`);
  console.log('  接下来：git add public/vault.json && git commit && git push\n');
}

main().catch((error) => fail(error?.stack ?? String(error)));
