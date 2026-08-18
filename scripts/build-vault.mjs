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

    notes.push({
      slug: relative.replace(/\.mdx?$/, ''),
      title: frontmatter.title ?? path.basename(relative).replace(/\.mdx?$/, ''),
      date: frontmatter.date ? new Date(frontmatter.date).toISOString() : null,
      tags: Array.isArray(frontmatter.tags) ? frontmatter.tags : [],
      description: frontmatter.description ?? '',
      html: rendered.code,
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
