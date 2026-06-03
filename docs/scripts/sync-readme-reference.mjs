import { mkdir, readdir, readFile, rm, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const docsRoot = path.resolve(__dirname, "..");
const repoRoot = path.resolve(docsRoot, "..");
const contentRoot = path.join(docsRoot, "content/docs");

const languages = ["en", "zh"];
const githubBlobBase = "https://github.com/haonan3/UniRL/blob/main";

// Output locations from the previous "README Reference" layout. They are removed
// on every run so the old nested section never lingers next to the new flat pages.
const legacyDirs = [
  path.join(contentRoot, "en/reference/readmes"),
  ...["architecture", "configuration", "guides", "getting-started"].map((section) =>
    path.join(contentRoot, "en", section, "readme-reference"),
  ),
];

// Each README is promoted to a normal page inside its owning section. The page
// file is `readme-<slug>.mdx`, which keeps a single git-ignore glob while the
// sidebar label comes from the `title` frontmatter.
const readmes = [
  {
    source: "README.md",
    section: "getting-started",
    slug: "project",
    title: "Project README",
    description: "Project quick start, documentation entry points, checks, and citation.",
  },
  {
    source: "docs/README.md",
    section: "getting-started",
    slug: "docs-site",
    title: "Docs Site README",
    description: "Fumadocs site commands, structure, and maintenance notes.",
  },
  {
    source: "unirl/config/README.md",
    section: "configuration",
    slug: "config-package",
    title: "Config Package",
    description: "Config registration, instantiation, validation, and extension contracts.",
  },
  {
    source: "unirl/README.md",
    section: "architecture",
    slug: "code-architecture",
    title: "Code Architecture",
    description: "Runtime loop, module map, data flow, and package boundaries.",
  },
  {
    source: "unirl/rollout/README.md",
    section: "architecture",
    slug: "rollout",
    title: "Rollout",
    description: "Rollout modes, engines, request planning, and response contracts.",
  },
  {
    source: "unirl/train/readme.md",
    section: "architecture",
    slug: "train-stack",
    title: "Train Stack",
    description:
      "v2 single-stage train stack: FSDPBackend, TrainStack, structural injection, EMA shadow, and the train-step contract.",
  },
  {
    source: "unirl/algorithms/README.md",
    section: "architecture",
    slug: "algorithms",
    title: "Algorithms",
    description: "Train-side loss algorithm contracts and the reward-to-gradient path.",
  },
  {
    source: "unirl/sde/README.md",
    section: "architecture",
    slug: "sde",
    title: "SDE",
    description: "SDE strategy rules, schedules, kernels, and log-probability paths.",
  },
  {
    source: "unirl/distributed/weight_sync/README.md",
    section: "architecture",
    slug: "weight-sync",
    title: "Weight Sync",
    description: "Trainer-to-rollout weight synchronization backends and contracts.",
  },
  {
    source: "unirl/reward/README.md",
    section: "guides",
    slug: "reward-package",
    title: "Reward Package",
    description: "Reward service, backends, scorers, and the reward extension workflow.",
  },
  {
    source: "unirl/models/README.md",
    section: "guides",
    slug: "models",
    title: "Models",
    description: "Model bundle, stage, condition, and per-model package contracts.",
  },
  {
    source: "unirl-reward-service/README.md",
    section: "guides",
    slug: "reward-service",
    title: "Reward Service",
    description: "Standalone remote reward service: scorers, HTTP API, and deployment.",
  },
];

function stripTopLevelHeading(markdown) {
  return markdown.replace(/^# .*(?:\r?\n){1,2}/, "");
}

function rewriteRelativeMarkdownLinks(entry, markdown) {
  const sourceDir = path.posix.dirname(entry.source);

  return markdown.replace(
    /(?<!!)\[([^\]]+)\]\((?!#|https?:\/\/|mailto:|\/)([^)\s]+\.md(?:#[^)]+)?)\)/g,
    (_match, label, target) => {
      const [targetPath, anchor = ""] = target.split("#");
      const resolvedPath = path.posix.normalize(path.posix.join(sourceDir, targetPath));
      const resolvedAnchor = anchor ? `#${anchor}` : "";

      return `[${label}](${githubBlobBase}/${resolvedPath}${resolvedAnchor})`;
    },
  );
}

function frontmatterValue(value) {
  return JSON.stringify(value);
}

function renderPage(entry, body) {
  const content = stripTopLevelHeading(rewriteRelativeMarkdownLinks(entry, body)).trim();
  const sourceUrl = `${githubBlobBase}/${entry.source}`;

  return `---
title: ${frontmatterValue(entry.title)}
description: ${frontmatterValue(entry.description)}
---

{/* Generated from ${entry.source} by docs/scripts/sync-readme-reference.mjs. Edit the source README, not this file. */}

${content}

> Source: [\`${entry.source}\`](${sourceUrl}) — edit the README next to the code, then run \`npm run sync:readmes\` from \`docs/\`.
`;
}

async function removeGeneratedPages(dir) {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return;
  }

  await Promise.all(
    entries
      .filter((entry) => entry.isFile() && entry.name.startsWith("readme-") && entry.name.endsWith(".mdx"))
      .map((entry) => unlink(path.join(dir, entry.name))),
  );
}

async function main() {
  for (const dir of legacyDirs) {
    await rm(dir, { recursive: true, force: true });
  }

  const sections = new Set(readmes.map((entry) => entry.section));
  for (const lang of languages) {
    for (const section of sections) {
      await removeGeneratedPages(path.join(contentRoot, lang, section));
    }
  }

  let count = 0;
  for (const entry of readmes) {
    const body = await readFile(path.join(repoRoot, entry.source), "utf8");
    const page = renderPage(entry, body);

    for (const lang of languages) {
      const dir = path.join(contentRoot, lang, entry.section);
      await mkdir(dir, { recursive: true });
      await writeFile(path.join(dir, `readme-${entry.slug}.mdx`), page);
      count += 1;
    }
  }

  console.log(
    `Generated ${count} embedded README pages (${readmes.length} READMEs x ${languages.length} languages)`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
