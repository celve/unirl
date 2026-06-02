import { getLLMText } from "@/lib/get-llm-text";
import { source } from "@/lib/source";
import { llms } from "fumadocs-core/source";

function getBuildSha() {
  return process.env.SOURCE_COMMIT ?? process.env.VERCEL_GIT_COMMIT_SHA ?? "unknown";
}

export function getLLMSIndexText() {
  const [, ...indexLines] = llms(source).index("en").split("\n");
  const buildSha = getBuildSha();

  return `# UniRL

> Agent-readable index for UniRL, built from commit: ${buildSha}.

${indexLines.join("\n").trim()}

## Agent Access Patterns

- Treat this file as a compact discovery endpoint, not as a docs category.
- Start with \`/md/agents/index.md\` for task routing in Markdown, or \`/en/docs/agents\` when reading the rendered site.
- Use \`/llms-full.txt\` for a single-file Markdown corpus.
- Use \`/md/<docs-slug>/index.md\` for one page as Markdown when focused context is better.

## Authoritative Runtime Entry

- Training entrypoints: \`python -m unirl.train_diffusion --config-name=<recipe>\` (also \`train_vlm\`, \`train_pe\`, \`train_hi3\`).
- Recipes: self-contained \`conf/<recipe>.yaml\` files in a flat \`conf/\` directory, selected with \`--config-name\`.
`;
}

export async function getLLMSFullText() {
  const buildSha = getBuildSha();
  const pages = await Promise.all(source.getPages("en").map(getLLMText));
  const header = `# UniRL Full Documentation

> Agent-readable Markdown corpus built from commit: ${buildSha}.`;

  return [header, ...pages].join("\n\n---\n\n");
}
