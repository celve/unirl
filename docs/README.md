# UniRL Docs Site

This directory contains the Fumadocs + Next.js documentation site. It is isolated from the Python package so Node dependencies and generated files do not affect framework installs.

## Commands

```bash
npm install
npm run sync:readmes
npm run dev
npm run build
npm run typecheck
```

`npm run sync:readmes` regenerates the README Reference pages from the
repository README files. It also runs automatically before `dev`, `build`, and
`typecheck`.

`npm run build` statically exports the site to `out/`. The build also generates `.source/`, which backs the `collections/server` import used by Fumadocs MDX.

## Structure

| Path | Purpose |
|---|---|
| `app/` | Next.js App Router pages and static route handlers |
| `content/docs/` | MDX source pages and `meta.json` sidebar ordering |
| `lib/source.ts` | Fumadocs loader source |
| `lib/get-llm-text.ts` | Markdown conversion for agent endpoints |
| `components/mdx.tsx` | MDX component mapping |
| `source.config.ts` | Fumadocs MDX collection config |
| `scripts/sync-readme-reference.mjs` | Generates README Reference pages from near-code README files |

## Agent Endpoints

Static builds expose:

- `/llms.txt` for a compact documentation index.
- `/llms-full.txt` for the full Markdown corpus.
- `/md/<slug>/index.md` for one page as Markdown, for example `/md/configuration/hydra/index.md`.
- `/api/search.json` for the static search index consumed by Fumadocs UI.

Generated README Reference pages are included in these outputs after
`npm run sync:readmes`.

Keep these routes extension-safe so static hosts can infer MIME types without custom server configuration.

Agent endpoints intentionally read only the English source (`content/docs/en`). Chinese pages are for human reading and are not included in `/llms-full.txt` or `/md/<slug>/index.md`.

Do not add `llms.txt` as a docs sidebar category. The rendered `Agents` section explains how agents should navigate the docs; `/llms.txt` and related routes stay as root-level machine-readable endpoints.

## Human Languages

Human documentation is language-prefixed:

- `/en/docs` for English.
- `/zh/docs` for Chinese.

Content is organized by directory:

```text
content/docs/en/...
content/docs/zh/...
```

English is the source of truth and fallback language. If a Chinese page is missing, Fumadocs falls back to the English source.

## Adding Pages

1. Add the English `.mdx` file under `content/docs/en/`.
2. Add the Chinese `.mdx` file under `content/docs/zh/` when the page is intended for human Chinese readers.
3. Add or update the nearest `meta.json` `pages` list to control sidebar order.
4. Prefer paths and command examples that match the current repository state.
5. Run `npm run build` before submitting changes.

## Maintenance Notes

- Node `>=20.19.0` is declared because current transitive file-watcher dependencies require it. Older Node versions may still build but will warn during install.
- `includeProcessedMarkdown` is enabled in `source.config.ts`; do not remove it unless `/llms-full.txt` and `/md/<slug>/index.md` are replaced with another Markdown source.
- Generated directories `.next/`, `.source/`, `out/`, and `node_modules/` are ignored by the repository.
