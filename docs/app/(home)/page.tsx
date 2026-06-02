import Link from "next/link";

const docLinks = [
  {
    href: "/en/docs",
    label: "English Docs",
    description: "Narrative docs for researchers and engineers.",
  },
  {
    href: "/zh/docs",
    label: "中文文档",
    description: "中文安装、运行和开发路径。",
  },
  {
    href: "/en/docs/agents",
    label: "Agent Index",
    description: "Human-readable map for how coding agents should use the docs.",
  },
];

const capabilities = [
  {
    title: "Distributed Training",
    body: "Coordinate Ray actor groups for diffusion and multimodal RL workloads.",
  },
  {
    title: "Hydra Recipes",
    body: "Compose reproducible experiments from typed configs and focused overrides.",
  },
  {
    title: "Pluggable Rollouts",
    body: "Swap rollout engines, rewards, and policy logic without changing the entrypoint.",
  },
];

const readingPaths = [
  ["Start", "Install dependencies, then launch a first single-node recipe."],
  ["Configure", "Choose an experiment recipe and inspect the resolved Hydra config."],
  ["Scale", "Adapt recipes for multinode runs and cluster-specific runtime paths."],
];

const agentEndpoints = [
  "/llms.txt",
  "/llms-full.txt",
  "/md/agents/index.md",
  "/md/configuration/hydra/index.md",
];

export default function HomePage() {
  return (
    <main className="relative isolate min-h-screen overflow-hidden bg-fd-background text-fd-foreground">
      <div className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[520px] bg-[radial-gradient(circle_at_top_left,rgba(99,102,241,0.24),transparent_34%),radial-gradient(circle_at_top_right,rgba(14,165,233,0.18),transparent_32%)]" />

      <nav className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 py-6">
        <Link className="text-sm font-semibold tracking-tight" href="/">
          UniRL
        </Link>
        <div className="flex items-center gap-2 text-sm text-fd-muted-foreground">
          <Link className="rounded-full px-3 py-1.5 transition hover:bg-fd-muted hover:text-fd-foreground" href="/en/docs">
            EN
          </Link>
          <Link className="rounded-full px-3 py-1.5 transition hover:bg-fd-muted hover:text-fd-foreground" href="/zh/docs">
            中文
          </Link>
        </div>
      </nav>

      <section className="mx-auto grid w-full max-w-7xl gap-10 px-6 pb-16 pt-10 lg:grid-cols-[1.08fr_0.92fr] lg:items-center lg:pb-24 lg:pt-16">
        <div className="max-w-3xl">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border bg-fd-background/80 px-3 py-1 text-xs font-medium text-fd-muted-foreground shadow-sm backdrop-blur">
            <span className="h-2 w-2 rounded-full bg-fd-primary" />
            Agent-first docs for unified multimodal RL
          </div>
          <h1 className="text-5xl font-semibold tracking-tight sm:text-6xl lg:text-7xl">
            Build, run, and inspect unified multimodal RL experiments.
          </h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-fd-muted-foreground">
            UniRL combines Ray actor groups, Hydra recipes, composable training stacks, and pluggable
            rollout engines for diffusion and autoregressive generative models.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              className="rounded-full bg-fd-primary px-5 py-3 text-sm font-medium text-fd-primary-foreground shadow-lg shadow-fd-primary/20 transition hover:opacity-90"
              href="/en/docs"
            >
              Read English Docs
            </Link>
            <Link
              className="rounded-full border bg-fd-background/70 px-5 py-3 text-sm font-medium transition hover:bg-fd-muted"
              href="/zh/docs"
            >
              打开中文文档
            </Link>
            <Link
              className="rounded-full border bg-fd-background/70 px-5 py-3 text-sm font-medium transition hover:bg-fd-muted"
              href="/en/docs/agents"
            >
              Agent Guide
            </Link>
          </div>
        </div>

        <div className="rounded-[2rem] border bg-fd-card/80 p-4 shadow-2xl shadow-black/10 backdrop-blur">
          <div className="rounded-[1.5rem] border bg-fd-background p-5">
            <div className="flex items-center justify-between border-b pb-4">
              <div>
                <p className="text-sm font-medium">Launch UniRL Experiments</p>
                <p className="text-xs text-fd-muted-foreground">Hydra configs, Ray scheduling, and rollout engines</p>
              </div>
              <span className="rounded-full bg-fd-muted px-3 py-1 text-xs text-fd-muted-foreground">Ready to run</span>
            </div>
            <div className="space-y-4 py-5">
              {[
                ["Task domain", "Unified multimodal generative RL"],
                ["Models", "Stable Diffusion 3, Qwen-Image, WAN, HunyuanImage3"],
                ["Algorithms", "GRPO, DanceGRPO, MixGRPO, Flow-DPPO, NFT"],
                ["Entrypoint", "python -m unirl.train_diffusion"],
              ].map(([label, value]) => (
                <div className="grid grid-cols-[7rem_1fr] gap-3 text-sm" key={label}>
                  <span className="text-fd-muted-foreground">{label}</span>
                  <span className="font-medium">{value}</span>
                </div>
              ))}
            </div>
            <div className="rounded-2xl bg-fd-muted p-4 font-mono text-xs leading-6 text-fd-muted-foreground">
              <p className="font-sans text-[0.7rem] font-medium uppercase tracking-wide text-fd-foreground">Quick start</p>
              <p className="mt-2">python -m unirl.train_diffusion \</p>
              <p>&nbsp;&nbsp;--config-name=diffusion_rl/sd3_trainside \</p>
              <p>&nbsp;&nbsp;num_devices=8</p>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto w-full max-w-7xl px-6 pb-10">
        <div className="grid gap-4 md:grid-cols-3">
          {capabilities.map((item) => (
            <article className="rounded-3xl border bg-fd-card p-6 shadow-sm" key={item.title}>
              <h2 className="text-lg font-semibold">{item.title}</h2>
              <p className="mt-3 leading-7 text-fd-muted-foreground">{item.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="mx-auto grid w-full max-w-7xl gap-6 px-6 pb-20 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-3xl border bg-fd-card p-6">
          <p className="text-sm font-medium text-fd-muted-foreground">Documentation Entrypoints</p>
          <div className="mt-5 grid gap-3">
            {docLinks.map((link) => (
              <Link
                className="group rounded-2xl border bg-fd-background p-4 transition hover:border-fd-primary/50 hover:shadow-sm"
                href={link.href}
                key={link.href}
              >
                <div className="flex items-center justify-between gap-4">
                  <span className="font-medium">{link.label}</span>
                  <span className="text-fd-muted-foreground transition group-hover:translate-x-0.5">-&gt;</span>
                </div>
                <p className="mt-2 text-sm text-fd-muted-foreground">{link.description}</p>
              </Link>
            ))}
          </div>
        </div>

        <div className="grid gap-6">
          <div className="rounded-3xl border bg-fd-card p-6">
            <p className="text-sm font-medium text-fd-muted-foreground">Recommended Path</p>
            <div className="mt-5 grid gap-4 md:grid-cols-3">
              {readingPaths.map(([title, body], index) => (
                <div className="rounded-2xl bg-fd-muted p-4" key={title}>
                  <span className="text-xs font-medium text-fd-muted-foreground">0{index + 1}</span>
                  <h3 className="mt-3 font-semibold">{title}</h3>
                  <p className="mt-2 text-sm leading-6 text-fd-muted-foreground">{body}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-3xl border bg-fd-card p-6">
            <p className="text-sm font-medium text-fd-muted-foreground">Agent-Readable Endpoints</p>
            <div className="mt-5 grid gap-2 sm:grid-cols-2">
              {agentEndpoints.map((endpoint) => (
                <Link
                  className="rounded-2xl border bg-fd-background px-4 py-3 font-mono text-sm transition hover:bg-fd-muted"
                  href={endpoint}
                  key={endpoint}
                >
                  {endpoint}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
