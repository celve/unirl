# Benchmark core

> **Where it fits:** the generation and scoring drivers behind `benchmarks.run`.
> Full map: [`../README.md`](../README.md).

## Gotchas

### Resume matches on identity, never on configuration

`t2i_jobs` skips any `(prompt, sample)` whose PNG already exists, and `run_text` skips any
`(id, sample)` already present in the completions jsonl. Neither the image filename
(`p{idx:05d}_s{idx}.png`) nor a completion row records what produced it, so re-running a tag
after changing a checkpoint, sampler setting or decoding parameter would extend the directory
in place — counts reconcile, and nothing marks the boundary between the two populations.

`provenance.guard` closes that: it fingerprints the generation configuration into a
`generation_manifest.json` beside the outputs, refuses a changed one and names the differing
field. **What it covers is divergence between two runs.** Three distinct failures are worth
keeping apart:

| Failure | Caught by |
|---|---|
| A second run extends a directory a different configuration produced | this guard, before the first write |
| A directory's *first* run used the wrong configuration | nothing here — the manifest faithfully records the wrong one, and no check compares a tag's name against its manifest |
| Wrong contents discovered after the fact | an image/adapter hash audit downstream |

So a citer should not read the guard as covering all three. It prevents mixing; it does not
validate that a tag's name describes its contents.

Two properties of the manifest itself:

- It is written **before** generation, so it records "this configuration was started here",
  not "this configuration produced these outputs". A directory abandoned mid-generation carries
  a complete-looking manifest; manifest presence is not evidence of completeness.
- A directory predating the guard is stamped `preexisting_outputs_unattributed: true` rather
  than blocked. Blocking would break every live output directory, and claiming provenance for
  contents nothing observed would be the worse failure — it would look like evidence.

The text guard deliberately runs *after* the job list and before the first write, while the
t2i guard runs first. Placing the text guard ahead of the early return would fetch the served
model id over the network, turning a completed no-op resume into a hard failure whenever the
inference server is down. The t2i payload needs no network, so it goes first there.

### `_load_pipe` rejects an untrained adapter

`all(not p.abs().sum().item() for p in lora_b)` with an empty-list guard — **all** `lora_B`
zero, never **any**. A single zero `lora_B` is legitimate: SD3's last block sets
`context_pre_only`, so `attn.add_q_proj` feeds nothing and never takes a gradient.

This is the repo-wide convention, not a local choice — three sites agree, and two reached it
independently: here, `unirl/tools/`-adjacent export checking via
`experiments/evaluation/audit_verlomni_export.py`, and E2's hand verification of a real
export. Write a new zero-guard the same way rather than rederiving it; testing **any**
`lora_B` zero would fail every SD3 adapter on block 23 alone.
