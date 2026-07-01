"""Prepare ALFWorld rollout-driver rows for UniRL (LIN-519).

ALFWorld tasks live in the environment — the *prompt* is the env's reset observation,
not a jsonl field. This emits N trivial driver rows, one per game,
``{"prompt": "", "metadata": {"game_index": i}}``, so ``MultimodalRLDataSource`` drives
N rollouts and :meth:`AlfworldEnv.reset` maps each ``game_index`` to a game. The index
ordering matches :func:`unirl.rollout.loop.alfworld_env.list_alfworld_games`, so the row
and the env agree on which game an index selects (and the ``n`` GRPO siblings of a
prompt share the same game).

  ALFWORLD_DATA=/path/to/alfworld/data python -m unirl.utils.prepare_alfworld --out-dir data/alfworld
"""

from __future__ import annotations

import argparse
import json
import os

from unirl.rollout.loop.alfworld_env import list_alfworld_games


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit ALFWorld game-index driver rows.")
    ap.add_argument("--out-dir", default="data/alfworld")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="cap number of games (0 = all)")
    args = ap.parse_args()

    games = list_alfworld_games(args.split)
    if not games:
        raise SystemExit(
            "No ALFWorld games found. Set $ALFWORLD_DATA and run `alfworld-download` first."
        )
    if args.limit:
        games = games[: args.limit]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(games)):
            f.write(json.dumps({"prompt": "", "metadata": {"game_index": i}}) + "\n")
    print(f"wrote {len(games)} rows -> {out_path}")


if __name__ == "__main__":
    main()
