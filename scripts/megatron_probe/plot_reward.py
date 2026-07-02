"""Parse the smoke log's reward line and plot raw + moving-average reward vs rollout."""
import re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "/root/unirl/megatron_smoke.log"
out = sys.argv[2] if len(sys.argv) > 2 else "/root/unirl/megatron_reward.png"
r = [float(m) for m in re.findall(r"rollout \d+/\d+  reward=([0-9.]+)", open(log).read())]
if not r:
    print("no rewards found"); sys.exit(1)


def mavg(x, k=5):
    return [sum(x[max(0, i - k + 1): i + 1]) / len(x[max(0, i - k + 1): i + 1]) for i in range(len(x))]


x = list(range(1, len(r) + 1))
plt.figure(figsize=(9, 5))
plt.plot(x, r, color="#bbb", lw=1, label="reward (per rollout)")
plt.plot(x, mavg(r, 5), color="#c00", lw=2.2, label="5-rollout moving avg")
plt.xlabel("rollout"); plt.ylabel("mean reward"); plt.title("Megatron backend GRPO (Qwen3-0.6B, dapo-math) — reward vs rollout")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out, dpi=120)
n = len(r)
print(f"rollouts={n} first10={sum(r[:10])/10:.4f} last10={sum(r[-10:])/10:.4f} max={max(r):.4f} -> {out}")
