"""VideoAlign smoke test + benchmark.

对运行中的 VideoAlign RewardService 做功能验证和性能测试。
包含：健康检查、单视频/批量/video_path 三种模式、延迟测量。

用法：
    # 用 VideoAlign 仓库自带的示例视频（自动检测）
    python3 scripts/test_videoalign.py

    # 指定服务地址
    python3 scripts/test_videoalign.py --url http://localhost:8090

    # 指定视频文件
    python3 scripts/test_videoalign.py --video /path/to/video.mp4

    # 只做功能验证，跳过 benchmark
    python3 scripts/test_videoalign.py --no-bench

    # 完整 benchmark（含不同 batch size）
    python3 scripts/test_videoalign.py --bench-full
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

# ── 默认视频路径（VideoAlign 仓库示例） ──────────────────────────
_VIDEO_DIRS = [
    Path("/path/to/VideoAlign/datasets/train/videos"),
    Path(__file__).resolve().parent.parent / "tests" / "assets",
]

_PROMPTS = {
    "example_1_A.mp4": "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it.",
    "example_1_B.mp4": "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it.",
    "example_2_A.mp4": "The camera follows a young explorer through an abandoned urban building at night.",
    "example_2_B.mp4": "The camera follows a young explorer through an abandoned urban building at night.",
    "example_3_A.mp4": "A lively street scene with people walking and cars passing by.",
    "example_3_B.mp4": "A lively street scene with people walking and cars passing by.",
}

_DEFAULT_PROMPT = "A high quality video showing a natural scene."


def _find_videos() -> list[Path]:
    """Search known directories for .mp4 test files."""
    for d in _VIDEO_DIRS:
        if d.is_dir():
            videos = sorted(d.glob("*.mp4"))
            if videos:
                return videos
    return []


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _prompt_for(path: Path) -> str:
    return _PROMPTS.get(path.name, _DEFAULT_PROMPT)


def _post_score(url: str, payload: dict, timeout: float = 300) -> dict:
    resp = requests.post(f"{url}/score", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── 测试函数 ──────────────────────────────────────────────────────

def test_health(url: str) -> bool:
    print("── 健康检查 ──")
    try:
        resp = requests.get(f"{url}/health", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        print(f"  状态: {body['status']}")
        for name, replicas in body.get("rewards", {}).items():
            print(f"  {name}: {replicas}")
        print()
        return body["status"] == "ok"
    except Exception as e:
        print(f"  [FAIL] {e}\n")
        return False


def test_rewards_endpoint(url: str) -> list[str]:
    print("── 已注册 rewards ──")
    resp = requests.get(f"{url}/rewards", timeout=10)
    resp.raise_for_status()
    rewards = resp.json()["rewards"]
    print(f"  {rewards}")
    if "videoalign" not in rewards:
        print("  [WARN] videoalign 未注册！")
    print()
    return rewards


def test_single_video_b64(url: str, video: Path) -> bool:
    print(f"── 单视频测试 (video_b64) ──")
    print(f"  视频: {video.name} ({video.stat().st_size / 1024:.0f} KB)")

    payload = {
        "requests": [{
            "history": [{"text": _prompt_for(video), "video_b64": _b64(video)}],
            "required_rewards": ["videoalign"],
        }]
    }

    t0 = time.perf_counter()
    body = _post_score(url, payload)
    elapsed = time.perf_counter() - t0

    result = body["results"][0]
    errors = body["errors"][0]

    if errors:
        print(f"  [FAIL] 错误: {errors}")
        print(f"  延迟: {elapsed:.2f}s\n")
        return False

    scores = result.get("videoalign", {})
    print(f"  VQ={scores['VQ']:.4f}  MQ={scores['MQ']:.4f}  TA={scores['TA']:.4f}  Overall={scores['Overall']:.4f}")
    print(f"  延迟: {elapsed:.2f}s")
    print()
    return True


def test_single_video_path(url: str, video: Path) -> bool:
    print(f"── 单视频测试 (video_path) ──")
    print(f"  路径: {video}")

    payload = {
        "requests": [{
            "history": [{"text": _prompt_for(video), "video_path": str(video)}],
            "required_rewards": ["videoalign"],
        }]
    }

    t0 = time.perf_counter()
    body = _post_score(url, payload)
    elapsed = time.perf_counter() - t0

    result = body["results"][0]
    errors = body["errors"][0]

    if errors:
        print(f"  [FAIL] 错误: {errors}")
        print(f"  延迟: {elapsed:.2f}s\n")
        return False

    scores = result.get("videoalign", {})
    print(f"  VQ={scores['VQ']:.4f}  MQ={scores['MQ']:.4f}  TA={scores['TA']:.4f}  Overall={scores['Overall']:.4f}")
    print(f"  延迟: {elapsed:.2f}s")
    print()
    return True


def test_batch(url: str, videos: list[Path]) -> bool:
    n = min(len(videos), 6)
    batch = videos[:n]
    print(f"── 批量测试 ({n} 视频, video_b64) ──")

    payload = {
        "requests": [
            {
                "history": [{"text": _prompt_for(v), "video_b64": _b64(v)}],
                "required_rewards": ["videoalign"],
            }
            for v in batch
        ]
    }

    t0 = time.perf_counter()
    body = _post_score(url, payload)
    elapsed = time.perf_counter() - t0

    all_ok = True
    for i, (result, errors) in enumerate(zip(body["results"], body["errors"])):
        if errors:
            print(f"  [{i}] {batch[i].name}: ERROR {errors}")
            all_ok = False
        else:
            s = result.get("videoalign", {})
            print(f"  [{i}] {batch[i].name}: VQ={s['VQ']:.3f} MQ={s['MQ']:.3f} TA={s['TA']:.3f} Overall={s['Overall']:.3f}")

    per_video = elapsed / n
    print(f"  总延迟: {elapsed:.2f}s | 每视频: {per_video:.2f}s | 吞吐: {n / elapsed:.1f} videos/s")
    print()
    return all_ok


def test_consistency(url: str, video: Path, n_runs: int = 3) -> bool:
    """同一视频多次评分，检查结果一致性。"""
    print(f"── 一致性测试 ({n_runs} 次重复) ──")

    payload = {
        "requests": [{
            "history": [{"text": _prompt_for(video), "video_b64": _b64(video)}],
            "required_rewards": ["videoalign"],
        }]
    }

    results = []
    for i in range(n_runs):
        body = _post_score(url, payload)
        scores = body["results"][0].get("videoalign", {})
        results.append(scores)

    # 检查分数是否一致（容差 0.001）
    ok = True
    for key in ("VQ", "MQ", "TA", "Overall"):
        values = [r[key] for r in results]
        spread = max(values) - min(values)
        status = "OK" if spread < 0.01 else "DRIFT"
        if spread >= 0.01:
            ok = False
        print(f"  {key}: {values} spread={spread:.6f} [{status}]")

    print()
    return ok


# ── Benchmark ────────────────────────────────────────────────────

def benchmark(url: str, video: Path, batch_sizes: list[int], warmup: int = 2, repeats: int = 3) -> None:
    print("══════════════════════════════════════")
    print(" VideoAlign 性能 Benchmark")
    print("══════════════════════════════════════")

    video_b64 = _b64(video)
    prompt = _prompt_for(video)

    # Warmup
    print(f"\n  Warmup ({warmup} 次)...")
    for _ in range(warmup):
        payload = {
            "requests": [{
                "history": [{"text": prompt, "video_b64": video_b64}],
                "required_rewards": ["videoalign"],
            }]
        }
        _post_score(url, payload)

    # Per batch size
    print(f"\n  {'Batch':>5s}  {'Avg(s)':>8s}  {'Per-vid(s)':>10s}  {'Throughput':>10s}  {'Runs':>20s}")
    print(f"  {'─' * 5}  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 20}")

    for bs in batch_sizes:
        payload = {
            "requests": [
                {
                    "history": [{"text": prompt, "video_b64": video_b64}],
                    "required_rewards": ["videoalign"],
                }
                for _ in range(bs)
            ]
        }

        latencies = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _post_score(url, payload)
            latencies.append(time.perf_counter() - t0)

        avg = sum(latencies) / len(latencies)
        per_vid = avg / bs
        throughput = bs / avg
        runs_str = ", ".join(f"{l:.2f}" for l in latencies)
        print(f"  {bs:>5d}  {avg:>8.2f}  {per_vid:>10.2f}  {throughput:>8.1f}/s  [{runs_str}]")

    print()


# ── 入口 ─────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="VideoAlign RewardService 测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--url", default="http://localhost:8080", help="服务地址")
    ap.add_argument("--video", type=Path, nargs="+", default=None, help="测试视频路径")
    ap.add_argument("--no-bench", action="store_true", help="跳过性能测试")
    ap.add_argument("--bench-full", action="store_true", help="完整 benchmark (batch 1-6)")
    args = ap.parse_args()

    url = args.url.rstrip("/")

    # 查找视频
    if args.video:
        videos = list(args.video)
        missing = [v for v in videos if not v.is_file()]
        if missing:
            print(f"[ERROR] 视频文件不存在: {missing}", file=sys.stderr)
            return 1
    else:
        videos = _find_videos()
        if not videos:
            print("[ERROR] 未找到测试视频。请用 --video 指定。", file=sys.stderr)
            return 1
        print(f"[INFO] 自动找到 {len(videos)} 个测试视频: {videos[0].parent}/\n")

    # ── 功能测试 ──
    passed = 0
    failed = 0

    if test_health(url):
        passed += 1
    else:
        print("[FATAL] 服务不健康，终止测试。", file=sys.stderr)
        return 1

    test_rewards_endpoint(url)

    for test_fn, test_args in [
        (test_single_video_b64, (url, videos[0])),
        (test_single_video_path, (url, videos[0])),
        (test_batch, (url, videos)),
        (test_consistency, (url, videos[0])),
    ]:
        try:
            if test_fn(*test_args):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print(f"══ 功能测试结果: {passed} passed, {failed} failed ══\n")

    # ── Benchmark ──
    if not args.no_bench:
        if args.bench_full:
            batch_sizes = [1, 2, 3, 4, 6]
        else:
            batch_sizes = [1, 3, 6]
        benchmark(url, videos[0], batch_sizes)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
