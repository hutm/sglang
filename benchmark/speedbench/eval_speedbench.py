"""Evaluate SpeedBench-style throughput through an SGLang endpoint."""

import argparse
import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base_url",
        required=True,
        help="Server base URL, e.g. http://localhost:30001/v1",
    )
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument(
        "--model",
        default="default",
        help="Model name to send in OpenAI-compatible requests.",
    )
    p.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent requests (1=serial)",
    )
    p.add_argument(
        "--stats_file", default=None, help="Path to server-side DLLM stats JSONL"
    )
    p.add_argument(
        "--single_turn_only", action="store_true", help="Skip multi-turn prompts"
    )
    p.add_argument(
        "--dataset_name",
        default=None,
        help="HuggingFace dataset name to load (multi-turn instruction-following "
        "benchmark). The dataset should expose 'turns', 'category', and "
        "optionally 'multiturn' fields. Required.",
    )
    p.add_argument(
        "--dataset_config",
        default=None,
        help="HuggingFace dataset configuration name (e.g., 'qualitative').",
    )
    p.add_argument(
        "--dataset_split",
        default="test",
        help="HuggingFace dataset split to load (default: test).",
    )
    p.add_argument(
        "--dataset_cache",
        default=None,
        help="HuggingFace cache directory override. Default: HF_HOME.",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Filter to specific dataset categories (e.g., math coding)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for quick comparison runs",
    )
    p.add_argument(
        "--api",
        choices=["completion", "chat"],
        default="completion",
        help="API type: completion (default, for DLLM) or chat (for Qwen3/Eagle3)",
    )
    p.add_argument(
        "--no_thinking",
        action="store_true",
        help="Disable thinking mode for chat API (Qwen3 models)",
    )
    p.add_argument(
        "--summary_path", default=None, help="Override path for summary JSON output"
    )
    return p.parse_args()


def load_speedbench(name, config=None, split="test", cache_dir=None):
    if not name:
        raise ValueError(
            "--dataset_name is required (a HuggingFace multi-turn dataset with "
            "'turns' / 'category' / 'multiturn' fields)."
        )
    kwargs = {}
    if config is not None:
        kwargs["name"] = config
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset(name, split=split, **kwargs)
    return list(ds)


def generate(
    base_url,
    prompt_or_turns,
    max_tokens,
    model="default",
    api="completion",
    no_thinking=False,
):
    if api == "chat":
        url = f"{base_url}/chat/completions"
        messages = []
        turns = (
            prompt_or_turns if isinstance(prompt_or_turns, list) else [prompt_or_turns]
        )
        for i, t in enumerate(turns):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": t})
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if no_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = f"{base_url}/completions"
        prompt = (
            "\n\n".join(prompt_or_turns)
            if isinstance(prompt_or_turns, list)
            else prompt_or_turns
        )
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    return completion_tokens


def read_stats_tail(stats_file, n_before):
    if not stats_file or not os.path.exists(stats_file):
        return []
    with open(stats_file) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines[n_before:]]


def main():
    args = parse_args()
    samples = load_speedbench(
        name=args.dataset_name,
        config=args.dataset_config,
        split=args.dataset_split,
        cache_dir=args.dataset_cache,
    )

    if args.single_turn_only:
        samples = [s for s in samples if not s.get("multiturn", False)]
        print(f"Single-turn only: {len(samples)} samples")
    else:
        print(
            f"All samples: {len(samples)} (incl. {sum(1 for s in samples if s.get('multiturn'))} multi-turn)"
        )

    if args.categories:
        cats = set(args.categories)
        available = {s.get("category") for s in samples}
        unknown = cats - available
        if unknown:
            raise SystemExit(
                f"--categories includes unknown values {sorted(unknown)}; "
                f"dataset has {sorted(c for c in available if c is not None)}"
            )
        samples = [s for s in samples if s.get("category") in cats]
        print(f"Filtered to categories {sorted(cats)}: {len(samples)} samples")
        if not samples:
            raise SystemExit(
                f"--categories filter produced 0 samples (requested {sorted(cats)})"
            )
    if args.limit is not None:
        samples = samples[: args.limit]
        print(f"Limited to {len(samples)} samples")

    stats_before = 0
    if args.stats_file and os.path.exists(args.stats_file):
        with open(args.stats_file) as f:
            stats_before = sum(1 for _ in f)

    cat_tokens = defaultdict(int)
    cat_time = defaultdict(float)
    cat_count = defaultdict(int)
    total_tokens = 0
    t_start = time.time()

    if args.concurrent <= 1:
        for sample in tqdm(samples, desc="Generating"):
            cat = sample["category"]
            turns = sample["turns"]
            t0 = time.time()
            toks = generate(
                args.base_url,
                turns,
                args.max_tokens,
                model=args.model,
                api=args.api,
                no_thinking=args.no_thinking,
            )
            elapsed = time.time() - t0
            cat_tokens[cat] += toks
            cat_time[cat] += elapsed
            cat_count[cat] += 1
            total_tokens += toks
    else:
        lock = threading.Lock()

        def _run(sample):
            toks = generate(
                args.base_url,
                sample["turns"],
                args.max_tokens,
                model=args.model,
                api=args.api,
                no_thinking=args.no_thinking,
            )
            return sample["category"], toks

        with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
            futures = {executor.submit(_run, s): s for s in samples}
            for future in tqdm(
                as_completed(futures),
                total=len(samples),
                desc=f"Generating (c={args.concurrent})",
            ):
                cat, toks = future.result()
                with lock:
                    cat_tokens[cat] += toks
                    cat_count[cat] += 1
                    total_tokens += toks

    total_time = time.time() - t_start
    if not samples:
        raise SystemExit("No samples to evaluate after filtering/limit.")
    overall_toks = sum(cat_tokens.values())

    new_stats = read_stats_tail(args.stats_file, stats_before)
    stats_with_counts = [
        d for d in new_stats if "forward_passes" in d and "tokens" in d
    ]
    has_stats = len(stats_with_counts) > 0

    print("\n" + "=" * 70)
    print(
        f"SPEED-Bench Results  |  {len(samples)} samples  |  max_tokens={args.max_tokens}"
    )
    print(
        f"Total time: {total_time:.1f}s  |  Overall throughput: {overall_toks/total_time:.1f} tok/s"
    )
    if has_stats:
        total_fp = sum(d["forward_passes"] for d in stats_with_counts)
        total_tok_stat = sum(d["tokens"] for d in stats_with_counts)
        n_blk = len(stats_with_counts)
        acc_rates = [
            d["acceptance_rate"] for d in stats_with_counts if "acceptance_rate" in d
        ]
        acc_rate = sum(acc_rates) / len(acc_rates) if acc_rates else None
        tok_per_fp = total_tok_stat / total_fp if total_fp else 0.0
        stats_line = (
            f"tok/FP: {tok_per_fp:.3f}  |  "
            f"FPs/blk: {total_fp/n_blk:.3f}  |  "
            f"tok/blk: {total_tok_stat/n_blk:.3f}"
        )
        if acc_rate is not None:
            stats_line += f"  |  Acceptance: {acc_rate*100:.2f}%"
        print(stats_line)
    print()

    cats = sorted(cat_tokens.keys())
    print(f"{'Category':<16} {'n':>4} {'tok/s':>8} {'avg_len':>8}", end="")
    if has_stats:
        print("  (overall stats above)", end="")
    print()
    print("-" * 42)
    for cat in cats:
        n = cat_count[cat]
        tps = cat_tokens[cat] / cat_time[cat] if cat_time[cat] > 0 else float("nan")
        avg_len = cat_tokens[cat] / n if n > 0 else 0
        tps_str = f"{tps:>8.1f}" if cat_time[cat] > 0 else "       N/A"
        print(f"{cat:<16} {n:>4} {tps_str} {avg_len:>8.1f}")
    print("-" * 42)
    print(
        f"{'TOTAL':<16} {len(samples):>4} {overall_toks/total_time:>8.1f} {overall_toks/len(samples):>8.1f}"
    )

    out = {
        "total_samples": len(samples),
        "total_tokens": overall_toks,
        "total_time_s": round(total_time, 2),
        "throughput_toks": round(overall_toks / total_time, 2),
        "max_tokens": args.max_tokens,
        "concurrent": args.concurrent,
        "per_category": {
            cat: {
                "n": cat_count[cat],
                "tokens": cat_tokens[cat],
                "time_s": round(cat_time[cat], 2) if cat_time[cat] > 0 else None,
                "toks_per_s": (
                    round(cat_tokens[cat] / cat_time[cat], 2)
                    if cat_time[cat] > 0
                    else None
                ),
                "avg_gen_len": (
                    round(cat_tokens[cat] / cat_count[cat], 1)
                    if cat_count[cat] > 0
                    else 0
                ),
            }
            for cat in cats
        },
    }
    if has_stats:
        out["tok_per_fp"] = round(tok_per_fp, 3)
        out["fps_per_blk"] = round(total_fp / n_blk, 3)
        out["tok_per_blk"] = round(total_tok_stat / n_blk, 3)
        if acc_rate is not None:
            out["acceptance_rate"] = round(acc_rate, 4)

    if args.summary_path:
        summary_path = args.summary_path
    elif args.stats_file:
        summary_path = args.stats_file.replace(".jsonl", "_summary.json")
    else:
        import tempfile

        summary_path = os.path.join(tempfile.gettempdir(), "speedbench_summary.json")
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
