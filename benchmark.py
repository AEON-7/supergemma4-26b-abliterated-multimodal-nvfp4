"""Benchmark SuperGemma4-26B NVFP4 AWQ on DGX Spark.

Measures:
  1. Single-request latency — TTFT + decode throughput (greedy)
  2. Concurrent throughput — 4 parallel streams (matches --max-num-seqs 4)
  3. Long-context prefill — 16K input, short decode

Prints a markdown table ready to paste into the model card.
"""

import argparse
import asyncio
import random
import statistics
import string
import time
from contextlib import asynccontextmanager

import httpx

BASE_URL = "http://localhost:8000/v1"
MODEL = "supergemma4-26b"

# Deterministic prompts for reproducible numbers
PROMPT_SHORT = "Write a detailed explanation of how transformers work in machine learning."
PROMPT_REASONING = "A cyclist rides 12 km east, then 5 km north. How far are they from the starting point, and in which direction? Show your reasoning step by step."


# ── utilities ────────────────────────────────────────────────────────────────

def filler_tokens(approx_tokens: int, seed: int = 0) -> str:
    """Build a prompt that tokenizes to roughly `approx_tokens`. Uses Lorem-like
    filler so the model isn't computing on real text we care about.
    """
    random.seed(seed)
    # ~4 chars/token on average with English; pad generously
    n_chars = approx_tokens * 4
    words = [
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 9)))
        for _ in range(n_chars // 5)
    ]
    return " ".join(words)


async def stream_completion(client: httpx.AsyncClient, messages: list, max_tokens: int, temperature: float = 0.0):
    """Stream a chat completion. Returns (ttft_sec, decode_tokens, total_sec)."""
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    async with client.stream(
        "POST",
        f"{BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout=300.0,
    ) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            import json
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if obj.get("choices") and obj["choices"][0].get("delta", {}).get("content"):
                tokens += 1  # chunk-level, close enough for vLLM
            if obj.get("usage"):
                # Prefer server-reported count when available (exact)
                tokens = obj["usage"].get("completion_tokens", tokens)
    total = time.perf_counter() - t0
    return ttft, tokens, total


# ── benchmarks ───────────────────────────────────────────────────────────────

async def bench_single(max_new: int = 256, n_runs: int = 3):
    """Single-stream latency."""
    print(f"\n── [1] Single-request (greedy, max_new={max_new}, n={n_runs}) ──")
    results = []
    async with httpx.AsyncClient() as client:
        for i in range(n_runs):
            ttft, toks, total = await stream_completion(
                client,
                [{"role": "user", "content": PROMPT_SHORT}],
                max_new,
            )
            decode_tps = toks / (total - ttft) if total > ttft else 0
            results.append({"ttft": ttft, "decode_tps": decode_tps, "total": total, "toks": toks})
            print(f"  run {i+1}: TTFT={ttft*1000:6.0f}ms  decode={decode_tps:6.1f} tok/s  ({toks} tokens in {total:.2f}s)")
    median_ttft = statistics.median(r["ttft"] for r in results)
    median_tps = statistics.median(r["decode_tps"] for r in results)
    print(f"  MEDIAN: TTFT {median_ttft*1000:.0f}ms, decode {median_tps:.1f} tok/s")
    return {"ttft_ms": median_ttft * 1000, "decode_tps": median_tps}


async def bench_concurrent(n_parallel: int = 4, max_new: int = 256, n_runs: int = 2):
    """Concurrent throughput — aggregate tok/s across n_parallel streams."""
    print(f"\n── [2] Concurrent ({n_parallel} parallel, max_new={max_new}, n={n_runs}) ──")
    results = []
    async with httpx.AsyncClient() as client:
        for i in range(n_runs):
            t0 = time.perf_counter()
            coros = [
                stream_completion(
                    client,
                    [{"role": "user", "content": PROMPT_SHORT + f" (run {i} stream {k})"}],
                    max_new,
                )
                for k in range(n_parallel)
            ]
            outs = await asyncio.gather(*coros)
            total = time.perf_counter() - t0
            total_toks = sum(o[1] for o in outs)
            agg_tps = total_toks / total
            median_ttft = statistics.median(o[0] for o in outs)
            results.append({"agg_tps": agg_tps, "ttft": median_ttft, "total_toks": total_toks, "wall": total})
            print(f"  run {i+1}: {total_toks} tokens in {total:.2f}s → {agg_tps:.1f} tok/s aggregate  (median TTFT {median_ttft*1000:.0f}ms)")
    median_agg = statistics.median(r["agg_tps"] for r in results)
    median_ttft = statistics.median(r["ttft"] for r in results)
    print(f"  MEDIAN aggregate: {median_agg:.1f} tok/s  (= {median_agg/n_parallel:.1f} per stream)")
    return {"agg_tps": median_agg, "per_stream_tps": median_agg / n_parallel, "ttft_ms": median_ttft * 1000}


async def bench_long_context(prompt_tokens: int = 16000, max_new: int = 64):
    """Long-context prefill speed."""
    print(f"\n── [3] Long-context prefill ({prompt_tokens} input tokens, {max_new} output) ──")
    filler = filler_tokens(prompt_tokens - 50)
    messages = [
        {"role": "user", "content": filler + "\n\nSummarize the above in one sentence."}
    ]
    async with httpx.AsyncClient() as client:
        ttft, toks, total = await stream_completion(client, messages, max_new)
    # TTFT here is mostly prefill time; decode_tps below is the post-TTFT speed
    decode_tps = toks / (total - ttft) if total > ttft else 0
    print(f"  TTFT (~prefill): {ttft*1000:.0f}ms  ({prompt_tokens/ttft:.0f} input-tok/s)")
    print(f"  decode: {decode_tps:.1f} tok/s  ({toks} tokens)")
    return {
        "input_tokens": prompt_tokens,
        "ttft_ms": ttft * 1000,
        "prefill_tps": prompt_tokens / ttft,
        "decode_tps": decode_tps,
    }


# ── correctness sanity ───────────────────────────────────────────────────────

async def correctness_check():
    """Run a few sanity prompts — just print outputs, no pass/fail asserts."""
    print("\n── [0] Correctness sanity ──")
    prompts = [
        "What is the capital of France? Answer in one sentence.",
        "What is 17 * 23? Give just the number.",
        "Write a haiku about autumn leaves.",
    ]
    async with httpx.AsyncClient(timeout=60.0) as client:
        for p in prompts:
            t0 = time.perf_counter()
            r = await client.post(
                f"{BASE_URL}/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": 120,
                    "temperature": 0.0,
                },
            )
            dt = time.perf_counter() - t0
            body = r.json()
            text = body["choices"][0]["message"]["content"]
            reasoning = body["choices"][0]["message"].get("reasoning_content") or ""
            print(f"  [{dt:.1f}s] Q: {p}")
            if reasoning:
                print(f"    thinking: {reasoning[:150].strip()}…")
            print(f"    A: {text.strip()[:200]}")


# ── main ─────────────────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    # Warmup to avoid counting any first-request autotuning
    print("Warmup…")
    async with httpx.AsyncClient(timeout=60.0) as client:
        await client.post(
            f"{BASE_URL}/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "temperature": 0.0},
        )

    if not args.skip_correctness:
        await correctness_check()

    single = await bench_single(max_new=args.max_new)
    concurrent = await bench_concurrent(n_parallel=args.parallel, max_new=args.max_new)
    if not args.skip_long:
        long_ctx = await bench_long_context()
    else:
        long_ctx = None

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY — SuperGemma4-26B NVFP4 AWQ on DGX Spark GB10")
    print("=" * 60)
    print(f"{'Metric':<38} {'Value':>18}")
    print("-" * 60)
    print(f"{'Single-stream TTFT':<38} {single['ttft_ms']:>15.0f} ms")
    print(f"{'Single-stream decode':<38} {single['decode_tps']:>15.1f} tok/s")
    print(f"{f'Aggregate @ {args.parallel} parallel streams':<38} {concurrent['agg_tps']:>15.1f} tok/s")
    print(f"{f'Per-stream @ {args.parallel} parallel':<38} {concurrent['per_stream_tps']:>15.1f} tok/s")
    print(f"{f'TTFT @ {args.parallel} parallel':<38} {concurrent['ttft_ms']:>15.0f} ms")
    if long_ctx:
        print(f"{'Prefill @ 16K input':<38} {long_ctx['prefill_tps']:>13.0f} tok/s")
        print(f"{'TTFT @ 16K input':<38} {long_ctx['ttft_ms']:>15.0f} ms")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
