"""Generate three-mode comparison table (Baseline vs Offload vs LMCache).

Creates a formatted comparison table for all three modes across multiple concurrency levels.

Output format:
Metric            Baseline c=1  Offload c=1  LMCache c=1  Baseline c=4  Offload c=4  LMCache c=4  ...
completed         10/10         10/10        10/10        40/40         40/40        40/40
RTF               0.299         0.283        0.285        0.890         0.780        0.795
TTFT (ms)         89.6          71.4         73.2         1363          2047         1856
E2EL (ms)         8027          7659         7701         26394         23163        24102
throughput (r/s)  0.125         0.131        0.130        0.146         0.166        0.160

Usage:
    python compare_three_modes.py \
        --baseline results/bench_baseline_*.json \
        --offload results/bench_offload_*.json \
        --lmcache results/bench_lmcache_*.json \
        --output results/three_mode_comparison.txt
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> list[dict]:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def generate_comparison_table(baseline: list[dict], offload: list[dict], lmcache: list[dict]) -> str:
    """Generate formatted comparison table for three modes."""

    # Sort by concurrency
    baseline = sorted(baseline, key=lambda x: x["concurrency"])
    offload = sorted(offload, key=lambda x: x["concurrency"])
    lmcache = sorted(lmcache, key=lambda x: x["concurrency"])

    # Get concurrency levels
    concurrency_levels = [r["concurrency"] for r in baseline]

    # Build header
    header_parts = ["Metric"]
    for conc in concurrency_levels:
        header_parts.extend([f"Baseline c={conc}", f"Offload c={conc}", f"LMCache c={conc}"])

    # Calculate column widths (minimum 12 chars)
    col_widths = [max(15, len(h)) for h in header_parts]

    lines = []

    # Header
    header_line = "    ".join(h.ljust(w) for h, w in zip(header_parts, col_widths))
    lines.append(header_line)
    lines.append("=" * len(header_line))

    # Completed row
    completed_parts = ["completed"]
    for b, o, l in zip(baseline, offload, lmcache):
        completed_parts.extend([
            f"{b['completed']}/{b['num_prompts']}",
            f"{o['completed']}/{o['num_prompts']}",
            f"{l['completed']}/{l['num_prompts']}"
        ])
    lines.append("    ".join(p.ljust(w) for p, w in zip(completed_parts, col_widths)))

    # RTF row
    rtf_parts = ["RTF"]
    for b, o, l in zip(baseline, offload, lmcache):
        rtf_parts.extend([
            f"{b['mean_rtf']:.3f}",
            f"{o['mean_rtf']:.3f}",
            f"{l['mean_rtf']:.3f}"
        ])
    lines.append("    ".join(p.ljust(w) for p, w in zip(rtf_parts, col_widths)))

    # TTFT row
    ttft_parts = ["TTFT (ms)"]
    for b, o, l in zip(baseline, offload, lmcache):
        ttft_parts.extend([
            f"{b['mean_ttfp_ms']:.1f}",
            f"{o['mean_ttfp_ms']:.1f}",
            f"{l['mean_ttfp_ms']:.1f}"
        ])
    lines.append("    ".join(p.ljust(w) for p, w in zip(ttft_parts, col_widths)))

    # E2EL row
    e2el_parts = ["E2EL (ms)"]
    for b, o, l in zip(baseline, offload, lmcache):
        e2el_parts.extend([
            f"{b['mean_e2e_ms']:.0f}",
            f"{o['mean_e2e_ms']:.0f}",
            f"{l['mean_e2e_ms']:.0f}"
        ])
    lines.append("    ".join(p.ljust(w) for p, w in zip(e2el_parts, col_widths)))

    # Throughput row
    throughput_parts = ["throughput (r/s)"]
    for b, o, l in zip(baseline, offload, lmcache):
        throughput_parts.extend([
            f"{b['request_throughput']:.3f}",
            f"{o['request_throughput']:.3f}",
            f"{l['request_throughput']:.3f}"
        ])
    lines.append("    ".join(p.ljust(w) for p, w in zip(throughput_parts, col_widths)))

    lines.append("=" * len(header_line))

    return "\n".join(lines)


def generate_delta_analysis(baseline: list[dict], offload: list[dict], lmcache: list[dict]) -> str:
    """Generate delta analysis comparing offload and lmcache against baseline."""

    lines = ["\n", "DELTA ANALYSIS (vs Baseline)", "=" * 80, ""]

    baseline = sorted(baseline, key=lambda x: x["concurrency"])
    offload = sorted(offload, key=lambda x: x["concurrency"])
    lmcache = sorted(lmcache, key=lambda x: x["concurrency"])

    for b, o, l in zip(baseline, offload, lmcache):
        conc = b["concurrency"]
        lines.append(f"Concurrency: {conc}")
        lines.append("-" * 80)

        # RTF deltas
        rtf_delta_offload = ((o['mean_rtf'] - b['mean_rtf']) / b['mean_rtf'] * 100) if b['mean_rtf'] > 0 else 0
        rtf_delta_lmcache = ((l['mean_rtf'] - b['mean_rtf']) / b['mean_rtf'] * 100) if b['mean_rtf'] > 0 else 0
        lines.append(f"RTF:          Baseline={b['mean_rtf']:.3f}  Offload={o['mean_rtf']:.3f} ({rtf_delta_offload:+.1f}%)  LMCache={l['mean_rtf']:.3f} ({rtf_delta_lmcache:+.1f}%)")

        # TTFT deltas
        ttft_delta_offload = ((o['mean_ttfp_ms'] - b['mean_ttfp_ms']) / b['mean_ttfp_ms'] * 100) if b['mean_ttfp_ms'] > 0 else 0
        ttft_delta_lmcache = ((l['mean_ttfp_ms'] - b['mean_ttfp_ms']) / b['mean_ttfp_ms'] * 100) if b['mean_ttfp_ms'] > 0 else 0
        lines.append(f"TTFT (ms):    Baseline={b['mean_ttfp_ms']:.1f}  Offload={o['mean_ttfp_ms']:.1f} ({ttft_delta_offload:+.1f}%)  LMCache={l['mean_ttfp_ms']:.1f} ({ttft_delta_lmcache:+.1f}%)")

        # E2EL deltas
        e2el_delta_offload = ((o['mean_e2e_ms'] - b['mean_e2e_ms']) / b['mean_e2e_ms'] * 100) if b['mean_e2e_ms'] > 0 else 0
        e2el_delta_lmcache = ((l['mean_e2e_ms'] - b['mean_e2e_ms']) / b['mean_e2e_ms'] * 100) if b['mean_e2e_ms'] > 0 else 0
        lines.append(f"E2EL (ms):    Baseline={b['mean_e2e_ms']:.0f}  Offload={o['mean_e2e_ms']:.0f} ({e2el_delta_offload:+.1f}%)  LMCache={l['mean_e2e_ms']:.0f} ({e2el_delta_lmcache:+.1f}%)")

        # Throughput deltas
        tp_delta_offload = ((o['request_throughput'] - b['request_throughput']) / b['request_throughput'] * 100) if b['request_throughput'] > 0 else 0
        tp_delta_lmcache = ((l['request_throughput'] - b['request_throughput']) / b['request_throughput'] * 100) if b['request_throughput'] > 0 else 0
        lines.append(f"Throughput:   Baseline={b['request_throughput']:.3f}  Offload={o['request_throughput']:.3f} ({tp_delta_offload:+.1f}%)  LMCache={l['request_throughput']:.3f} ({tp_delta_lmcache:+.1f}%)")

        lines.append("")

    return "\n".join(lines)


def generate_detailed_stats(baseline: list[dict], offload: list[dict], lmcache: list[dict]) -> str:
    """Generate detailed statistics section."""
    lines = ["\n", "DETAILED STATISTICS", "=" * 80, ""]

    baseline = sorted(baseline, key=lambda x: x["concurrency"])
    offload = sorted(offload, key=lambda x: x["concurrency"])
    lmcache = sorted(lmcache, key=lambda x: x["concurrency"])

    for b, o, l in zip(baseline, offload, lmcache):
        conc = b["concurrency"]
        lines.append(f"Concurrency: {conc}")
        lines.append("-" * 80)

        # TTFT percentiles
        lines.append("TTFT Percentiles (ms):")
        lines.append(f"  Baseline: p50={b['median_ttfp_ms']:.1f}  p90={b['p90_ttfp_ms']:.1f}  p95={b['p95_ttfp_ms']:.1f}  p99={b['p99_ttfp_ms']:.1f}")
        lines.append(f"  Offload:  p50={o['median_ttfp_ms']:.1f}  p90={o['p90_ttfp_ms']:.1f}  p95={o['p95_ttfp_ms']:.1f}  p99={o['p99_ttfp_ms']:.1f}")
        lines.append(f"  LMCache:  p50={l['median_ttfp_ms']:.1f}  p90={l['p90_ttfp_ms']:.1f}  p95={l['p95_ttfp_ms']:.1f}  p99={l['p99_ttfp_ms']:.1f}")

        # E2E percentiles
        lines.append("E2EL Percentiles (ms):")
        lines.append(f"  Baseline: p50={b['median_e2e_ms']:.1f}  p90={b['p90_e2e_ms']:.1f}  p95={b['p95_e2e_ms']:.1f}  p99={b['p99_e2e_ms']:.1f}")
        lines.append(f"  Offload:  p50={o['median_e2e_ms']:.1f}  p90={o['p90_e2e_ms']:.1f}  p95={o['p95_e2e_ms']:.1f}  p99={o['p99_e2e_ms']:.1f}")
        lines.append(f"  LMCache:  p50={l['median_e2e_ms']:.1f}  p90={l['p90_e2e_ms']:.1f}  p95={l['p95_e2e_ms']:.1f}  p99={l['p99_e2e_ms']:.1f}")

        # Audio throughput
        lines.append("Audio Throughput (audio-sec/wall-sec):")
        lines.append(f"  Baseline: {b['audio_throughput']:.2f}")
        lines.append(f"  Offload:  {o['audio_throughput']:.2f}")
        lines.append(f"  LMCache:  {l['audio_throughput']:.2f}")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare three-mode benchmark results")
    parser.add_argument("--baseline", required=True, help="Path to baseline results JSON")
    parser.add_argument("--offload", required=True, help="Path to offload results JSON")
    parser.add_argument("--lmcache", required=True, help="Path to lmcache results JSON")
    parser.add_argument("--output", help="Output path for comparison table")
    args = parser.parse_args()

    try:
        baseline_results = load_results(args.baseline)
        offload_results = load_results(args.offload)
        lmcache_results = load_results(args.lmcache)

        # Generate comparison table
        table = generate_comparison_table(baseline_results, offload_results, lmcache_results)

        # Generate delta analysis
        delta = generate_delta_analysis(baseline_results, offload_results, lmcache_results)

        # Generate detailed stats
        detailed = generate_detailed_stats(baseline_results, offload_results, lmcache_results)

        output = f"\n{table}\n{delta}\n{detailed}"

        # Output
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Comparison saved to {args.output}")

        # Always print to stdout
        print(output)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
