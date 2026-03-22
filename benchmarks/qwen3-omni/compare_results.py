"""Generate PR vs Main comparison table.

Creates a formatted comparison table matching the style:
Metric    PR #1330 c=1    Main c=1    Delta    PR #1330 c=4    Main c=4    Delta
completed    10/10    10/10    =    40/40    40/40    =
RTF    0.283    0.299    -5.4%    0.780    0.890    -12.4%
TTFT (ms)    71.4    89.6    -20.3%    2047    1363    +50.2%
E2EL (ms)    7659    8027    -4.6%    23163    26394    -12.2%
throughput (req/s)    0.131    0.125    +4.8%    0.166    0.146    +13.7%

Usage:
    python compare_results.py \
        --pr results/bench_pr_1330_*.json \
        --main results/bench_main_*.json \
        --output results/pr_vs_main.txt
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> list[dict]:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def format_delta(pr_val: float, main_val: float, reverse: bool = False) -> str:
    """Format percentage delta between PR and Main.

    Args:
        pr_val: PR value
        main_val: Main value
        reverse: If True, reverse the sign (for metrics where lower is worse)

    Returns:
        Formatted delta string with sign and percentage
    """
    if main_val == 0:
        return "N/A"

    delta_pct = ((pr_val - main_val) / main_val) * 100
    if reverse:
        delta_pct = -delta_pct

    if abs(delta_pct) < 0.05:
        return "="

    return f"{delta_pct:+.1f}%"


def generate_comparison_table(pr_results: list[dict], main_results: list[dict]) -> str:
    """Generate formatted comparison table."""

    # Sort by concurrency
    pr_results = sorted(pr_results, key=lambda x: x["concurrency"])
    main_results = sorted(main_results, key=lambda x: x["concurrency"])

    # Build header
    concurrency_levels = [r["concurrency"] for r in pr_results]

    # Create header row
    header_parts = ["Metric"]
    for conc in concurrency_levels:
        header_parts.extend([f"PR #1330 c={conc}", f"Main c={conc}", "Delta"])

    # Calculate column widths
    col_widths = [max(12, len(h)) for h in header_parts]

    # Build table
    lines = []

    # Header
    header_line = "    ".join(h.ljust(w) for h, w in zip(header_parts, col_widths))
    lines.append(header_line)
    lines.append("=" * len(header_line))

    # Completed row
    completed_parts = ["completed"]
    for pr, main in zip(pr_results, main_results):
        pr_completed = f"{pr['completed']}/{pr['num_prompts']}"
        main_completed = f"{main['completed']}/{main['num_prompts']}"
        delta = "=" if pr_completed == main_completed else "!"
        completed_parts.extend([pr_completed, main_completed, delta])

    lines.append("    ".join(p.ljust(w) for p, w in zip(completed_parts, col_widths)))

    # RTF row (lower is better)
    rtf_parts = ["RTF"]
    for pr, main in zip(pr_results, main_results):
        pr_rtf = pr["mean_rtf"]
        main_rtf = main["mean_rtf"]
        delta = format_delta(pr_rtf, main_rtf, reverse=False)
        rtf_parts.extend([f"{pr_rtf:.3f}", f"{main_rtf:.3f}", delta])

    lines.append("    ".join(p.ljust(w) for p, w in zip(rtf_parts, col_widths)))

    # TTFT (TTFP) row (lower is better)
    ttft_parts = ["TTFT (ms)"]
    for pr, main in zip(pr_results, main_results):
        pr_ttft = pr["mean_ttfp_ms"]
        main_ttft = main["mean_ttfp_ms"]
        delta = format_delta(pr_ttft, main_ttft, reverse=False)
        ttft_parts.extend([f"{pr_ttft:.1f}", f"{main_ttft:.1f}", delta])

    lines.append("    ".join(p.ljust(w) for p, w in zip(ttft_parts, col_widths)))

    # E2EL row (lower is better)
    e2el_parts = ["E2EL (ms)"]
    for pr, main in zip(pr_results, main_results):
        pr_e2el = pr["mean_e2e_ms"]
        main_e2el = main["mean_e2e_ms"]
        delta = format_delta(pr_e2el, main_e2el, reverse=False)
        e2el_parts.extend([f"{pr_e2el:.0f}", f"{main_e2el:.0f}", delta])

    lines.append("    ".join(p.ljust(w) for p, w in zip(e2el_parts, col_widths)))

    # Throughput row (higher is better, so don't reverse)
    throughput_parts = ["throughput (req/s)"]
    for pr, main in zip(pr_results, main_results):
        pr_throughput = pr["request_throughput"]
        main_throughput = main["request_throughput"]
        delta = format_delta(pr_throughput, main_throughput, reverse=False)
        throughput_parts.extend([f"{pr_throughput:.3f}", f"{main_throughput:.3f}", delta])

    lines.append("    ".join(p.ljust(w) for p, w in zip(throughput_parts, col_widths)))

    lines.append("=" * len(header_line))

    return "\n".join(lines)


def generate_detailed_stats(pr_results: list[dict], main_results: list[dict]) -> str:
    """Generate detailed statistics section."""
    lines = ["\n", "DETAILED STATISTICS", "=" * 80, ""]

    for pr, main in zip(pr_results, main_results):
        conc = pr["concurrency"]
        lines.append(f"Concurrency: {conc}")
        lines.append("-" * 80)

        # TTFT percentiles
        lines.append(f"TTFT (ms):")
        lines.append(f"  PR #1330:  p50={pr['median_ttfp_ms']:.1f}  p90={pr['p90_ttfp_ms']:.1f}  p95={pr['p95_ttfp_ms']:.1f}  p99={pr['p99_ttfp_ms']:.1f}")
        lines.append(f"  Main:      p50={main['median_ttfp_ms']:.1f}  p90={main['p90_ttfp_ms']:.1f}  p95={main['p95_ttfp_ms']:.1f}  p99={main['p99_ttfp_ms']:.1f}")

        # E2E percentiles
        lines.append(f"E2EL (ms):")
        lines.append(f"  PR #1330:  p50={pr['median_e2e_ms']:.1f}  p90={pr['p90_e2e_ms']:.1f}  p95={pr['p95_e2e_ms']:.1f}  p99={pr['p99_e2e_ms']:.1f}")
        lines.append(f"  Main:      p50={main['median_e2e_ms']:.1f}  p90={main['p90_e2e_ms']:.1f}  p95={main['p95_e2e_ms']:.1f}  p99={main['p99_e2e_ms']:.1f}")

        # RTF percentiles
        lines.append(f"RTF:")
        lines.append(f"  PR #1330:  mean={pr['mean_rtf']:.3f}  median={pr['median_rtf']:.3f}")
        lines.append(f"  Main:      mean={main['mean_rtf']:.3f}  median={main['median_rtf']:.3f}")

        # Throughput
        lines.append(f"Throughput:")
        lines.append(f"  PR #1330:  {pr['request_throughput']:.3f} req/s  ({pr['audio_throughput']:.2f} audio-sec/wall-sec)")
        lines.append(f"  Main:      {main['request_throughput']:.3f} req/s  ({main['audio_throughput']:.2f} audio-sec/wall-sec)")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare PR vs Main benchmark results")
    parser.add_argument("--pr", required=True, help="Path to PR results JSON")
    parser.add_argument("--main", required=True, help="Path to Main results JSON")
    parser.add_argument("--output", help="Output path for comparison table (optional, prints to stdout if not provided)")
    args = parser.parse_args()

    try:
        pr_results = load_results(args.pr)
        main_results = load_results(args.main)

        if len(pr_results) != len(main_results):
            print(f"WARNING: Different number of concurrency levels (PR: {len(pr_results)}, Main: {len(main_results)})", file=sys.stderr)

        # Generate comparison table
        table = generate_comparison_table(pr_results, main_results)

        # Generate detailed stats
        detailed = generate_detailed_stats(pr_results, main_results)

        output = f"\n{table}\n{detailed}"

        # Output
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Comparison saved to {args.output}")

        # Always print to stdout
        print(output)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
