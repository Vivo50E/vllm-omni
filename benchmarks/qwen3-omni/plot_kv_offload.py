"""Plot KV cache offloading benchmark comparison.

Generates bar charts comparing TTFP, E2E latency, and RTF between
KV offload ON and OFF configurations.

Usage:
    python plot_kv_offload.py \
        --off results/bench_kv_offload_off_*.json \
        --on results/bench_kv_offload_on_*.json \
        --output results/qwen3_omni_kv_offload_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> list[dict]:
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def plot_comparison(off_results: list[dict], on_results: list[dict], output_path: str):
    """Create comparison plots for KV offload ON vs OFF."""

    # Extract concurrency levels
    concurrency_levels = [r["concurrency"] for r in off_results]

    # Extract metrics
    off_ttfp = [r["mean_ttfp_ms"] for r in off_results]
    on_ttfp = [r["mean_ttfp_ms"] for r in on_results]

    off_e2e = [r["mean_e2e_ms"] for r in off_results]
    on_e2e = [r["mean_e2e_ms"] for r in on_results]

    off_rtf = [r["mean_rtf"] for r in off_results]
    on_rtf = [r["mean_rtf"] for r in on_results]

    off_throughput = [r["request_throughput"] for r in off_results]
    on_throughput = [r["request_throughput"] for r in on_results]

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Qwen3-Omni-30B: KV Cache Offloading Performance Comparison', fontsize=16, fontweight='bold')

    x = np.arange(len(concurrency_levels))
    width = 0.35

    # Plot 1: TTFP (Time to First Packet)
    ax1.bar(x - width/2, off_ttfp, width, label='KV Offload OFF', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, on_ttfp, width, label='KV Offload ON', color='#3498db', alpha=0.8)
    ax1.set_xlabel('Concurrency', fontweight='bold')
    ax1.set_ylabel('TTFP (ms)', fontweight='bold')
    ax1.set_title('Time to First Packet (Lower is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(concurrency_levels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add percentage change annotations
    for i, (off_val, on_val) in enumerate(zip(off_ttfp, on_ttfp)):
        if off_val > 0:
            pct_change = ((on_val - off_val) / off_val) * 100
            color = 'green' if pct_change < 0 else 'red'
            ax1.text(i, max(off_val, on_val) * 1.05, f'{pct_change:+.1f}%',
                    ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    # Plot 2: E2E Latency
    ax2.bar(x - width/2, off_e2e, width, label='KV Offload OFF', color='#e74c3c', alpha=0.8)
    ax2.bar(x + width/2, on_e2e, width, label='KV Offload ON', color='#3498db', alpha=0.8)
    ax2.set_xlabel('Concurrency', fontweight='bold')
    ax2.set_ylabel('E2E Latency (ms)', fontweight='bold')
    ax2.set_title('End-to-End Latency (Lower is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(concurrency_levels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage change annotations
    for i, (off_val, on_val) in enumerate(zip(off_e2e, on_e2e)):
        if off_val > 0:
            pct_change = ((on_val - off_val) / off_val) * 100
            color = 'green' if pct_change < 0 else 'red'
            ax2.text(i, max(off_val, on_val) * 1.05, f'{pct_change:+.1f}%',
                    ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    # Plot 3: RTF (Real-Time Factor)
    ax3.bar(x - width/2, off_rtf, width, label='KV Offload OFF', color='#e74c3c', alpha=0.8)
    ax3.bar(x + width/2, on_rtf, width, label='KV Offload ON', color='#3498db', alpha=0.8)
    ax3.set_xlabel('Concurrency', fontweight='bold')
    ax3.set_ylabel('RTF', fontweight='bold')
    ax3.set_title('Real-Time Factor (Lower is Better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(concurrency_levels)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Real-time threshold')

    # Add percentage change annotations
    for i, (off_val, on_val) in enumerate(zip(off_rtf, on_rtf)):
        if off_val > 0:
            pct_change = ((on_val - off_val) / off_val) * 100
            color = 'green' if pct_change < 0 else 'red'
            ax3.text(i, max(off_val, on_val) * 1.05, f'{pct_change:+.1f}%',
                    ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    # Plot 4: Request Throughput
    ax4.bar(x - width/2, off_throughput, width, label='KV Offload OFF', color='#e74c3c', alpha=0.8)
    ax4.bar(x + width/2, on_throughput, width, label='KV Offload ON', color='#3498db', alpha=0.8)
    ax4.set_xlabel('Concurrency', fontweight='bold')
    ax4.set_ylabel('Requests/sec', fontweight='bold')
    ax4.set_title('Request Throughput (Higher is Better)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(concurrency_levels)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Add percentage change annotations
    for i, (off_val, on_val) in enumerate(zip(off_throughput, on_throughput)):
        if off_val > 0:
            pct_change = ((on_val - off_val) / off_val) * 100
            color = 'green' if pct_change > 0 else 'red'
            ax4.text(i, max(off_val, on_val) * 1.05, f'{pct_change:+.1f}%',
                    ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Concurrency':<12} {'Metric':<20} {'OFF':<15} {'ON':<15} {'Change':<10}")
    print("-" * 80)

    for i, conc in enumerate(concurrency_levels):
        print(f"{conc:<12} {'TTFP (ms)':<20} {off_ttfp[i]:<15.1f} {on_ttfp[i]:<15.1f} {((on_ttfp[i]-off_ttfp[i])/off_ttfp[i]*100):+.1f}%")
        print(f"{'':<12} {'E2E Latency (ms)':<20} {off_e2e[i]:<15.1f} {on_e2e[i]:<15.1f} {((on_e2e[i]-off_e2e[i])/off_e2e[i]*100):+.1f}%")
        print(f"{'':<12} {'RTF':<20} {off_rtf[i]:<15.3f} {on_rtf[i]:<15.3f} {((on_rtf[i]-off_rtf[i])/off_rtf[i]*100):+.1f}%")
        print(f"{'':<12} {'Throughput (req/s)':<20} {off_throughput[i]:<15.2f} {on_throughput[i]:<15.2f} {((on_throughput[i]-off_throughput[i])/off_throughput[i]*100):+.1f}%")
        if i < len(concurrency_levels) - 1:
            print("-" * 80)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Plot KV cache offloading benchmark comparison")
    parser.add_argument("--off", required=True, help="Path to KV offload OFF results JSON")
    parser.add_argument("--on", required=True, help="Path to KV offload ON results JSON")
    parser.add_argument("--output", required=True, help="Output path for plot image")
    args = parser.parse_args()

    off_results = load_results(args.off)
    on_results = load_results(args.on)

    # Ensure results are sorted by concurrency
    off_results = sorted(off_results, key=lambda x: x["concurrency"])
    on_results = sorted(on_results, key=lambda x: x["concurrency"])

    plot_comparison(off_results, on_results, args.output)


if __name__ == "__main__":
    main()
