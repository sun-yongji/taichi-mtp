"""
TaiChi-MTP example: demonstrate three prediction regimes on synthetic data.
"""

import numpy as np

from taichi_mtp import MTPEngine, create_mtp_engine, DepthMode


def main():
    print("=" * 60)
    print("  TaiChi-MTP: Multi-Token Prediction Demo")
    print("=" * 60)

    engine = create_mtp_engine(hidden_dim=256, output_dim=128, preset="balanced")

    print(f"\nEngine: {engine}")
    print(f"Heads: {len(engine.heads)} (one per yao position)")
    print(f"Coupling topology: C6 hexagonal (60° cyclic)")

    # Simulate three input regimes
    # Use structured noise: create a hidden state where specific
    # dimension groups have different variances
    rng = np.random.default_rng(42)
    test_cases = {
        "STABLE (low noise)": np.ones((32, 256)) * 0.5 + rng.normal(0, 0.01, (32, 256)),
        "NORMAL (moderate noise)": rng.normal(0, 0.5, (32, 256)),
        # Turbulent: first 128 dims have 10x variance
        "TURBULENT (high noise)": np.column_stack([
            rng.normal(0, 3.0, (32, 128)),
            rng.normal(0, 0.3, (32, 128)),
        ]),
    }

    for label, data in test_cases.items():
        result = engine(data)

        print(f"\n{'─' * 50}")
        print(f"  {label}")
        print(f"  Depth used: {result.depth_used}/6 | Mode: {result.mode.value}")
        print(f"  Head couplings:")
        for i, c in enumerate(result.head_couplings):
            bar = "█" * int(c * 50)
            yao_names = ["初爻", "二爻", "三爻", "四爻", "五爻", "上爻"]
            print(f"    Head {i} ({yao_names[i]:3s}) {c:.3f}  {bar}")
        print(f"  Head scores: {result.head_scores.round(3)}")

    # Depth stats
    stats = engine.depth_stats()
    print(f"\n{'─' * 50}")
    print(f"  Session stats:")
    print(f"    Calls: {stats['calls']}")
    print(f"    Mean depth: {stats['mean']:.2f} ± {stats['std']:.2f}")
    print(f"    Range: [{stats['min']}, {stats['max']}]")


if __name__ == "__main__":
    main()
