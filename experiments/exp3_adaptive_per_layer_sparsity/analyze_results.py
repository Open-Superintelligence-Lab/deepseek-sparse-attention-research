"""
Analysis Script for Experiment 3 Results

Provides detailed analysis and comparison of different sparsity schedules.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt


def load_results() -> Dict:
    """Load all experiment results"""
    results = {}
    results_dir = Path('results')

    if not results_dir.exists():
        print("No results directory found. Run the experiment first!")
        return results

    # Find all sequence length directories
    for seq_dir in sorted(results_dir.glob('seq_*')):
        seq_len = int(seq_dir.name.split('_')[1])
        summary_path = seq_dir / 'comparison_summary.json'

        if summary_path.exists():
            with open(summary_path, 'r') as f:
                results[seq_len] = json.load(f)

    return results


def print_comparison_table(results: Dict):
    """Print formatted comparison table"""
    if not results:
        print("No results to display")
        return

    schedules = list(next(iter(results.values())).keys())
    seq_lens = sorted(results.keys())

    print("\n" + "="*120)
    print("EXPERIMENT 3: ADAPTIVE PER-LAYER SPARSITY - RESULTS SUMMARY")
    print("="*120)

    # Validation Loss Table
    print("\n📊 VALIDATION LOSS (Lower is Better)")
    print("-"*120)
    header = f"{'Schedule':<30} " + " ".join([f"Seq {s:<8}" for s in seq_lens])
    print(header)
    print("-"*120)

    for schedule in schedules:
        row = f"{schedule:<30} "
        for seq_len in seq_lens:
            if schedule in results[seq_len]:
                loss = results[seq_len][schedule]['val_loss']
                row += f"{loss:<12.4f} "
            else:
                row += f"{'N/A':<12} "
        print(row)

    # Validation Accuracy Table
    print("\n📊 VALIDATION ACCURACY (Higher is Better)")
    print("-"*120)
    print(header)
    print("-"*120)

    for schedule in schedules:
        row = f"{schedule:<30} "
        for seq_len in seq_lens:
            if schedule in results[seq_len]:
                acc = results[seq_len][schedule]['val_accuracy'] * 100
                row += f"{acc:<12.2f}% "
            else:
                row += f"{'N/A':<12} "
        print(row)

    # Training Speed Table
    print("\n⏱️  TRAINING SPEED (Time per Step in ms)")
    print("-"*120)
    print(header)
    print("-"*120)

    for schedule in schedules:
        row = f"{schedule:<30} "
        for seq_len in seq_lens:
            if schedule in results[seq_len]:
                time_ms = results[seq_len][schedule]['time_per_step'] * 1000
                row += f"{time_ms:<12.2f} "
            else:
                row += f"{'N/A':<12} "
        print(row)

    print("="*120)


def analyze_improvements(results: Dict):
    """Analyze improvements over uniform sparse baseline"""
    if not results:
        return

    print("\n" + "="*120)
    print("IMPROVEMENT ANALYSIS: Comparing Against Uniform Sparse (Exp2 Baseline)")
    print("="*120)

    for seq_len in sorted(results.keys()):
        if 'uniform_sparse' not in results[seq_len]:
            continue

        baseline_loss = results[seq_len]['uniform_sparse']['val_loss']
        baseline_acc = results[seq_len]['uniform_sparse']['val_accuracy']

        print(f"\n📏 Sequence Length: {seq_len}")
        print(f"   Baseline (Uniform Sparse): Loss={baseline_loss:.4f}, Acc={baseline_acc*100:.2f}%")
        print("-"*120)
        print(f"{'Schedule':<30} {'Loss Improvement':<20} {'Acc Improvement':<20} {'Status':<30}")
        print("-"*120)

        improvements = []
        for schedule, data in results[seq_len].items():
            if schedule == 'uniform_sparse':
                continue

            loss = data['val_loss']
            acc = data['val_accuracy']

            loss_improvement = ((baseline_loss - loss) / baseline_loss) * 100
            acc_improvement = ((acc - baseline_acc) / baseline_acc) * 100

            # Determine status
            if loss_improvement > 5 and acc_improvement > 5:
                status = "✅ Strong Improvement"
            elif loss_improvement > 0 and acc_improvement > 0:
                status = "✓ Improvement"
            elif loss_improvement < -5 or acc_improvement < -5:
                status = "❌ Worse"
            else:
                status = "≈ Similar"

            improvements.append((schedule, loss_improvement, acc_improvement, status))

        # Sort by loss improvement
        improvements.sort(key=lambda x: x[1], reverse=True)

        for schedule, loss_imp, acc_imp, status in improvements:
            print(f"{schedule:<30} {loss_imp:>+18.2f}% {acc_imp:>+18.2f}% {status:<30}")

    print("="*120)


def highlight_1024_results(results: Dict):
    """Special focus on 1024 token results (Exp2 failure case)"""
    if 1024 not in results:
        print("\n⚠️  No 1024 token results found")
        return

    print("\n" + "="*120)
    print("🎯 FOCUS: 1024 TOKEN RESULTS (Exp2 Failure Case)")
    print("="*120)
    print("\nExp2 Result: Uniform sparse was -41% WORSE than baseline MHLA at 1024 tokens")
    print("Goal: Recover this performance loss through adaptive per-layer sparsity\n")
    print("-"*120)

    data_1024 = results[1024]

    # Sort by validation loss
    sorted_schedules = sorted(data_1024.items(), key=lambda x: x[1]['val_loss'])

    print(f"{'Rank':<6} {'Schedule':<30} {'Val Loss':<12} {'Val Acc':<12} {'Status':<30}")
    print("-"*120)

    for rank, (schedule, data) in enumerate(sorted_schedules, 1):
        loss = data['val_loss']
        acc = data['val_accuracy'] * 100

        # Determine status
        if schedule == 'dense_baseline':
            status = "🥇 Upper Bound (Dense)"
        elif schedule == 'uniform_sparse':
            status = "⚠️  Exp2 Baseline (Failed)"
        elif rank == 1:
            status = "🏆 BEST ADAPTIVE"
        elif rank == 2:
            status = "🥈 2nd Best"
        elif rank == 3:
            status = "🥉 3rd Best"
        else:
            status = ""

        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        print(f"{emoji:<6} {schedule:<30} {loss:<12.4f} {acc:<12.2f}% {status:<30}")

    print("-"*120)

    # Calculate if we solved the Exp2 problem
    if 'uniform_sparse' in data_1024 and sorted_schedules[0][0] != 'uniform_sparse' and sorted_schedules[0][0] != 'dense_baseline':
        best_adaptive = sorted_schedules[0][0]
        best_loss = sorted_schedules[0][1]['val_loss']
        uniform_loss = data_1024['uniform_sparse']['val_loss']
        improvement = ((uniform_loss - best_loss) / uniform_loss) * 100

        print(f"\n✨ RESULT: {best_adaptive} improved {improvement:.2f}% over Uniform Sparse at 1024 tokens!")

        if 'dense_baseline' in data_1024:
            dense_loss = data_1024['dense_baseline']['val_loss']
            gap = ((best_loss - dense_loss) / dense_loss) * 100
            print(f"   Gap to Dense Baseline: {gap:.2f}%")
            if abs(gap) < 10:
                print("   📊 Near-optimal performance while using adaptive sparsity!")

    print("="*120)


def plot_layer_wise_analysis():
    """Plot layer-wise sparsity patterns"""
    from adaptive_sparse_attention import create_sparsity_schedule, SparsitySchedule

    n_layers = 6
    seq_len = 1024

    schedules = [
        SparsitySchedule.UNIFORM_SPARSE,
        SparsitySchedule.AGGRESSIVE_MIDDLE,
        SparsitySchedule.PROGRESSIVE_SPARSE,
        SparsitySchedule.DENSE_TO_SPARSE,
    ]

    fig, axes = plt.subplots(1, len(schedules), figsize=(20, 5))
    fig.suptitle('Per-Layer Sparsity Patterns (Sequence Length = 1024)', fontsize=14, fontweight='bold')

    for idx, schedule in enumerate(schedules):
        ax = axes[idx]
        config = create_sparsity_schedule(schedule, n_layers, seq_len)

        layers = list(range(n_layers))
        k_ratios = [config.layer_k_ratios[i] * 100 for i in range(n_layers)]

        # Color code by layer type
        colors = []
        for i in range(n_layers):
            if i < n_layers // 3:
                colors.append('#2E7D32')  # Green for early
            elif i < 2 * n_layers // 3:
                colors.append('#1976D2')  # Blue for middle
            else:
                colors.append('#C62828')  # Red for late

        bars = ax.bar(layers, k_ratios, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('k as % of Sequence Length')
        ax.set_title(schedule.value.replace('_', ' ').title())
        ax.set_ylim([0, 105])
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}%',
                   ha='center', va='bottom', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E7D32', alpha=0.7, edgecolor='black', label='Early Layers (Local)'),
        Patch(facecolor='#1976D2', alpha=0.7, edgecolor='black', label='Middle Layers (Compositional)'),
        Patch(facecolor='#C62828', alpha=0.7, edgecolor='black', label='Late Layers (Global)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('results/layer_wise_sparsity_patterns.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: results/layer_wise_sparsity_patterns.png")


def main():
    """Main analysis function"""
    print("\n" + "#"*120)
    print("EXPERIMENT 3: ADAPTIVE PER-LAYER SPARSITY - DETAILED ANALYSIS")
    print("#"*120)

    # Load results
    print("\n📂 Loading results...")
    results = load_results()

    if not results:
        print("❌ No results found. Please run the experiment first:")
        print("   python run_experiment.py")
        return

    print(f"✓ Loaded results for {len(results)} sequence lengths")

    # Print comparison tables
    print_comparison_table(results)

    # Analyze improvements
    analyze_improvements(results)

    # Highlight 1024 results
    highlight_1024_results(results)

    # Create visualizations
    print("\n📊 Creating layer-wise sparsity visualization...")
    try:
        plot_layer_wise_analysis()
    except Exception as e:
        print(f"⚠️  Could not create layer-wise plot: {e}")

    print("\n" + "="*120)
    print("ANALYSIS COMPLETE!")
    print("="*120)
    print("\nKey Findings:")
    print("  1. Check the tables above for performance comparisons")
    print("  2. Focus on 1024 token results (Exp2 failure case)")
    print("  3. Look for schedules that beat Uniform Sparse")
    print("  4. Verify training speed is similar across schedules")
    print("\nVisualization:")
    print("  - results/comprehensive_comparison.png (Full comparison)")
    print("  - results/layer_wise_sparsity_patterns.png (Sparsity schedules)")
    print("\n" + "="*120)


if __name__ == "__main__":
    main()
