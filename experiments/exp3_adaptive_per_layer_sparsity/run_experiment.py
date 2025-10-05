"""
Experiment 3: Adaptive Per-Layer Sparsity - Sequence Length Comparison

Tests different per-layer sparsity schedules based on layer specialization research:
- Dense Baseline: All layers dense (no sparsity)
- Uniform Sparse: All layers k=L/2 (Exp2 baseline)
- Dense-to-Sparse: Dense early → sparse late
- Aggressive-Middle: Sparse middle layers (most redundant)
- Progressive-Sparse: Dense early → aggressive middle → moderate late
- Reverse-Progressive: Sparse early → dense late

Research Basis:
- Early layers: Local patterns, short-range dependencies
- Middle layers: Feature composition, functionally redundant
- Late layers: Global context consolidation, semantic abstraction
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

from data.dataset import TextTokenDataset
from data.loader import load_and_cache_data
from configs.moe_config import MoEModelConfig
from exp3_models import (
    create_adaptive_model,
    create_dense_model,
    count_parameters,
    SparsitySchedule
)
from adaptive_sparse_attention import print_schedule_info, create_sparsity_schedule


# Test sequence lengths - focus on 1024 where Exp2 failed
SEQUENCE_LENGTHS = [64, 256, 1024]

# Sparsity schedules to test
SCHEDULES_TO_TEST = [
    SparsitySchedule.DENSE_BASELINE,
    SparsitySchedule.UNIFORM_SPARSE,
    SparsitySchedule.AGGRESSIVE_MIDDLE,
    SparsitySchedule.PROGRESSIVE_SPARSE,
    SparsitySchedule.DENSE_TO_SPARSE,
]

# Base config
BASE_CONFIG = {
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,  # 6 layers for clear early/middle/late division
    'd_ff': 512,
    'num_experts': 4,
    'expert_top_k': 2,
    'indexer_heads': 4,
    'indexer_dim': 64,
    'batch_size': 8,
    'steps': 1000,  # Moderate training for comparison
    'learning_rate': 3e-3,
    'eval_every': 200,
    'max_tokens': 50000,
    'num_documents': 1000,
    'dropout': 0.1,
    'load_balancing_weight': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


def load_data(seq_len):
    """Load data for a given sequence length"""
    data_config = MoEModelConfig(
        max_seq_len=seq_len,
        max_tokens=BASE_CONFIG['max_tokens'],
        num_documents=BASE_CONFIG['num_documents']
    )
    texts, tokenizer, tokens = load_and_cache_data(data_config)

    full_dataset = TextTokenDataset(tokens, seq_len)
    val_size = len(full_dataset) // 10
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    return train_dataset, val_dataset, data_config.vocab_size


def get_dynamic_batch_size(seq_len):
    """Adjust batch size based on sequence length"""
    if seq_len <= 256:
        return 8
    elif seq_len <= 512:
        return 4
    else:  # 1024+
        return 2


def evaluate(model, val_loader, vocab_size, device, is_adaptive=False):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, targets = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            if is_adaptive:
                logits, _, _ = model(input_ids, return_stats=False)
            else:
                logits, _ = model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, config, vocab_size, is_adaptive=False, schedule_name=""):
    """Train a model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    results = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'steps': [],
        'time_per_step': []
    }

    model.train()
    step = 0

    print(f"  Training {schedule_name}...")

    while step < config['steps']:
        for batch in train_loader:
            if step >= config['steps']:
                break

            step_start = time.time()

            input_ids, targets = batch
            input_ids = input_ids.to(config['device'])
            targets = targets.to(config['device'])

            if is_adaptive:
                logits, aux_loss, _ = model(input_ids, return_stats=False)
            else:
                logits, aux_loss = model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )
            if aux_loss is not None:
                loss = loss + aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_time = time.time() - step_start
            results['time_per_step'].append(step_time)

            # Evaluate
            if (step + 1) % config['eval_every'] == 0:
                model.eval()
                val_loss, val_acc = evaluate(model, val_loader, vocab_size,
                                            config['device'], is_adaptive)
                model.train()

                results['train_loss'].append(loss.item())
                results['val_loss'].append(val_loss)
                results['val_accuracy'].append(val_acc)
                results['steps'].append(step + 1)

                print(f"    Step {step+1}/{config['steps']}: "
                      f"Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, "
                      f"Val Acc={val_acc:.4f}")

            step += 1

    # Final evaluation
    model.eval()
    final_val_loss, final_val_acc = evaluate(model, val_loader, vocab_size,
                                             config['device'], is_adaptive)

    results['final_val_loss'] = final_val_loss
    results['final_val_accuracy'] = final_val_acc
    results['avg_time_per_step'] = sum(results['time_per_step']) / len(results['time_per_step'])

    return results


def run_for_sequence_length(seq_len):
    """Run all schedules for a given sequence length"""
    print(f"\n{'='*80}")
    print(f"SEQUENCE LENGTH: {seq_len}")
    print(f"{'='*80}\n")

    # Load data
    print(f"📚 Loading data for seq_len={seq_len}...")
    train_dataset, val_dataset, vocab_size = load_data(seq_len)
    batch_size = get_dynamic_batch_size(seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create config for this sequence length
    config = BASE_CONFIG.copy()
    config['max_seq_len'] = seq_len
    config['vocab_size'] = vocab_size
    config['batch_size'] = batch_size

    # Create results directory (use absolute path to avoid issues)
    results_dir = Path(__file__).parent / 'results' / f'seq_{seq_len}'
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Results directory: {results_dir.absolute()}")

    schedule_results = {}

    # Test each schedule
    for schedule in SCHEDULES_TO_TEST:
        schedule_name = schedule.value
        print(f"\n{'─'*80}")
        print(f"Testing Schedule: {schedule_name}")
        print(f"{'─'*80}")

        # Print schedule details
        sched_config = create_sparsity_schedule(schedule, config['n_layers'], seq_len)
        print_schedule_info(sched_config, config['n_layers'])

        # Create model
        torch.manual_seed(42)  # Fair comparison
        torch.cuda.manual_seed(42)

        if schedule == SparsitySchedule.DENSE_BASELINE:
            model = create_dense_model(config).to(config['device'])
            is_adaptive = False
        else:
            model = create_adaptive_model(config, schedule).to(config['device'])
            is_adaptive = True

        print(f"  Parameters: {count_parameters(model):,}")

        # Train
        results = train_model(
            model, train_loader, val_loader, config, vocab_size,
            is_adaptive=is_adaptive, schedule_name=schedule_name
        )

        # Save results
        schedule_results[schedule_name] = {
            'val_loss': results['final_val_loss'],
            'val_accuracy': results['final_val_accuracy'],
            'time_per_step': results['avg_time_per_step'],
            'parameters': count_parameters(model),
            'training_curves': {
                'steps': results['steps'],
                'train_loss': results['train_loss'],
                'val_loss': results['val_loss'],
                'val_accuracy': results['val_accuracy']
            }
        }

        # Save individual schedule results
        with open(results_dir / f'{schedule_name}_results.json', 'w') as f:
            json.dump(schedule_results[schedule_name], f, indent=2)

        print(f"  ✓ Final Val Loss: {results['final_val_loss']:.4f}")
        print(f"  ✓ Final Val Accuracy: {results['final_val_accuracy']:.4f}")

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Save comparison summary
    with open(results_dir / 'comparison_summary.json', 'w') as f:
        json.dump(schedule_results, f, indent=2)

    return schedule_results


def create_visualizations():
    """Create comprehensive visualization comparing all schedules"""
    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}\n")

    # Collect all results (use absolute path)
    results_base = Path(__file__).parent / 'results'
    all_results = {}
    for seq_len in SEQUENCE_LENGTHS:
        results_path = results_base / f'seq_{seq_len}' / 'comparison_summary.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                all_results[seq_len] = json.load(f)

    if not all_results:
        print("No results found to visualize")
        return

    # Create comprehensive comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Experiment 3: Adaptive Per-Layer Sparsity - Comprehensive Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Validation Loss vs Sequence Length
    ax = axes[0, 0]
    for schedule in [s.value for s in SCHEDULES_TO_TEST]:
        seq_lens = []
        losses = []
        for seq_len in SEQUENCE_LENGTHS:
            if seq_len in all_results and schedule in all_results[seq_len]:
                seq_lens.append(seq_len)
                losses.append(all_results[seq_len][schedule]['val_loss'])
        if seq_lens:
            ax.plot(seq_lens, losses, marker='o', linewidth=2, label=schedule)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Plot 2: Validation Accuracy vs Sequence Length
    ax = axes[0, 1]
    for schedule in [s.value for s in SCHEDULES_TO_TEST]:
        seq_lens = []
        accs = []
        for seq_len in SEQUENCE_LENGTHS:
            if seq_len in all_results and schedule in all_results[seq_len]:
                seq_lens.append(seq_len)
                accs.append(all_results[seq_len][schedule]['val_accuracy'] * 100)
        if seq_lens:
            ax.plot(seq_lens, accs, marker='o', linewidth=2, label=schedule)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy vs Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Plot 3: Training Time Comparison
    ax = axes[0, 2]
    for schedule in [s.value for s in SCHEDULES_TO_TEST]:
        seq_lens = []
        times = []
        for seq_len in SEQUENCE_LENGTHS:
            if seq_len in all_results and schedule in all_results[seq_len]:
                seq_lens.append(seq_len)
                times.append(all_results[seq_len][schedule]['time_per_step'] * 1000)  # ms
        if seq_lens:
            ax.plot(seq_lens, times, marker='o', linewidth=2, label=schedule)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time per Step (ms)')
    ax.set_title('Training Speed Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)

    # Plot 4: Relative Performance at 1024 tokens (Focus on Exp2 failure case)
    ax = axes[1, 0]
    if 1024 in all_results:
        schedules = []
        losses = []
        for schedule in [s.value for s in SCHEDULES_TO_TEST]:
            if schedule in all_results[1024]:
                schedules.append(schedule.replace('_', '\n'))
                losses.append(all_results[1024][schedule]['val_loss'])

        colors = ['green' if 'dense_baseline' in s.lower() else 'blue' if 'uniform' in s.lower() else 'red'
                 for s in schedules]
        bars = ax.bar(range(len(schedules)), losses, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(schedules)))
        ax.set_xticklabels(schedules, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Validation Loss')
        ax.set_title('Performance at 1024 Tokens (Exp2 Failure Case)')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)

    # Plot 5: Improvement over Uniform Sparse
    ax = axes[1, 1]
    for seq_len in SEQUENCE_LENGTHS:
        if seq_len in all_results and 'uniform_sparse' in all_results[seq_len]:
            baseline_loss = all_results[seq_len]['uniform_sparse']['val_loss']
            improvements = []
            schedule_names = []

            for schedule in [s.value for s in SCHEDULES_TO_TEST]:
                if schedule != 'uniform_sparse' and schedule in all_results[seq_len]:
                    loss = all_results[seq_len][schedule]['val_loss']
                    improvement = ((baseline_loss - loss) / baseline_loss) * 100
                    improvements.append(improvement)
                    schedule_names.append(schedule)

            if improvements:
                x = np.arange(len(schedule_names))
                ax.bar(x + SEQUENCE_LENGTHS.index(seq_len) * 0.15, improvements,
                      width=0.15, label=f'Seq {seq_len}', alpha=0.7)

    if len(schedule_names) > 0:
        ax.set_xticks(x + 0.15)
        ax.set_xticklabels([s.replace('_', '\n') for s in schedule_names], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Improvement over Uniform (%)')
    ax.set_title('Improvement over Uniform Sparse (Exp2 Baseline)')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 6: Accuracy Comparison Table (as text)
    ax = axes[1, 2]
    ax.axis('off')
    table_data = []
    table_data.append(['Schedule', 'Seq 64', 'Seq 256', 'Seq 1024'])

    for schedule in [s.value for s in SCHEDULES_TO_TEST]:
        row = [schedule.replace('_', ' ').title()]
        for seq_len in SEQUENCE_LENGTHS:
            if seq_len in all_results and schedule in all_results[seq_len]:
                acc = all_results[seq_len][schedule]['val_accuracy'] * 100
                row.append(f'{acc:.1f}%')
            else:
                row.append('N/A')
        table_data.append(row)

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Validation Accuracy Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'results' / 'comprehensive_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")


def main():
    """Run full experiment"""
    print(f"\n{'#'*80}")
    print("EXPERIMENT 3: ADAPTIVE PER-LAYER SPARSITY")
    print(f"{'#'*80}\n")
    print("Research Hypothesis:")
    print("  Different transformer layers specialize in different functions:")
    print("  - Early layers: Local patterns (short-range dependencies)")
    print("  - Middle layers: Feature composition (functionally redundant)")
    print("  - Late layers: Global context (semantic abstraction)")
    print("")
    print("  Therefore, each layer should have optimized sparsity based on its role.")
    print(f"\n{'#'*80}\n")

    # Create results directory (absolute path)
    results_base = Path(__file__).parent / 'results'
    results_base.mkdir(exist_ok=True)
    print(f"📁 Base results directory: {results_base.absolute()}\n")

    # Run experiments for each sequence length
    for seq_len in SEQUENCE_LENGTHS:
        run_for_sequence_length(seq_len)

    # Create visualizations
    create_visualizations()

    results_base = Path(__file__).parent / 'results'

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved in: {results_base.absolute()}")
    print(f"  - Per-sequence-length results: {results_base}/seq_*/")
    print(f"  - Comprehensive comparison: {results_base}/comprehensive_comparison.png")
    print("\nKey files:")
    print(f"  - {results_base}/seq_1024/comparison_summary.json (Focus: Exp2 failure case)")
    print(f"  - {results_base}/comprehensive_comparison.png (All schedules compared)")


if __name__ == "__main__":
    main()
