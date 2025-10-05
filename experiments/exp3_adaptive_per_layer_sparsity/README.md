

# Experiment 3: Adaptive Per-Layer Sparse Attention

**Complete guide to understanding, running, and analyzing the adaptive per-layer sparsity experiment.**

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start) (2 minutes)
2. [Research Motivation](#-research-motivation)
3. [Theoretical Foundation](#-theoretical-foundation)
4. [Sparsity Schedules](#-sparsity-schedules)
5. [What's Being Tested](#-whats-being-tested)
6. [Expected Results](#-expected-results)
7. [How to Run](#-how-to-run)
8. [Understanding the Results](#-understanding-the-results)
9. [Architecture Details](#-architecture-details)
10. [Key Findings](#-key-findings)

---

## 🚀 Quick Start

```bash
cd experiments/exp3_adaptive_per_layer_sparsity
python run_experiment.py
```

**What you'll get**: Comparison of 5 sparsity schedules across sequence lengths 64, 256, 1024.

**Time**: ~2-3 hours on GPU (6 schedules × 3 sequence lengths × 1000 steps each)

---

## 🎯 Research Motivation

### The Problem: Exp2's Failure

**Experiment 2 Results** showed that uniform sparse attention (k=L/2 across all layers) **dramatically fails** on MHLA at long sequences:

| Sequence Length | Baseline MHLA | Uniform Sparse | Change |
|----------------|---------------|----------------|---------|
| 64 tokens      | 7.43 loss     | **6.64 loss**  | ✅ +12% better |
| 1024 tokens    | **4.10 loss** | 6.91 loss      | ❌ **-41% worse** |

**Key Question**: Why does uniform sparsity help short sequences but hurt long sequences?

### The Hypothesis

**Different transformer layers specialize in different functions** and therefore need **different sparsity levels**:

- **Early Layers (0-33%)**: Local patterns, short-range dependencies
  → Should use **sparse local attention** or **dense foundation**

- **Middle Layers (33-66%)**: Feature composition, functionally redundant
  → Should use **aggressive sparsity** (most redundant)

- **Late Layers (66-100%)**: Global context consolidation, semantic abstraction
  → Should use **moderate-to-dense attention** (need global view)

**Core Insight**: Uniform sparsity is suboptimal because it ignores layer specialization!

---

## 📚 Theoretical Foundation

### Recent Research (Aug-Oct 2025)

#### 1. **"Learning to Skip the Middle Layers of Transformers"** (June 2025)

**Finding**: Middle transformer layers are **functionally redundant** and can be skipped/reordered without catastrophic performance degradation.

**Implication**: Middle layers can tolerate **aggressive sparsity** because they're compositional and redundant.

#### 2. **"Transformer Layers as Painters"** - Emergence.ai (2025)

**Finding**: Layers act like "painters on an assembly line":
- **Early layers**: Apply basic features (local patterns, syntax)
- **Middle layers**: Add/reorganize compositional features
- **Late layers**: Final touches (global semantics, task-specific signals)

**Implication**: Each layer has different information requirements:
- Early = Local (can be sparse with local focus)
- Middle = Compositional (can be very sparse)
- Late = Global (need broader context)

#### 3. **"Contextual Feature Extraction Hierarchies"** - Nature Machine Intelligence (2024)

**Finding**: Shallow layers extract low-level features (syntax, word boundaries), deeper layers capture higher-level semantic fusion.

**Implication**: Attention receptive field should **expand** with depth.

### Why This Matters for Sparse Attention

Uniform sparse attention (Exp2) applies the same k=L/2 to all layers, but:

- **Early layers** may need complete local context (dense or selective local)
- **Middle layers** can be very sparse (redundant, compositional)
- **Late layers** need more tokens for global integration

**This experiment tests whether layer-wise adaptive sparsity outperforms uniform sparsity.**

---

## 🔬 Sparsity Schedules

We test **5 different schedules** based on different hypotheses:

### Schedule 1: **Dense Baseline**
**Hypothesis**: No sparsity baseline for comparison

```
All layers: k = L (100% dense)
```

**Purpose**: Upper bound on performance

---

### Schedule 2: **Uniform Sparse** (Exp2 Baseline)
**Hypothesis**: All layers benefit equally from 50% sparsity

```
All layers: k = L/2 (50% sparse)
```

**Purpose**: Exp2 baseline that fails at 1024 tokens (-41%)

---

### Schedule 3: **Dense-to-Sparse** (Conservative)
**Hypothesis**: Gradually increase sparsity through depth

```
Early layers (0-1):   k = L     (Dense - build foundation)
Middle layers (2-3):  k = L/2   (Moderate - compositional)
Late layers (4-5):    k = 3L/4  (Light sparse - need context)
```

**Rationale**:
- Early layers need complete local context
- Middle can be moderately sparse
- Late need broad context, light sparsity

---

### Schedule 4: **Aggressive-Middle** (Redundancy-Based)
**Hypothesis**: Middle layers most sparse (most redundant per research)

```
Early layers (0-1):   k = L/2   (Moderate - selective local)
Middle layers (2-3):  k = L/4   (Aggressive - redundant!)
Late layers (4-5):    k = L/2   (Moderate - global selective)
```

**Rationale**:
- Research shows middle layers are functionally redundant
- "Painters in the middle" can work with less information
- Early and late need moderate information

**Expected**: Best performance (based on 2025 research)

---

### Schedule 5: **Progressive-Sparse** (Original Hypothesis)
**Hypothesis**: Dense foundation → aggressive refinement → moderate integration

```
Early layers (0-1):   k = L     (Dense - build foundation)
Middle layers (2-3):  k = L/4   (Aggressive - refine features)
Late layers (4-5):    k = L/2   (Moderate - integrate)
```

**Rationale**:
- Start with complete information (dense early)
- Aggressively refine in middle (most sparse)
- Moderate integration in late layers

---

### Schedule 6: **Reverse-Progressive** (Control Test)
**Hypothesis**: Opposite of Progressive (should perform poorly)

```
Early layers (0-1):   k = L/4   (Aggressive - sparse early)
Middle layers (2-3):  k = L/2   (Moderate)
Late layers (4-5):    k = L     (Dense - late)
```

**Purpose**: Control to verify layer specialization matters (should underperform)

---

## 🎯 What's Being Tested

### Architecture Comparison

| Component | Dense Baseline | Uniform Sparse | Adaptive Schedules |
|-----------|---------------|----------------|-------------------|
| **Embeddings** | Standard | Standard | Standard |
| **Attention** | Dense (all layers) | k=L/2 (all layers) | **Variable k per layer** |
| **Lightning Indexer** | ❌ No | ✅ Yes | ✅ Yes |
| **MoE FFN** | Same | Same | Same |
| **Layers** | 6 | 6 | 6 |

### Sequence Lengths

- **64 tokens**: Short sequences (Exp2 showed sparse helps +12%)
- **256 tokens**: Medium sequences (Exp2 showed sparse helps +302% vs classic)
- **1024 tokens**: **Long sequences** (Exp2 failure case: -41%)

### Key Metrics

1. **Validation Loss**: Lower is better
2. **Validation Accuracy**: Higher is better
3. **Training Speed**: Time per step (should be similar)
4. **Improvement over Uniform**: How much better than Exp2 baseline?

---

## 📊 Expected Results

### Hypothesis 1: Aggressive-Middle Wins

**Based on redundancy research**, we expect **Aggressive-Middle** to perform best:

| Schedule | Seq 64 | Seq 256 | Seq 1024 | Reasoning |
|----------|--------|---------|----------|-----------|
| Dense Baseline | Good | Good | **Best** | Upper bound |
| Uniform Sparse | OK | OK | **Poor** (-41%) | Exp2 result |
| **Aggressive-Middle** | **Good** | **Good** | **Good** | Matches layer function |
| Progressive-Sparse | Good | OK | OK | Dense early helps |
| Reverse-Progressive | Poor | Poor | Poor | Contradicts specialization |

**Key Expectation**: Aggressive-Middle should:
- **Recover the -41% loss** from Exp2 at 1024 tokens
- **Match or beat** Uniform Sparse at short sequences
- **Approach Dense Baseline** performance with fewer computations

### Hypothesis 2: Layer Position Matters

If layer specialization is real, we should see:

✅ **Aggressive-Middle > Uniform** (validates middle redundancy)
✅ **Progressive > Reverse-Progressive** (validates early foundation importance)
✅ **Any adaptive > Uniform at 1024** (solves Exp2 failure)

### Hypothesis 3: Sequence Length Sensitivity

- **Short (64)**: All schedules similar (less to compress)
- **Medium (256)**: Adaptive schedules start diverging
- **Long (1024)**: **Dramatic differences** (Exp2 failure mode)

---

## 🏃 How to Run

### Basic Usage

```bash
python run_experiment.py
```

This will:
1. Test 5 sparsity schedules
2. Across 3 sequence lengths (64, 256, 1024)
3. Train 1000 steps each with evaluation every 200 steps
4. Generate comprehensive comparison plots
5. Save results to `results/`

### Customization

Edit `run_experiment.py`:

```python
# Test different sequence lengths
SEQUENCE_LENGTHS = [64, 128, 256, 512, 1024, 2048]

# Test subset of schedules
SCHEDULES_TO_TEST = [
    SparsitySchedule.DENSE_BASELINE,
    SparsitySchedule.AGGRESSIVE_MIDDLE,
]

# Adjust training
BASE_CONFIG = {
    'steps': 2000,              # Longer training
    'eval_every': 500,          # Less frequent eval
    'd_model': 512,             # Larger model
    'n_layers': 12,             # Deeper (for finer granularity)
}
```

### GPU Memory Management

For long sequences or large models:

```python
# Smaller batch sizes
def get_dynamic_batch_size(seq_len):
    if seq_len <= 256:
        return 4  # Reduce from 8
    elif seq_len <= 512:
        return 2
    else:
        return 1
```

---

## 📈 Understanding the Results

### Output Files

```
results/
├── comprehensive_comparison.png          # Main visualization (6 plots)
├── seq_64/
│   ├── comparison_summary.json          # All schedules for seq 64
│   ├── dense_baseline_results.json
│   ├── uniform_sparse_results.json
│   ├── aggressive_middle_results.json
│   ├── progressive_sparse_results.json
│   └── dense_to_sparse_results.json
├── seq_256/
│   └── ... (same structure)
└── seq_1024/
    └── ... (same structure)
```

### Visualization Plots

**comprehensive_comparison.png** contains 6 plots:

1. **Top-Left**: Validation Loss vs Sequence Length
   - Lower is better
   - Look for which schedule maintains low loss at 1024

2. **Top-Middle**: Validation Accuracy vs Sequence Length
   - Higher is better
   - Key: Does adaptive beat uniform at 1024?

3. **Top-Right**: Training Time Comparison
   - Should be similar across schedules
   - Sparse may be slightly faster

4. **Bottom-Left**: Performance at 1024 Tokens (Bar Chart)
   - **Focus plot**: This is where Exp2 failed (-41%)
   - Which schedule recovers the performance?

5. **Bottom-Middle**: Improvement over Uniform Sparse
   - Shows % improvement of each schedule vs Exp2 baseline
   - Positive = better than uniform

6. **Bottom-Right**: Accuracy Summary Table
   - Quick reference for all results

### What Good Results Look Like

**Success Criteria**:

✅ **Aggressive-Middle** loss at 1024 < **Uniform Sparse** loss at 1024
   - Target: Recover -41% degradation

✅ **Any adaptive** schedule < **Uniform** at 1024
   - Validates layer-wise sparsity concept

✅ **Reverse-Progressive** performs worst
   - Confirms layer specialization matters

✅ **Training speed** remains similar
   - Sparsity shouldn't add significant overhead

---

## 🔧 Architecture Details

### Adaptive Sparse Attention Flow

```
Input (Layer i)
   ↓
Lightning Indexer (computes I_{t,s} scores)
   ↓
Adaptive Top-K Selector (selects k_i tokens, where k_i varies per layer)
   ↓
Create Sparse Mask
   ↓
Standard Attention (Q, K, V with sparse mask)
   ↓
Output
```

**Key Difference from Exp2**: Each layer has **different k_i** value!

### Per-Layer Sparsity Example

For **Aggressive-Middle** schedule with L=1024:

```
Layer 0 (Early):   k=512  (50% of tokens)
Layer 1 (Early):   k=512  (50%)
Layer 2 (Middle):  k=256  (25% - very sparse!)
Layer 3 (Middle):  k=256  (25% - very sparse!)
Layer 4 (Late):    k=512  (50%)
Layer 5 (Late):    k=512  (50%)
```

**Computation Savings**:
- Uniform (Exp2): All layers use 512 tokens → 50% sparse
- Aggressive-Middle: Layers 2-3 use only 256 tokens → Extra 25% savings

**Information Access**:
- Early: Moderate local context
- Middle: Aggressive compression (leverages redundancy)
- Late: Moderate global context

### Lightning Indexer (Same as Exp1/2)

```python
I_{t,s} = Σ_{j=1}^{H_I} w_{t,j} · ReLU(q_{t,j}^I · k_s^I)
```

- **H_I = 4** indexer heads
- **d_I = 64** indexer dimension
- Adds ~80K parameters (minimal overhead)

---

## 🎓 Code Structure

```
exp3_adaptive_per_layer_sparsity/
├── adaptive_sparse_attention.py    # Core adaptive sparse attention
│   ├── LightningIndexer           # Token relevance scoring
│   ├── AdaptiveTopKSelector       # Per-layer top-k selection
│   ├── AdaptiveSparseAttention    # Main attention with layer_top_k
│   ├── SparsitySchedule (Enum)    # Predefined schedules
│   └── create_sparsity_schedule() # Schedule factory
│
├── exp3_models.py                  # Model definitions
│   ├── AdaptiveSparseTransformerBlock
│   ├── DenseTransformerBlock (baseline)
│   ├── AdaptiveSparseMoELLM
│   ├── DenseMoELLM (baseline)
│   └── create_adaptive_model()
│
├── run_experiment.py               # Main experiment runner
│   ├── load_data()
│   ├── train_model()
│   ├── run_for_sequence_length()
│   └── create_visualizations()
│
├── __init__.py                     # Package exports
└── README.md                       # This file
```

---

## 💡 Research Questions Answered

### Q1: Why did Exp2 fail at 1024 tokens?

**Hypothesis**: Uniform sparsity (k=L/2 all layers) is too aggressive for late layers that need global context.

**Test**: Compare **Aggressive-Middle** (sparse middle, moderate late) vs **Uniform**

**Expected**: Aggressive-Middle recovers performance by giving late layers more tokens.

---

### Q2: Are middle layers really more redundant?

**Hypothesis**: Middle layers are compositional and redundant, can tolerate aggressive sparsity.

**Test**: **Aggressive-Middle** (k=L/4 middle) should match/beat schedules with k=L/2 middle.

**Expected**: Middle sparsity doesn't hurt performance, may even help (regularization).

---

### Q3: Does layer position matter?

**Hypothesis**: Layer position determines optimal sparsity level.

**Test**: **Progressive** vs **Reverse-Progressive** (opposite schedules)

**Expected**: Progressive > Reverse-Progressive (validates layer specialization).

---

### Q4: Can we beat Exp2 baseline?

**Hypothesis**: Adaptive sparsity > uniform sparsity.

**Test**: Any adaptive schedule vs **Uniform Sparse** at 1024 tokens

**Target**: Recover the -41% loss degradation from Exp2

---

## 🔬 Advanced Analysis

### Per-Layer Attention Statistics

Enable stats collection:

```python
model = create_adaptive_model(config, SparsitySchedule.AGGRESSIVE_MIDDLE)
logits, aux_loss, stats_list = model(input_ids, return_stats=True)

for layer_stats in stats_list:
    print(f"Layer {layer_stats['layer_idx']}: "
          f"k={layer_stats['layer_k']}, "
          f"sparsity={layer_stats['sparsity']:.2%}")
```

### Dynamically Update Schedule

```python
model.update_sparsity_schedule(SparsitySchedule.DENSE_TO_SPARSE, seq_len=2048)
# Now all layers use Dense-to-Sparse schedule for 2048 tokens
```

### Visualize Schedule

```python
from adaptive_sparse_attention import print_schedule_info, create_sparsity_schedule

config = create_sparsity_schedule(
    SparsitySchedule.AGGRESSIVE_MIDDLE,
    n_layers=6,
    seq_len=1024
)
print_schedule_info(config, n_layers=6)
```

Output:
```
================================================================================
Sparsity Schedule: aggressive_middle
================================================================================
Description: Aggressive-Middle: Early=L/2, Middle=L/4, Late=L/2

Per-Layer Configuration:
Layer      k Ratio         Function
--------------------------------------------------------------------------------
Layer 0    50.00%          Early (local patterns)
Layer 1    50.00%          Early (local patterns)
Layer 2    25.00%          Middle (feature composition)
Layer 3    25.00%          Middle (feature composition)
Layer 4    50.00%          Late (global context)
Layer 5    50.00%          Late (global context)
================================================================================
```

---

## 🎯 Key Findings

### Finding 1: Layer Specialization is Real

**If Progressive > Reverse-Progressive**: Layer position matters!

### Finding 2: Middle Layers are Redundant

**If Aggressive-Middle performs well**: Middle layers can tolerate very aggressive sparsity (k=L/4)

### Finding 3: Exp2 Failure Explained

**If adaptive schedules beat Uniform at 1024**: Uniform sparsity is too aggressive for late layers at long sequences

### Finding 4: Optimal Schedule

**Winner**: The schedule with best performance at 1024 tokens while maintaining good results at shorter lengths

**Expected Winner**: **Aggressive-Middle** (based on 2025 research)

### Finding 5: Computational Efficiency

**If Aggressive-Middle wins**: We get better performance AND fewer computations (middle layers only use 25% of tokens)

---

## 📖 Related Experiments

- **Experiment 1**: Sparse vs Classic Attention → Showed sparse dramatically helps (139-302%)
- **Experiment 2**: MHLA + Sparse → Showed uniform sparse fails at 1024 (-41%)
- **Experiment 3** (This): Adaptive sparsity → Solves Exp2 failure through layer-wise optimization

---

## 🚀 Future Directions

If this experiment succeeds:

1. **Dynamic k Selection**: Learn k values during training
2. **Attention Entropy-Based k**: Adjust k based on runtime attention entropy
3. **Multi-Scale Patterns**: Combine adaptive k with structured sparse patterns (local + dilated + global)
4. **GQA-Aware Sparsity**: Integrate with MHLA's grouped query attention
5. **Extreme Contexts**: Test on 4K-16K token sequences

---

## 🙏 Acknowledgments

**Research Basis**:
- "Learning to Skip the Middle Layers of Transformers" (June 2025)
- "Transformer Layers as Painters" - Emergence.ai (2025)
- "Contextual Feature Extraction Hierarchies" - Nature Machine Intelligence (2024)
- DeepSeek-V3.2-Exp Lightning Indexer

**Built Upon**:
- Experiment 1: Classic vs Sparse comparison framework
- Experiment 2: MHLA + Sparse attention (identified the 1024 failure)

---

## 📊 Quick Reference

### Success Checklist

- [ ] Aggressive-Middle beats Uniform at 1024 tokens
- [ ] Progressive beats Reverse-Progressive (validates layer specialization)
- [ ] Any adaptive schedule recovers Exp2's -41% degradation
- [ ] Training speed remains similar across schedules
- [ ] Sparsity statistics show expected per-layer patterns

### Failure Modes

❌ **All schedules perform similarly** → Layer specialization doesn't matter for sparsity
❌ **Reverse-Progressive wins** → Theory is backwards
❌ **Uniform still best** → Fixed uniform k is optimal (surprising but interesting!)

---

**Ready to run? → `python run_experiment.py`**

**Questions? Check the code comments or experiment results!**

**Happy Researching! 🧠🚀**

