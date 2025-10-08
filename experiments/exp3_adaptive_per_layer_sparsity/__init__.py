"""
Experiment 3: Adaptive Per-Layer Sparsity

Tests whether different transformer layers benefit from different sparsity levels
based on their functional specialization in the hierarchy.
"""

from .adaptive_sparse_attention import (
    AdaptiveSparseAttention,
    LightningIndexer,
    AdaptiveTopKSelector,
    SparsitySchedule,
    LayerSparsityConfig,
    create_sparsity_schedule,
    print_schedule_info
)

from .exp3_models import (
    AdaptiveSparseMoELLM,
    DenseMoELLM,
    AdaptiveSparseTransformerBlock,
    DenseTransformerBlock,
    create_adaptive_model,
    create_dense_model,
    count_parameters
)

__all__ = [
    # Attention modules
    'AdaptiveSparseAttention',
    'LightningIndexer',
    'AdaptiveTopKSelector',

    # Configuration
    'SparsitySchedule',
    'LayerSparsityConfig',
    'create_sparsity_schedule',
    'print_schedule_info',

    # Models
    'AdaptiveSparseMoELLM',
    'DenseMoELLM',
    'AdaptiveSparseTransformerBlock',
    'DenseTransformerBlock',
    'create_adaptive_model',
    'create_dense_model',
    'count_parameters',
]
