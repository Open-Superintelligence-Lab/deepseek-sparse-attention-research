"""
Model Definitions for Experiment 3: Adaptive Per-Layer Sparse Attention

This module defines:
1. Adaptive Sparse Attention Model (with per-layer k values)
2. Dense Baseline Model (no sparsity)
3. Uniform Sparse Model (Exp2 baseline with uniform k)

All models use the same architecture (MoE + Attention) but differ in sparsity schedules.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
import sys
import os

# Add parent directories to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, root_dir)

# Import from parent models package
from models.components import MixtureOfExperts
from models.layers import MultiHeadAttention

# Import local adaptive sparse attention
from adaptive_sparse_attention import (
    AdaptiveSparseAttention,
    SparsitySchedule,
    create_sparsity_schedule,
    LayerSparsityConfig
)


class AdaptiveSparseTransformerBlock(nn.Module):
    """
    Transformer block with Adaptive Sparse Attention and MoE

    Each layer has a different top-k value based on its role in the hierarchy.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        layer_idx: int,
        layer_top_k: int,
        num_experts: int = 4,
        expert_top_k: int = 2,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Adaptive Sparse Attention
        self.attention = AdaptiveSparseAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            layer_idx=layer_idx,
            layer_top_k=layer_top_k,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
            dropout=dropout
        )

        # MoE layer
        self.feed_forward = MixtureOfExperts(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=expert_top_k,
            dropout=dropout
        )

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass

        Returns:
            - output: Block output
            - aux_loss: MoE auxiliary loss
            - attn_stats: Attention statistics if return_stats=True
        """
        # Self-attention with adaptive sparsity
        attn_out, attn_stats = self.attention(
            self.norm1(x),
            return_stats=return_stats
        )
        x = x + self.dropout(attn_out)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x, aux_loss, attn_stats


class DenseTransformerBlock(nn.Module):
    """
    Transformer block with Dense Attention and MoE (baseline)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 4,
        expert_top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Classic Dense Attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # MoE layer
        self.feed_forward = MixtureOfExperts(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=expert_top_k,
            dropout=dropout
        )

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass"""
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x, aux_loss


class AdaptiveSparseMoELLM(nn.Module):
    """
    Language Model with Adaptive Per-Layer Sparse Attention and MoE

    Different layers have different sparsity levels based on functional role.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        sparsity_schedule: SparsitySchedule,
        num_experts: int = 4,
        expert_top_k: int = 2,
        indexer_heads: int = 4,
        indexer_dim: int = 64,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.load_balancing_weight = load_balancing_weight

        # Create sparsity schedule
        self.sparsity_config = create_sparsity_schedule(
            schedule=sparsity_schedule,
            n_layers=n_layers,
            seq_len=max_seq_len
        )

        # Token embeddings
        self.embed = nn.Embedding(vocab_size, d_model)

        # Transformer blocks with adaptive sparsity
        self.blocks = nn.ModuleList([
            AdaptiveSparseTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                layer_idx=i,
                layer_top_k=self.sparsity_config.layer_k_values[i],
                num_experts=num_experts,
                expert_top_k=expert_top_k,
                indexer_heads=indexer_heads,
                indexer_dim=indexer_dim,
                dropout=dropout
            )
            for i in range(n_layers)
        ])

        # Final layer norm and output projection
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Dict]]]:
        """
        Forward pass

        Args:
            x: Input token indices [batch_size, seq_len]
            return_stats: Whether to return per-layer statistics

        Returns:
            - logits: Output logits [batch_size, seq_len, vocab_size]
            - total_aux_loss: Total auxiliary loss (MoE load balancing)
            - stats_list: List of per-layer attention statistics (if requested)
        """
        # Embed tokens
        x = self.embed(x)

        # Pass through transformer blocks
        total_aux_loss = 0.0
        stats_list = [] if return_stats else None

        for block in self.blocks:
            x, aux_loss, attn_stats = block(x, return_stats=return_stats)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss
            if return_stats and attn_stats is not None:
                stats_list.append(attn_stats)

        # Final normalization and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        # Scale auxiliary loss
        if total_aux_loss != 0.0:
            total_aux_loss = total_aux_loss * self.load_balancing_weight
        else:
            total_aux_loss = None

        return logits, total_aux_loss, stats_list

    def get_sparsity_info(self) -> Dict:
        """Get information about the sparsity schedule"""
        return {
            'schedule_name': self.sparsity_config.schedule_name,
            'description': self.sparsity_config.description,
            'layer_k_values': self.sparsity_config.layer_k_values,
            'layer_k_ratios': self.sparsity_config.layer_k_ratios
        }

    def enable_sparse_attention(self):
        """Enable sparse attention in all layers"""
        for block in self.blocks:
            block.attention.enable_sparse()

    def disable_sparse_attention(self):
        """Disable sparse attention (use dense) in all layers"""
        for block in self.blocks:
            block.attention.disable_sparse()

    def update_sparsity_schedule(self, new_schedule: SparsitySchedule, seq_len: int):
        """Dynamically update the sparsity schedule"""
        new_config = create_sparsity_schedule(
            schedule=new_schedule,
            n_layers=self.n_layers,
            seq_len=seq_len
        )
        self.sparsity_config = new_config

        # Update each layer's k value
        for i, block in enumerate(self.blocks):
            block.attention.update_layer_k(new_config.layer_k_values[i])


class DenseMoELLM(nn.Module):
    """
    Language Model with Dense Attention and MoE (baseline)

    No sparsity - all tokens attend to all tokens (within causal mask).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        num_experts: int = 4,
        expert_top_k: int = 2,
        dropout: float = 0.1,
        load_balancing_weight: float = 0.01
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.load_balancing_weight = load_balancing_weight

        # Token embeddings
        self.embed = nn.Embedding(vocab_size, d_model)

        # Transformer blocks with dense attention
        self.blocks = nn.ModuleList([
            DenseTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                num_experts=num_experts,
                expert_top_k=expert_top_k,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Final layer norm and output projection
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass"""
        # Embed tokens
        x = self.embed(x)

        # Pass through transformer blocks
        total_aux_loss = 0.0

        for block in self.blocks:
            x, aux_loss = block(x)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        # Final normalization and projection
        x = self.norm(x)
        logits = self.lm_head(x)

        # Scale auxiliary loss
        if total_aux_loss != 0.0:
            total_aux_loss = total_aux_loss * self.load_balancing_weight
        else:
            total_aux_loss = None

        return logits, total_aux_loss


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_adaptive_model(config: dict, schedule: SparsitySchedule) -> AdaptiveSparseMoELLM:
    """
    Create adaptive sparse attention model from config

    Args:
        config: Configuration dictionary
        schedule: Sparsity schedule to use

    Returns:
        model: AdaptiveSparseMoELLM instance
    """
    model = AdaptiveSparseMoELLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        sparsity_schedule=schedule,
        num_experts=config.get('num_experts', 4),
        expert_top_k=config.get('expert_top_k', 2),
        indexer_heads=config.get('indexer_heads', 4),
        indexer_dim=config.get('indexer_dim', 64),
        dropout=config.get('dropout', 0.1),
        load_balancing_weight=config.get('load_balancing_weight', 0.01)
    )
    return model


def create_dense_model(config: dict) -> DenseMoELLM:
    """
    Create dense attention model from config

    Args:
        config: Configuration dictionary

    Returns:
        model: DenseMoELLM instance
    """
    model = DenseMoELLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        num_experts=config.get('num_experts', 4),
        expert_top_k=config.get('expert_top_k', 2),
        dropout=config.get('dropout', 0.1),
        load_balancing_weight=config.get('load_balancing_weight', 0.01)
    )
    return model
