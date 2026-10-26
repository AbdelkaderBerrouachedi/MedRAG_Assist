#!/usr/bin/env python3
"""
MedRAG-Assist: Phase V - Linear Projection & Multi-Head Cross-Attention Fusion Layer.
PyTorch implementation of the dimension reconciliation mechanism bridging
768-dimensional BioBERT evidence embeddings into 4096-dimensional Llama-3-8B generative space.
"""

import math
import torch
import torch.nn as nn

class LinearProjectionBridge(nn.Module):
    """
    Linear projection matrix W_proj in R^(768 x 4096).
    Maps raw BioBERT representations E_raw in R^(L_R x 768)
    into the generative model's embedding space E_R in R^(L_R x 4096).
    """
    def __init__(self, d_bio: int = 768, d_model: int = 4096):
        super().__init__()
        self.proj = nn.Linear(d_bio, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, E_raw: torch.Tensor) -> torch.Tensor:
        # E_raw shape: [batch_size, L_R, 768]
        # E_R shape: [batch_size, L_R, 4096]
        return self.layer_norm(self.proj(E_raw))


class DecoupledCrossAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention Layer:
    Queries Q_D = E_D * W_Q (from dialogue context E_D in R^(L_D x 4096))
    Keys K_R = E_R * W_K (from projected evidence E_R in R^(L_R x 4096))
    Values V_R = E_R * W_V (from projected evidence E_R in R^(L_R x 4096))
    Number of heads: H = 32, per-head key dimension d_k = 128 (32 * 128 = 4096).
    Attention Complexity: O(L_D * L_R), bypassing O((L_D + L_R)^2).
    """
    def __init__(self, d_model: int = 4096, num_heads: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 128
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, E_D: torch.Tensor, E_R: torch.Tensor) -> torch.Tensor:
        """
        Args:
            E_D: Dialogue context tensor of shape [batch_size, L_D, 4096]
            E_R: Projected evidence tensor of shape [batch_size, L_R, 4096]
        Returns:
            E_F: Fused context tensor of shape [batch_size, L_D, 4096]
        """
        B, L_D, _ = E_D.shape
        _, L_R, _ = E_R.shape

        # Linear projections & split into heads: [B, num_heads, Seq_len, head_dim]
        Q = self.W_q(E_D).view(B, L_D, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(E_R).view(B, L_R, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(E_R).view(B, L_R, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product cross-attention: softmax(Q * K^T / sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # [B, num_heads, L_D, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, L_D, self.d_model)

        # Residual connection and layer normalization
        E_F = self.norm(E_D + self.out_proj(context))
        return E_F

if __name__ == "__main__":
    # Test forward pass with tensor shapes matching paper
    print("Testing LinearProjectionBridge and DecoupledCrossAttentionFusion...")
    bridge = LinearProjectionBridge(d_bio=768, d_model=4096)
    fusion = DecoupledCrossAttentionFusion(d_model=4096, num_heads=32)

    dummy_E_raw = torch.randn(1, 10, 768)    # 10 retrieved evidence passages
    dummy_E_D = torch.randn(1, 64, 4096)     # 64 dialogue turns

    dummy_E_R = bridge(dummy_E_raw)
    assert dummy_E_R.shape == (1, 10, 4096), f"E_R shape mismatch: {dummy_E_R.shape}"
    print("[OK] E_raw [1, 10, 768] -> W_proj -> E_R [1, 10, 4096] projection verified!")

    E_F = fusion(dummy_E_D, dummy_E_R)
    assert E_F.shape == (1, 64, 4096), f"E_F shape mismatch: {E_F.shape}"
    print("[OK] Cross-Attention Fusion E_F [1, 64, 4096] verified successfully!")
