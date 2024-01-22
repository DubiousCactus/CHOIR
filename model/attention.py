#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Attention blocks.
"""


from typing import Optional

import torch


class VectorAttentionLayer(torch.nn.Module):
    """Computes the (normalized) scaled dot product attention for one key, one query and
    one value. Not to be used for general attention with multiple keys, queries and values.
    """

    def __init__(self, q_dim: int, k_dim: int, v_dim: int, output_dim: int) -> None:
        super().__init__()
        self.w_q = torch.nn.Linear(q_dim, output_dim, bias=False)
        self.w_k = torch.nn.Linear(k_dim, output_dim, bias=False)
        self.w_v = torch.nn.Linear(v_dim, output_dim, bias=False)
        self.scale = torch.sqrt(torch.tensor(output_dim))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Computes the (normalized) scaled dot product attention for one key, one query and
        one value.
        Args:
            q: (batch_size, d_q)
            k: (batch_size, d_k)
            v: (batch_size, d_v)
        Returns:
            (batch_size, n, d_v)
        """
        q_ = self.w_q(q)
        k_ = self.w_k(k)
        v_ = self.w_v(v)
        dot_p = q_ @ k_.transpose(0, 1)
        scaled_dotp = torch.nn.functional.softmax(dot_p / self.scale, dim=-1)
        output = scaled_dotp @ v_
        return output


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        n_heads: int = 8,
        dim_head: int = 64,
        use_bias: bool = False,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * n_heads
        self.scale = torch.sqrt(torch.tensor(dim_head))
        self.heads = n_heads
        self.heads_k = torch.nn.Linear(k_dim, inner_dim, bias=use_bias)
        self.heads_q = torch.nn.Linear(q_dim, inner_dim, bias=use_bias)
        self.heads_v = torch.nn.Linear(v_dim, inner_dim, bias=use_bias)
        self.out_proj = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, q_dim, bias=use_bias),
            torch.nn.Dropout(p_dropout),
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Implements multi-head cross attention.
        I have used OpenAI's implementation as reference:
        https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/attention.py#L152
        Args:
            q: (B, C, D, H, W)
            k: (B, K, A, B, C)
            v: (B, K, A, B, C)
        Returns:
            (B, C, D, H, W)
        """
        bs, q_c, d, h, w = q.shape
        _, k_c, _, _, _ = k.shape
        _, v_c, _, _, _ = v.shape
        q_ = q.view(bs, q_c, -1).permute(0, 2, 1)  # (B, D*H*W, C)
        k_ = k.view(bs, k_c, -1).permute(0, 2, 1)  # (B, A*B*C, K)
        v_ = v.view(bs, v_c, -1).permute(0, 2, 1)  # (B, A*B*C, K)

        q_ = self.heads_q(q_)  # (B, D*H*W, heads*head_dim)
        k_ = self.heads_k(k_)  # (B, A*B*C, heads*head_dim)
        v_ = self.heads_v(v_)  # (B, A*B*C, heads*head_dim)

        def rearrange(x: torch.Tensor) -> torch.Tensor:
            b, d, inner_dim = x.shape  # inner_dim = heads * head_dim
            head_dim = inner_dim // self.heads
            return (
                x.reshape(b, d, self.heads, head_dim)  # (B, D*H*W, heads, head_dim)
                .permute(0, 2, 1, 3)  # (B, heads, D*H*W, head_dim)
                .reshape(b * self.heads, d, head_dim)  # (B*heads, D*H*W, head_dim)
            )

        q_, k_, v_ = map(rearrange, (q_, k_, v_))
        # torch.einsum allows to compute matrix multiplications and other multi-dimensional linear
        # algebra operations using Einstein's notation. It is super convenient to compute
        # dot-products with multiple dimensions of batches.
        scaled_dotp = torch.einsum("b i d, b j d -> b i j", q_, k_) / self.scale
        attn_w = scaled_dotp.softmax(dim=-1)
        out = torch.einsum(
            "b i j, b j d -> b i d", attn_w, v_
        )  # (B*heads, D*H*W, head_dim)
        b_h, n, head_dim = out.shape
        inner_dim = head_dim * self.heads
        out = (
            out.reshape(bs, self.heads, n, head_dim)  # (B, heads, D*H*W, head_dim)
            .permute(0, 2, 1, 3)  # (B, D*H*W, heads, head_dim)
            .reshape(bs, n, inner_dim)  # (B, D*H*W, heads*head_dim)
        )  # (B, D*H*W, heads*head_dim)
        return self.out_proj(out).view(bs, q_c, d, h, w)  # (B, C, D, H, W)


class GEGLU(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        project_in = (
            torch.nn.Sequential(torch.nn.Linear(dim, inner_dim), torch.nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = torch.nn.Sequential(
            project_in, torch.nn.Dropout(dropout), torch.nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class SpatialTransformer(torch.nn.Module):
    def __init__(
        self, channels: int, gated_ff: bool = True, context_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        self.in_channels = channels
        self.ff = FeedForward(channels, dim_out=channels, glu=gated_ff)
        # self.norm_1 = torch.nn.LayerNorm()
        # self.norm_2 = torch.nn.LayerNorm()
        # self.norm_3 = torch.nn.LayerNorm()
        self.norm_1 = torch.nn.GroupNorm(16, channels)
        self.norm_2 = torch.nn.GroupNorm(16, channels)
        self.norm_3 = torch.nn.LayerNorm(channels)
        self.self_attn = MultiHeadAttention(
            channels, channels, channels, n_heads=8, dim_head=64
        )
        self.cross_attn = MultiHeadAttention(
            channels, context_dim or channels, context_dim, n_heads=8, dim_head=64
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (B, C, D, H, W)
            context: (B, C, K, L, M)
        Returns:
            (B, C, D, H, W)
        """
        _x = x
        # print(f"X shape: {x.shape}")
        # print(f"context shape: {context.shape if context is not None else None}")
        # print(f"self shape: {self.self_attn(x, x, x).shape}")
        # print(self.norm_1)
        context = context if context is not None else x
        x = self.norm_1(self.self_attn(x, x, x) + x)
        # print(f"After self attn: {x.shape}")
        x = self.norm_2(self.cross_attn(x, context, context) + x)
        # print(f"After cross attn: {x.shape}")
        bs, c, d, h, w = x.shape
        x = x.view(bs, c, -1).permute(0, 2, 1)  # (B, D*H*W, C)
        # print(f"ff shape: {self.ff(x).shape}")
        x = self.norm_3(self.ff(x) + x)
        x = x.permute(0, 2, 1).view(bs, c, d, h, w)  # (B, C, D, H, W)
        return x + _x
