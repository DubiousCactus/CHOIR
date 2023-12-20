#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Attention blocks.
"""


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
