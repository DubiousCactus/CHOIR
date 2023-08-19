#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


from typing import Optional

import torch


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        kq_dim_in: int,
        kq_dim_out: int,
        v_dim_in: int,
        v_dim_out: Optional[int] = None,
        n_heads: int = 8,
        use_bias: bool = False,
        normalize: bool = True,
        post_process: bool = True,
    ):
        super().__init__()
        self._normalize = normalize
        if v_dim_out is None:
            v_dim_out = v_dim_in
        assert v_dim_out is not None
        head_kq_dim_out = kq_dim_out // n_heads
        head_v_dim_out = v_dim_out // n_heads
        self._heads_k = torch.nn.ModuleList(
            [
                torch.nn.Linear(kq_dim_in, head_kq_dim_out, bias=use_bias)
                for _ in range(n_heads)
            ]
        )
        self._heads_q = torch.nn.ModuleList(
            [
                torch.nn.Linear(kq_dim_in, head_kq_dim_out, bias=use_bias)
                for _ in range(n_heads)
            ]
        )
        self._heads_v = torch.nn.ModuleList(
            [
                torch.nn.Linear(v_dim_in, head_v_dim_out, bias=use_bias)
                for _ in range(n_heads)
            ]
        )
        self._head_o = (
            torch.nn.Linear(n_heads * head_v_dim_out, v_dim_out, bias=use_bias)
            if post_process
            else None
        )

    def _attention(self, q, k, v):
        """Computes the (normalized) scaled dot product attention.
        Args:
            q: (batch_size, n, d_q)
            k: (batch_size, m, d_k)
            v: (batch_size, m, d_v)
        Returns:
            (batch_size, n, d_v)
        """
        # print("Attention")
        # print("Q: ", q.shape)
        # print("K: ", k.shape)
        # print("V: ", v.shape)
        assert k.shape[1] == v.shape[1]
        assert q.shape[-1] == k.shape[-1]
        # torch.einsum allows to compute matrix multiplications and other multi-dimensional linear
        # algebra operations using Einstein's notation. It is super convenient to compute
        # dot-products with multiple dimensions of batches.
        dot_p = torch.einsum("bnd,bmd->bnm", q, k)
        scale = torch.sqrt(torch.tensor(k.shape[-1]))
        # print("Weights dot-p: ", dot_p.shape)
        scaled_dotp = dot_p / scale
        assert scaled_dotp.shape[1] == q.shape[1]
        assert scaled_dotp.shape[2] == v.shape[1]
        if self._normalize:
            scaled_dotp = torch.nn.functional.softmax(scaled_dotp, dim=-1)
        else:
            scaled_dotp = torch.sigmoid(scaled_dotp)
        output = torch.einsum("bnm,bmd->bnd", scaled_dotp, v)
        # print("Scaled dot product: ", output.shape)
        assert output.shape[1] == q.shape[1]
        assert output.shape[2] == v.shape[-1]
        return output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        output = []
        # print("Q: ", q.shape)
        # print("K: ", k.shape)
        # print("V: ", v.shape)

        for h in range(len(self._heads_k)):
            # print("Head: ", h)
            # print("Head Q: ", self._heads_q[h](q).shape)
            # print("Head K: ", self._heads_k[h](k).shape)
            # print(
            # "Head V: ",
            # self._heads_v[h](
            # self._attention(self._heads_q[h](q), self._heads_k[h](k), v)
            # ).shape,
            # )
            output.append(
                self._heads_v[h](
                    self._attention(self._heads_q[h](q), self._heads_k[h](k), v)
                )
            )
        output = torch.cat(output, dim=-1)
        # print(output.shape)
        # print(self._head_o(output).shape)
        if self._head_o is not None:
            output = self._head_o(output)
        return output


class AttentionAggregator(torch.nn.Module):
    def __init__(
        self,
        type: str,
        n_heads: int = 8,
        normalize: bool = True,
        laplace_scale: float = 1.0,
        multi_head_use_bias: bool = False,
        k_dim_in: Optional[int] = None,
        k_dim_out: Optional[int] = None,
        q_dim_in: Optional[int] = None,
        q_dim_out: Optional[int] = None,
        v_dim_in: Optional[int] = None,
        v_dim_out: Optional[int] = None,
    ):
        super().__init__()
        self._normalize = normalize
        self._laplace_scale = laplace_scale
        self._attn_fn = {
            "uniform": self._uniform,
            "laplace": self._laplace,
            "dot_product": self._dot_product,
            "multi_head": self._multi_head,
            "multi_head_pytorch": self._multi_head_pytorch,
        }[type]
        if type == "multi_head":
            assert k_dim_in is not None
            assert q_dim_in is not None
            assert v_dim_in is not None
            assert k_dim_in == q_dim_in
            assert k_dim_out is not None
            assert q_dim_out is not None
            self._multihead = MultiHeadAttention(
                k_dim_in,
                k_dim_out,
                v_dim_in,
                v_dim_out=v_dim_out,
                n_heads=n_heads,
                use_bias=multi_head_use_bias,
            )
        elif type == "multi_head_pytorch":
            assert k_dim_in is not None
            assert q_dim_in is not None
            assert v_dim_in is not None
            assert k_dim_in == q_dim_in
            assert k_dim_out is not None
            assert q_dim_out is not None
            if v_dim_out is None:
                v_dim_out = v_dim_in
            self._multihead = torch.nn.MultiheadAttention(
                v_dim_out * n_heads,
                n_heads,
                dropout=0.0,
                kdim=k_dim_in,
                vdim=v_dim_in,
                bias=multi_head_use_bias,
            )

    def _uniform(self, q, k, v):
        return torch.mean(v, dim=1, keepdim=True).expand(-1, q.shape[1], -1)

    def _laplace(self, q, k, v):
        raise NotImplementedError

    def _dot_product(self, q, k, v):
        """Computes the (normalized) scaled dot product attention.
        Args:
            q: (batch_size, n, d_q)
            k: (batch_size, m, d_k)
            v: (batch_size, m, d_v)
        Returns:
            (batch_size, n, d_v)
        """
        # print("Q: ", q.shape)
        # print("K: ", k.shape)
        # print("V: ", v.shape)
        assert k.shape[1] == v.shape[1]
        assert q.shape[-1] == k.shape[-1]
        # torch.einsum allows to compute matrix multiplications and other multi-dimensional linear
        # algebra operations using Einstein's notation. It is super convenient to compute
        # dot-products with multiple dimensions of batches.
        dot_p = torch.einsum("bnd,bmd->bnm", q, k)
        scale = torch.sqrt(torch.tensor(k.shape[-1]))
        # print("Weights dot-p: ", weights.shape)
        scaled_dotp = dot_p / scale
        assert scaled_dotp.shape[1] == q.shape[1]
        assert scaled_dotp.shape[2] == v.shape[1]
        if self._normalize:
            scaled_dotp = torch.nn.functional.softmax(scaled_dotp, dim=-1)
        else:
            scaled_dotp = torch.sigmoid(scaled_dotp)
        output = torch.einsum("bnm,bmd->bnd", scaled_dotp, v)
        # print("Scaled dot product: ", output.shape)
        assert output.shape[1] == q.shape[1]
        assert output.shape[2] == v.shape[-1]
        return output

    def _multi_head(self, q, k, v):
        return self._multihead(q, k, v)

    def _multi_head_pytorch(self, q, k, v):
        # TODO: Pre-embed q, k, v? It won't work as is because of dim issues
        raise NotImplementedError
        # Returns a tuple: attn_output, attn_output_weights
        return self._multihead(q, k, v)[0]

    def forward(self, x, r, x_q):
        """Given a set of key-value pairs (x, r) and a query x_q, returns the
        aggregated value r_q. Several attention mechanisms are available:
        - uniform: r_q = mean(r)
        - laplace: r_q = sum_i exp(-||x_i - x_q|| / sigma) * r_i
        - dot_product: r_q = sum_i exp(x_i^T x_q) * r_i
        - multi_head: r_q = sum_i exp(W_q x_i^T W_k x_q) * r_i
        x: (BATCH_SIZE, N, D)
        r: (BATCH_SIZE, N, D)
        x_q: (BATCH_SIZE, D)
        """
        return self._attn_fn(x_q, x, r)
