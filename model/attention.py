#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Attention blocks.
"""


import math
from inspect import isfunction

import numpy as np
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
        output_dim: int,
        n_heads: int = 8,
        use_bias: bool = False,
        normalize: bool = True,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self._normalize = normalize
        head_dim_out = output_dim // n_heads
        self.scale = torch.sqrt(torch.tensor(output_dim))
        self._heads_k = torch.nn.ModuleList(
            [
                torch.nn.Linear(k_dim, head_dim_out, bias=use_bias)
                for _ in range(n_heads)
            ]
        )
        self._heads_q = torch.nn.ModuleList(
            [
                torch.nn.Linear(q_dim, head_dim_out, bias=use_bias)
                for _ in range(n_heads)
            ]
        )
        self._heads_v = torch.nn.ModuleList(
            [
                torch.nn.Linear(v_dim, head_dim_out, bias=use_bias)
                for _ in range(n_heads)
            ]
        )
        self.out_proj = torch.nn.Sequential(
            torch.nn.Linear(n_heads * head_dim_out, output_dim, bias=use_bias),
            torch.nn.Dropout(p_dropout),
        )

    def _attention(self, q, k, v):
        """Computes the (normalized) scaled dot product attention.
        Args:
            q: (batch_size, C, T)
            k: (batch_size, C, T)
            v: (batch_size, C, T)
        Returns:
            (batch_size, C, D * H * W)
        """
        print("Attention")
        print("Q: ", q.shape)
        print("K: ", k.shape)
        print("V: ", v.shape)
        # torch.einsum allows to compute matrix multiplications and other multi-dimensional linear
        # algebra operations using Einstein's notation. It is super convenient to compute
        # dot-products with multiple dimensions of batches.
        dot_p = torch.einsum("bct,bcs->bts", q, k)
        print("Weights dot-p: ", dot_p.shape)
        scaled_dotp = dot_p / self.scale
        if self._normalize:
            scaled_dotp = torch.nn.functional.softmax(scaled_dotp, dim=-1)
        else:
            scaled_dotp = torch.sigmoid(scaled_dotp)
        output = torch.einsum("bts,bcs->bct", scaled_dotp, v)
        print("Scaled dot product: ", output.shape)
        return output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        Args:
            q: (batch_size, C, D * W * H)
            k: (batch_size, K, A * B * C)
            v: (batch_size, K, A * B * C)
        Returns:
            (batch_size, C, D * H * W)
        """

        output = []
        print("Q: ", q.shape)
        print("K: ", k.shape)
        print("V: ", v.shape)
        bs, q, *spatial = q.shape
        _, k, *_ = k.shape
        _, v, *_ = v.shape
        q_ = q.view(bs, q, -1)
        k_ = k.view(bs, k, -1)
        v_ = v.view(bs, v, -1)
        print("Q: ", q_.shape)
        print("K: ", k_.shape)
        print("V: ", v_.shape)

        for h in range(len(self._heads_k)):
            print("Head: ", h)
            print("Head Q: ", self._heads_q[h](q_).shape)
            print("Head K: ", self._heads_k[h](k_).shape)
            print("Head V: ", self._heads_v[h](v_).shape)
            output.append(
                self._attention(
                    self._heads_q[h](q_), self._heads_k[h](k_), self._heads_v[h](v_)
                )
            )
            print("Output: ", output[-1].shape)
        output = torch.cat(output, dim=-1)
        print(output.shape)
        print(self.out_proj(output).shape)
        return self.out_proj(output).view(bs, -1, *spatial)


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        n_heads: int = 8,
        use_bias: bool = False,
        normalize: bool = True,
    ):
        """
        Args:
            kq_dim_in: Dimension of the input to the key and query matrices.
            v_dim_in: Dimension of the input to the value matrix.
            output_dim: Dimension of the output of the attention layer.
            n_layers: Number of layers of attention to stack.
            n_heads: Number of heads to use in the attention layer.
            use_bias: Whether to use a bias in the attention layer.
            normalize: Whether to use softmax or sigmoid in the attention layer.
        """
        super().__init__()
        self._layers = torch.nn.ModuleList(
            [
                MultiHeadAttention(
                    n_heads=n_heads,
                    kq_dim_in=input_dim,
                    kq_dim_out=input_dim,
                    v_dim=input_dim,
                    v_dim_out=input_dim,
                    use_bias=use_bias,
                    normalize=normalize,
                )
                for _ in range(n_layers)
            ]
        )
        self._resize_output = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, ctx_x: torch.Tensor, ctx_y: torch.Tensor) -> torch.Tensor:
        """
        The input to MSA should be as such: (N, D) with N input vectors of dim D,  such that D is
        divided by M (n_heads). For low dims of input vectors, it doesn't make much sense to use
        self attention...

        For my problem, and in general for the ANP, we can see N as the number of context pairs,
        and D as the total pair dimensionality.
        """
        context = torch.concat((ctx_x, ctx_y), dim=-1)  # Concat x and y
        output = context
        for layer in self._layers:
            output = layer(output, output, output)
        return self._resize_output(output)


""" From here on, this is code taken from OpenAI's implementation and slightly adapted/simplified for our needs.
"""


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return torch.nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return torch.nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return torch.nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class AttentionBlock(torch.nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = torch.nn.GroupNorm(16, channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), True
        )  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(torch.nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(torch.nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(torch.nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim), torch.nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)
