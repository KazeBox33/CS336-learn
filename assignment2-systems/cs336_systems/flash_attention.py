from __future__ import annotations

from typing import Any

import torch

try:  # Mac 没有 Triton，所以使用 try 保留 PyTorch 实现。
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None

Q_TILE_SIZE = 16
K_TILE_SIZE = 16


def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "q, k, and v must have shape "
            "(batch_size, sequence_length, d_model)"
        )

    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")

    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same hidden dimension")

    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same sequence length")

    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")

    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")

    if not q.is_floating_point():
        raise TypeError("q, k, and v must use a floating-point dtype")

    if q.shape[1] == 0 or k.shape[1] == 0 or q.shape[-1] == 0:
        raise ValueError("attention dimensions must be non-empty")


@torch.compile
def _compiled_flash_backward( # torch.compile
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_float = q.to(torch.float32)
    k_float = k.to(torch.float32)
    v_float = v.to(torch.float32)
    output_float = output.to(torch.float32)
    grad_output_float = grad_output.to(torch.float32)

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale
    if is_causal:
        query_positions = torch.arange(q.shape[1], device=q.device)
        key_positions = torch.arange(k.shape[1], device=k.device)
        causal_mask = query_positions[:, None] >= key_positions[None, :]  # pytorch 通过广播把两个tensor拓展
        scores = scores.masked_fill(~causal_mask, -torch.inf)

    probabilities = torch.exp(scores - logsumexp.unsqueeze(-1))
    d_vector = torch.sum(
        output_float * grad_output_float,
        dim=-1,
        keepdim=True,
    )

    grad_v = torch.matmul(probabilities.transpose(-2, -1), grad_output_float) #计算dv
    grad_probabilities = torch.matmul( # 计算 dp
        grad_output_float,
        v_float.transpose(-2, -1),
    )
    grad_scores = probabilities * (grad_probabilities - d_vector)
    grad_q = torch.matmul(grad_scores, k_float) * scale
    grad_k = torch.matmul(grad_scores.transpose(-2, -1), q_float) * scale

    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        _validate_attention_inputs(q, k, v)  # 验证输入。

        if not isinstance(is_causal, bool):
            raise TypeError("is_causal must be a bool")

        batch_size, n_queries, d_model = q.shape
        n_keys = k.shape[1]
        scale = d_model**-0.5

        output = torch.empty_like(q)
        logsumexp = torch.empty(  # 每个 query row 保存一个 LSE。
            (batch_size, n_queries),
            device=q.device,
            dtype=torch.float32,
        )

        for query_start in range(0, n_queries, Q_TILE_SIZE):
            query_end = min(query_start + Q_TILE_SIZE, n_queries)
            query = q[:, query_start:query_end, :].to(torch.float32)
            query_tile_size = query_end - query_start

            output_accumulator = torch.zeros(  # 初始化 o。
                (batch_size, query_tile_size, d_model),
                device=q.device,
                dtype=torch.float32,
            )
            denominator = torch.zeros(  # 对应 l，保存每个 query row 当前累计的分母。
                (batch_size, query_tile_size),
                device=q.device,
                dtype=torch.float32,
            )
            row_maximum = torch.full(  # 对应 m，保存每个 query row 当前见过的最大 score。
                (batch_size, query_tile_size),
                -torch.inf,
                device=q.device,
                dtype=torch.float32,
            )

            for key_start in range(0, n_keys, K_TILE_SIZE):
                key_end = min(key_start + K_TILE_SIZE, n_keys)
                key = k[:, key_start:key_end, :].to(torch.float32)
                value = v[:, key_start:key_end, :].to(torch.float32)

                scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # 计算 score。
                new_row_maximum = torch.maximum(  # 更新最大值。
                    row_maximum,
                    scores.amax(dim=-1),
                )
                correction = torch.exp(row_maximum - new_row_maximum)  # 计算校正因子。
                probabilities = torch.exp(
                    scores - new_row_maximum.unsqueeze(-1)
                )

                denominator = (  # 更新累计分母。
                    correction * denominator
                    + probabilities.sum(dim=-1)
                )
                output_accumulator = (  # 更新 output 的未归一化累加值。
                    correction.unsqueeze(-1) * output_accumulator
                    + torch.matmul(probabilities, value)
                )
                row_maximum = new_row_maximum

            output[:, query_start:query_end, :] = (
                output_accumulator / denominator.unsqueeze(-1)
            ).to(q.dtype)
            logsumexp[:, query_start:query_end] = (
                row_maximum + torch.log(denominator)
            )

        ctx.save_for_backward(logsumexp, q, k, v, output)  # 供反向传播使用。
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(  # 接入了 torch.compile 写的反向传播
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        logsumexp, q, k, v, output = ctx.saved_tensors
        grad_q, grad_k, grad_v = _compiled_flash_backward(
            q,
            k,
            v,
            output,
            grad_output,
            logsumexp,
            ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None  # forward 输入了啥 这边就返回啥的梯度


if triton is not None:

    @triton.jit
    def _flash_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        logsumexp_ptr,
        stride_qb,  # Q 的形状为 (B, Nq, D)。
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        n_queries,
        n_keys,
        scale,
        d_model: tl.constexpr,
        query_tile_size: tl.constexpr,
        key_tile_size: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)  # 确定当前 program 负责的 query tile。
        batch_index = tl.program_id(1)
        query_start = query_tile_index * query_tile_size

        query_block_ptr = tl.make_block_ptr(
            base=q_ptr + batch_index * stride_qb,
            shape=(n_queries, d_model),
            strides=(stride_qq, stride_qd),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d_model),
            order=(1, 0),
        )
        key_block_ptr = tl.make_block_ptr(
            base=k_ptr + batch_index * stride_kb,
            shape=(d_model, n_keys),
            strides=(stride_kd, stride_kk),
            offsets=(0, 0),
            block_shape=(d_model, key_tile_size),
            order=(0, 1),
        )
        value_block_ptr = tl.make_block_ptr(
            base=v_ptr + batch_index * stride_vb,
            shape=(n_keys, d_model),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(key_tile_size, d_model),
            order=(1, 0),
        )
        output_block_ptr = tl.make_block_ptr(
            base=output_ptr + batch_index * stride_ob,
            shape=(n_queries, d_model),
            strides=(stride_oq, stride_od),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d_model),
            order=(1, 0),
        )

        query = tl.load(
            query_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        output_accumulator = tl.zeros(  # o
            (query_tile_size, d_model),
            dtype=tl.float32,
        )
        denominator = tl.zeros((query_tile_size,), dtype=tl.float32)  # l
        row_maximum = tl.full(  # m
            (query_tile_size,),
            -float("inf"),
            dtype=tl.float32,
        )

        query_offsets = query_start + tl.arange(0, query_tile_size)
        key_offsets = tl.arange(0, key_tile_size)

        for key_start in range(0, n_keys, key_tile_size):
            key = tl.load(
                key_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            value = tl.load(
                value_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            scores = tl.dot(query, key) * scale
            valid_keys = key_offsets + key_start < n_keys
            scores = tl.where(valid_keys[None, :], scores, -float("inf"))
            if is_causal:
                causal_mask = query_offsets[:, None] >= (
                    key_offsets[None, :] + key_start
                )
                scores = tl.where(causal_mask, scores, -float("inf"))

            new_row_maximum = tl.maximum(
                row_maximum,
                tl.max(scores, axis=1),
            )
            correction = tl.exp(row_maximum - new_row_maximum)
            probabilities = tl.exp(scores - new_row_maximum[:, None])

            denominator = (
                correction * denominator
                + tl.sum(probabilities, axis=1)
            )
            output_accumulator = (
                correction[:, None] * output_accumulator
                + tl.dot(probabilities.to(value.dtype), value)
            )
            row_maximum = new_row_maximum

            key_block_ptr = key_block_ptr.advance((0, key_tile_size))
            value_block_ptr = value_block_ptr.advance((key_tile_size, 0))

        output = output_accumulator / denominator[:, None]
        tl.store(
            output_block_ptr,
            output,
            boundary_check=(0, 1),
        )

        logsumexp_offsets = (
            batch_index * stride_lb
            + query_offsets * stride_lq
        )
        tl.store(
            logsumexp_ptr + logsumexp_offsets,
            row_maximum + tl.log(denominator),
            mask=query_offsets < n_queries,
        )


    @triton.jit
    def _flash_backward_preprocess_kernel( # 计算 delta
        output_ptr,
        grad_output_ptr,
        delta_ptr,
        stride_ob,
        stride_oq,
        stride_od,
        stride_gob,
        stride_goq,
        stride_god,
        stride_db,
        stride_dq,
        n_queries,
        d_model: tl.constexpr,
        query_tile_size: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_start = query_tile_index * query_tile_size

        output_block_ptr = tl.make_block_ptr(
            base=output_ptr + batch_index * stride_ob,
            shape=(n_queries, d_model),
            strides=(stride_oq, stride_od),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d_model),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            base=grad_output_ptr + batch_index * stride_gob,
            shape=(n_queries, d_model),
            strides=(stride_goq, stride_god),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d_model),
            order=(1, 0),
        )

        output = tl.load(
            output_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)
        grad_output = tl.load(
            grad_output_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)
        delta = tl.sum(output * grad_output, axis=1)

        query_offsets = query_start + tl.arange(0, query_tile_size)
        tl.store(
            delta_ptr
            + batch_index * stride_db
            + query_offsets * stride_dq,
            delta,
            mask=query_offsets < n_queries,
        )


    @triton.jit
    def _flash_backward_dkdv_kernel( # 计算 dk dv
        q_ptr,
        k_ptr,
        v_ptr,
        grad_output_ptr,
        logsumexp_ptr,
        delta_ptr,
        grad_k_ptr,
        grad_v_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_gob,
        stride_goq,
        stride_god,
        stride_lb,
        stride_lq,
        stride_db,
        stride_dq,
        stride_gkb,
        stride_gkk,
        stride_gkd,
        stride_gvb,
        stride_gvk,
        stride_gvd,
        n_queries,
        n_keys,
        scale,
        d_model: tl.constexpr,
        query_tile_size: tl.constexpr,
        key_tile_size: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        key_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        key_start = key_tile_index * key_tile_size

        key_transpose_block_ptr = tl.make_block_ptr(
            base=k_ptr + batch_index * stride_kb,
            shape=(d_model, n_keys),
            strides=(stride_kd, stride_kk),
            offsets=(0, key_start),
            block_shape=(d_model, key_tile_size),
            order=(0, 1),
        )
        value_transpose_block_ptr = tl.make_block_ptr(
            base=v_ptr + batch_index * stride_vb,
            shape=(d_model, n_keys),
            strides=(stride_vd, stride_vk),
            offsets=(0, key_start),
            block_shape=(d_model, key_tile_size),
            order=(0, 1),
        )
        query_block_ptr = tl.make_block_ptr(
            base=q_ptr + batch_index * stride_qb,
            shape=(n_queries, d_model),
            strides=(stride_qq, stride_qd),
            offsets=(0, 0),
            block_shape=(query_tile_size, d_model),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            base=grad_output_ptr + batch_index * stride_gob,
            shape=(n_queries, d_model),
            strides=(stride_goq, stride_god),
            offsets=(0, 0),
            block_shape=(query_tile_size, d_model),
            order=(1, 0),
        )
        grad_k_block_ptr = tl.make_block_ptr(
            base=grad_k_ptr + batch_index * stride_gkb,
            shape=(n_keys, d_model),
            strides=(stride_gkk, stride_gkd),
            offsets=(key_start, 0),
            block_shape=(key_tile_size, d_model),
            order=(1, 0),
        )
        grad_v_block_ptr = tl.make_block_ptr(
            base=grad_v_ptr + batch_index * stride_gvb,
            shape=(n_keys, d_model),
            strides=(stride_gvk, stride_gvd),
            offsets=(key_start, 0),
            block_shape=(key_tile_size, d_model),
            order=(1, 0),
        )

        key_transpose = tl.load(
            key_transpose_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        value_transpose = tl.load(
            value_transpose_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        grad_k_accumulator = tl.zeros(
            (key_tile_size, d_model),
            dtype=tl.float32,
        )
        grad_v_accumulator = tl.zeros(
            (key_tile_size, d_model),
            dtype=tl.float32,
        )

        key_offsets = key_start + tl.arange(0, key_tile_size)
        valid_keys = key_offsets < n_keys

        for query_start in range(0, n_queries, query_tile_size):
            query = tl.load(
                query_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            grad_output = tl.load(
                grad_output_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            query_offsets = query_start + tl.arange(0, query_tile_size)
            valid_queries = query_offsets < n_queries
            logsumexp = tl.load(
                logsumexp_ptr
                + batch_index * stride_lb
                + query_offsets * stride_lq,
                mask=valid_queries,
                other=0.0,
            )
            delta = tl.load(
                delta_ptr
                + batch_index * stride_db
                + query_offsets * stride_dq,
                mask=valid_queries,
                other=0.0,
            )

            scores = tl.dot(query, key_transpose) * scale
            valid_scores = valid_queries[:, None] & valid_keys[None, :]
            if is_causal:
                valid_scores &= query_offsets[:, None] >= key_offsets[None, :]
            scores = tl.where(valid_scores, scores, -float("inf"))

            probabilities = tl.exp(scores - logsumexp[:, None])
            probabilities_for_dot = probabilities.to(grad_output.dtype)
            grad_v_accumulator = tl.dot(
                tl.trans(probabilities_for_dot),
                grad_output,
                acc=grad_v_accumulator,
            )

            grad_probabilities = tl.dot(
                grad_output,
                value_transpose,
            )
            grad_scores = probabilities * (
                grad_probabilities - delta[:, None]
            )
            grad_scores_for_dot = grad_scores.to(query.dtype)
            grad_k_accumulator = tl.dot(
                tl.trans(grad_scores_for_dot),
                query,
                acc=grad_k_accumulator,
            )

            query_block_ptr = query_block_ptr.advance((query_tile_size, 0))
            grad_output_block_ptr = grad_output_block_ptr.advance(
                (query_tile_size, 0)
            )

        tl.store(
            grad_k_block_ptr,
            (grad_k_accumulator * scale).to(key_transpose.dtype),
            boundary_check=(0, 1),
        )
        tl.store(
            grad_v_block_ptr,
            grad_v_accumulator.to(value_transpose.dtype),
            boundary_check=(0, 1),
        )


    @triton.jit
    def _flash_backward_dq_kernel(  #计算 dQ
        q_ptr,
        k_ptr,
        v_ptr,
        grad_output_ptr,
        logsumexp_ptr,
        delta_ptr,
        grad_q_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_gob,
        stride_goq,
        stride_god,
        stride_lb,
        stride_lq,
        stride_db,
        stride_dq,
        stride_gqb,
        stride_gqq,
        stride_gqd,
        n_queries,
        n_keys,
        scale,
        d_model: tl.constexpr,
        query_tile_size: tl.constexpr,
        key_tile_size: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_start = query_tile_index * query_tile_size

        query_block_ptr = tl.make_block_ptr(
            base=q_ptr + batch_index * stride_qb,
            shape=(n_queries, d_model),
            strides=(stride_qq, stride_qd),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d_model),
            order=(1, 0),
        )
        grad_output_block_ptr = tl.make_block_ptr(
            base=grad_output_ptr + batch_index * stride_gob,
            shape=(n_queries, d_model),
            strides=(stride_goq, stride_god),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d_model),
            order=(1, 0),
        )
        key_transpose_block_ptr = tl.make_block_ptr(
            base=k_ptr + batch_index * stride_kb,
            shape=(d_model, n_keys),
            strides=(stride_kd, stride_kk),
            offsets=(0, 0),
            block_shape=(d_model, key_tile_size),
            order=(0, 1),
        )
        value_transpose_block_ptr = tl.make_block_ptr(
            base=v_ptr + batch_index * stride_vb,
            shape=(d_model, n_keys),
            strides=(stride_vd, stride_vk),
            offsets=(0, 0),
            block_shape=(d_model, key_tile_size),
            order=(0, 1),
        )
        grad_q_block_ptr = tl.make_block_ptr(
            base=grad_q_ptr + batch_index * stride_gqb,
            shape=(n_queries, d_model),
            strides=(stride_gqq, stride_gqd),
            offsets=(query_start, 0),
            block_shape=(query_tile_size, d_model),
            order=(1, 0),
        )

        query = tl.load(
            query_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        grad_output = tl.load(
            grad_output_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )

        query_offsets = query_start + tl.arange(0, query_tile_size)
        valid_queries = query_offsets < n_queries
        logsumexp = tl.load(
            logsumexp_ptr
            + batch_index * stride_lb
            + query_offsets * stride_lq,
            mask=valid_queries,
            other=0.0,
        )
        delta = tl.load(
            delta_ptr
            + batch_index * stride_db
            + query_offsets * stride_dq,
            mask=valid_queries,
            other=0.0,
        )
        grad_q_accumulator = tl.zeros(
            (query_tile_size, d_model),
            dtype=tl.float32,
        )

        key_offsets = tl.arange(0, key_tile_size)
        for key_start in range(0, n_keys, key_tile_size):
            key_transpose = tl.load(
                key_transpose_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            value_transpose = tl.load(
                value_transpose_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )

            current_key_offsets = key_start + key_offsets
            valid_keys = current_key_offsets < n_keys
            scores = tl.dot(query, key_transpose) * scale
            valid_scores = valid_queries[:, None] & valid_keys[None, :]
            if is_causal:
                valid_scores &= (
                    query_offsets[:, None]
                    >= current_key_offsets[None, :]
                )
            scores = tl.where(valid_scores, scores, -float("inf"))

            probabilities = tl.exp(scores - logsumexp[:, None])
            grad_probabilities = tl.dot(
                grad_output,
                value_transpose,
            )
            grad_scores = probabilities * (
                grad_probabilities - delta[:, None]
            )
            grad_q_accumulator = tl.dot(
                grad_scores.to(key_transpose.dtype),
                tl.trans(key_transpose),
                acc=grad_q_accumulator,
            )

            key_transpose_block_ptr = key_transpose_block_ptr.advance(
                (0, key_tile_size)
            )
            value_transpose_block_ptr = value_transpose_block_ptr.advance(
                (0, key_tile_size)
            )

        tl.store(
            grad_q_block_ptr,
            (grad_q_accumulator * scale).to(query.dtype),
            boundary_check=(0, 1),
        )

else:
    _flash_forward_kernel = None
    _flash_backward_preprocess_kernel = None
    _flash_backward_dkdv_kernel = None
    _flash_backward_dq_kernel = None


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        _validate_attention_inputs(q, k, v)

        if triton is None or _flash_forward_kernel is None:
            raise RuntimeError(
                "Triton FlashAttention requires Triton on a supported Linux system"
            )
        if not isinstance(is_causal, bool):
            raise TypeError("is_causal must be a bool")
        if q.device.type != "cuda":
            raise ValueError("Triton FlashAttention requires CUDA tensors")

        batch_size, n_queries, d_model = q.shape
        n_keys = k.shape[1]
        if d_model < 16 or not triton.next_power_of_2(d_model) == d_model:
            raise ValueError("d_model must be a power of two and at least 16")

        output = torch.empty_like(q)
        logsumexp = torch.empty(
            (batch_size, n_queries),
            device=q.device,
            dtype=torch.float32,
        )
        launch_grid = (
            triton.cdiv(n_queries, Q_TILE_SIZE),
            batch_size,
        )

        with torch.cuda.device(q.device):
            _flash_forward_kernel[launch_grid](
                q,
                k,
                v,
                output,
                logsumexp,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *output.stride(),
                *logsumexp.stride(),
                n_queries,
                n_keys,
                d_model**-0.5,
                d_model=d_model,
                query_tile_size=Q_TILE_SIZE,
                key_tile_size=K_TILE_SIZE,
                is_causal=is_causal,
            )

        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        logsumexp, q, k, v, output = ctx.saved_tensors
        if (
            triton is None
            or _flash_backward_preprocess_kernel is None
            or _flash_backward_dkdv_kernel is None
            or _flash_backward_dq_kernel is None
        ):
            raise RuntimeError(
                "Triton FlashAttention backward requires Triton on CUDA"
            )

        batch_size, n_queries, d_model = q.shape
        n_keys = k.shape[1]
        grad_output = grad_output.contiguous()
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        delta = torch.empty(
            (batch_size, n_queries),
            device=q.device,
            dtype=torch.float32,
        )

        preprocess_grid = (
            triton.cdiv(n_queries, Q_TILE_SIZE),
            batch_size,
        )
        dkdv_grid = (
            triton.cdiv(n_keys, K_TILE_SIZE),
            batch_size,
        )
        dq_grid = (
            triton.cdiv(n_queries, Q_TILE_SIZE),
            batch_size,
        )

        with torch.cuda.device(q.device):
            _flash_backward_preprocess_kernel[preprocess_grid](
                output,
                grad_output,
                delta,
                *output.stride(),
                *grad_output.stride(),
                *delta.stride(),
                n_queries,
                d_model=d_model,
                query_tile_size=Q_TILE_SIZE,
                num_warps=4,
            )
            _flash_backward_dkdv_kernel[dkdv_grid](
                q,
                k,
                v,
                grad_output,
                logsumexp,
                delta,
                grad_k,
                grad_v,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *grad_output.stride(),
                *logsumexp.stride(),
                *delta.stride(),
                *grad_k.stride(),
                *grad_v.stride(),
                n_queries,
                n_keys,
                d_model**-0.5,
                d_model=d_model,
                query_tile_size=Q_TILE_SIZE,
                key_tile_size=K_TILE_SIZE,
                is_causal=ctx.is_causal,
                num_warps=4,
            )
            _flash_backward_dq_kernel[dq_grid](
                q,
                k,
                v,
                grad_output,
                logsumexp,
                delta,
                grad_q,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *grad_output.stride(),
                *logsumexp.stride(),
                *delta.stride(),
                *grad_q.stride(),
                n_queries,
                n_keys,
                d_model**-0.5,
                d_model=d_model,
                query_tile_size=Q_TILE_SIZE,
                key_tile_size=K_TILE_SIZE,
                is_causal=ctx.is_causal,
                num_warps=4,
            )

        return grad_q, grad_k, grad_v, None
