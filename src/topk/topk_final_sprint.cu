/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * This kernel is an experimental specialization derived from vLLM's
 * topk_softmax implementation. See NOTICE for upstream attribution.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <torch/extension.h>

#include <cfloat>

namespace {

using CubAddOp = cuda::std::plus<>;
using CubMaxOp = cuda::maximum<>;
constexpr int kTopK = 4;

// Preserve the reference 256-thread CUB reduction and tie-breaking order,
// while keeping the softmax row in shared memory and performing TopK4 in the
// same launch. This removes the global FP32 workspace and the second launch.
__global__ __launch_bounds__(256) void topk_bf16_exact_kernel(
    const __nv_bfloat16* logits, float* weights, int32_t* ids,
    int32_t* source_rows, int rows, int experts, bool renormalize) {
  using FloatReduce = cub::BlockReduce<float, 256>;
  using Kvp = cub::KeyValuePair<int, float>;
  using KvpReduce = cub::BlockReduce<Kvp, 256>;
  __shared__ typename FloatReduce::TempStorage float_storage;
  __shared__ typename KvpReduce::TempStorage kvp_storage;
  __shared__ float row_max;
  __shared__ float inverse_sum;
  __shared__ float probabilities[64];

  const int row = blockIdx.x;
  float thread_data = -FLT_MAX;
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    thread_data = max(
        __bfloat162float(logits[row * experts + expert]), thread_data);
  }
  const float max_value =
      FloatReduce(float_storage).Reduce(thread_data, CubMaxOp());
  if (threadIdx.x == 0) row_max = max_value;
  __syncthreads();

  thread_data = 0.0f;
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    thread_data +=
        expf(__bfloat162float(logits[row * experts + expert]) - row_max);
  }
  const float sum = FloatReduce(float_storage).Reduce(thread_data, CubAddOp());
  if (threadIdx.x == 0) inverse_sum = 1.0f / sum;
  __syncthreads();

  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    float probability =
        expf(__bfloat162float(logits[row * experts + expert]) - row_max) *
        inverse_sum;
    // Match the reference behavior for invalid padded graph rows.
    if (isnan(probability) || isinf(probability)) probability = 0.0f;
    probabilities[expert] = probability;
  }
  __syncthreads();

  cub::ArgMax arg_max;
  float selected_sum = 0.0f;
#pragma unroll
  for (int rank = 0; rank < kTopK; ++rank) {
    Kvp thread_kvp(0, -1.0f);
    for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
      Kvp candidate(expert, probabilities[expert]);
#pragma unroll
      for (int prior = 0; prior < rank; ++prior) {
        if (ids[row * kTopK + prior] == expert) candidate = thread_kvp;
      }
      thread_kvp = arg_max(candidate, thread_kvp);
    }
    const Kvp result = KvpReduce(kvp_storage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int out = row * kTopK + rank;
      weights[out] = probabilities[result.key];
      ids[out] = result.key;
      source_rows[out] = rank * rows + row;
      if (renormalize) selected_sum += probabilities[result.key];
    }
    __syncthreads();
  }

  if (renormalize && threadIdx.x == 0) {
    const float denominator = selected_sum > 0.0f ? selected_sum : 1.0f;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
      weights[row * kTopK + rank] /= denominator;
    }
  }
}

}  // namespace

void topk_softmax_a100_small_cuda(
    const torch::Tensor& logits, torch::Tensor weights, torch::Tensor ids,
    torch::Tensor source_rows, bool renormalize) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kBFloat16,
              "topk_softmax_a100_small requires CUDA BF16 logits");
  TORCH_CHECK(logits.dim() == 2 && logits.size(1) >= kTopK &&
                  logits.size(1) <= 64 && logits.size(0) > 0 &&
                  logits.size(0) <= 64,
              "expected [M,E], 0<M<=64 and 4<=E<=64");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32 &&
                  ids.scalar_type() == torch::kInt32 &&
                  source_rows.scalar_type() == torch::kInt32,
              "output dtype mismatch");
  TORCH_CHECK(weights.is_contiguous() && ids.is_contiguous() &&
                  source_rows.is_contiguous() && logits.is_contiguous(),
              "all tensors must be contiguous");

  const c10::cuda::CUDAGuard guard(logits.device());
  const auto stream = at::cuda::getCurrentCUDAStream(logits.get_device());
  const int rows = logits.size(0);
  topk_bf16_exact_kernel<<<rows, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(logits.const_data_ptr()),
      weights.data_ptr<float>(), ids.data_ptr<int32_t>(),
      source_rows.data_ptr<int32_t>(), rows, logits.size(1), renormalize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
