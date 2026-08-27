// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/extension.h>

void topk_softmax_a100_small_cuda(
    const torch::Tensor& logits, torch::Tensor weights, torch::Tensor ids,
    torch::Tensor source_rows, bool renormalize);

TORCH_LIBRARY_FRAGMENT(_moe_C, m) {
  m.def(
      "topk_softmax_a100_small(Tensor logits, Tensor! weights, Tensor! ids, "
      "Tensor! source_rows, bool renormalize=False) -> ()");
}

TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("topk_softmax_a100_small", &topk_softmax_a100_small_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax_a100_small", &topk_softmax_a100_small_cuda);
}
