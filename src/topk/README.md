# Specialized Softmax + TopK4

This directory contains the only production candidate retained from the study.
It targets BF16 router logits on CUDA SM80 when `4 <= E <= 64`, `top_k=4`, and
decode `M <= 64`.

`topk_final_sprint.cu` preserves vLLM's 256-thread CUB reduction and ArgMax
tie-breaking order. It keeps the FP32 softmax row in shared memory and performs
TopK4 plus output writes in the same launch. `bindings.cpp` registers the
standalone CUDA op used by the runtime patch and benchmark scripts.

All unsupported shapes and platforms remain on vLLM's existing path. Earlier
prototypes with non-bitwise softmax reduction order are intentionally omitted.

This implementation is derived from vLLM's Apache-2.0 licensed MoE routing
kernel. See the repository `NOTICE` file.
