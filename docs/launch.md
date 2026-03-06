# Why ARIA? The Case for an Agnostic Runtime Inference Abstraction

**Date:** March 6, 2026
**Status:** ARIA 0.6-draft (Public Request for Comment)

---

The pace of innovation in Large Language Models (LLMs) is relentless. But there is a hidden bottleneck slowing us down: **The Runtime Fragmentation Problem.**

Every time a researcher invents a new architecture—Mamba, specific sparse attention patterns, or a novel MoE gating mechanism—they face a daunting reality. To get their model adopted, it needs to be implemented not just in PyTorch, but in high-performance inference runtimes: vLLM, llama.cpp, TGI, SGLang, and others.

Each runtime has its own internal abstractions for memory, KV caching, and tensor operations. A kernel written for vLLM doesn't work in llama.cpp. This creates an **O(M × N)** compatibility matrix, where *M* is the number of model architectures and *N* is the number of runtimes.

This fragmentation has two costs:
1.  **Lag**: New models take weeks or months to be supported in production runtimes.
2.  **Lock-in**: Optimization efforts are fractured across incompatible backends.

We need a standard interface. We need **ARIA**.

## What is ARIA?

**ARIA (Agnostic Runtime Inference Abstraction)** is a low-level specification that defines how a model's inference logic communicates with a runtime environment. It is not a framework; it is a **standard ABI**.

It allows a model author to write their inference implementation **once**—using standard C functions for tensor views, memory management, and synchronization—and have that single implementation run on any compliant runtime.

### Key Technical Features

1.  **Language Agnostic ABI**: ARIA is defined by a standard binary interface (ABI) that can be implemented in any language—Rust, C++, C#, Python, Go, or Zig. It imposes no specific framework dependencies, ensuring true interoperability across the stack.
2.  **Zero-Copy Tensor Views**: The spec defines a standard `AriaTensorView` for passing strided tensor data between the runtime and the model, supporting generic data types (FP16, BF16, FP8, Int4, NVFP4).
3.  **Semantic Tag Registry**: Instead of hardcoding "FlashAttention" or "MambaState" into the spec, ARIA uses a semantic tagging system. Models request state (e.g., `AttentionKV`) by tag. Runtimes provide opaque handles to that state. This allows the spec to evolve without breaking ABI.
4.  **Hardware Agnostic**: While focused on GPU/accelerator inference, the abstractions (queues, events, memory pools) map cleanly to CPU, CUDA, ROCm, and specialized AI accelerators.

## How It Works

In the ARIA model, the **Runtime** owns the hardware and memory. The **Model** owns the math.

```c
// Simplified Example: A model requests a KV cache
aria_status_t status = runtime->get_state(
    ctx, 
    "AttentionKV", // Semantic Tag
    &kv_cache_handle
);

// The model dispatches work to the runtime's stream
aria_launch_kernel(
    stream,
    my_custom_attention_kernel,
    input_tensor,
    kv_cache_handle
);
```

The runtime doesn't need to know *how* the attention kernel works. It just ensures the memory is ready and the stream executes the work.

## Why 0.6-draft?

We are publishing ARIA as a **0.6-draft** to signal that while the core ABI is stable enough for prototype implementation, we are actively seeking feedback from runtime maintainers and model architects.

We specifically need input on:
- **The Tensor View ABI**: Is it flexible enough for quantized formats?
- **The Extension Mechanism**: Does the tag registry cover your use cases?
- **Safety**: Are the memory ownership rules clear enough?

## Get Involved

ARIA is an open standard. The repository is live at [github.com/HaiberDyn/ARIA](https://github.com/HaiberDyn/ARIA).

- **Read the Spec**: `spec/ARIA-0.6-draft.md`
- **Propose a Tag**: See `spec/EXTENSIONS.md`
- **Join the Discussion**: [GitHub Discussions](https://github.com/HaiberDyn/ARIA/discussions)

Let's break the O(M × N) matrix. Let's make inference universal.
