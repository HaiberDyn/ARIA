<!-- ARIA module: ARIA-MASTER.md — do not edit directly; edit this file in spec/modules/ -->
# ARIA — Agnostic Runtime Inference Abstraction
## Specification 0.7-draft

**Status**: Draft
**Authors**: HaiberDyn
**Repository**: https://github.com/HaiberDyn/ARIA
**Version**: 0.7-draft (2026-03-16)
**License**: Spec text: CC BY-SA 4.0 · Reference code: Apache 2.0  
**IP Policy**: See `CONTRIBUTING.md` — patent non-assertion pledge, DCO contribution terms  
**Governance**: See `CONTRIBUTING.md` — working group structure, how to join  
**Conformance**: See `CONFORMANCE.md` — model and runtime requirements, portability levels

**Keywords**: The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are
to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

---

## Abstract

ARIA defines a minimal, architecture-agnostic contract between a language model implementation
and a serving runtime. It specifies two interfaces: `ILanguageModel` — the stateful inference
contract (`model_info`, `declare_state`, `prefill`, `decode`) — and `ITokenizer` — the
text↔token seam the model ships with and the runtime may optimize. The central design
principle: a model *declares* its state requirements — shape, dtype, semantic role — and the
runtime *allocates* those resources opaquely. Neither side exposes internals to the other.

This separation enables model architectures, serving runtimes, and hardware backends to evolve
independently. Adding a new architecture (attention → SSM → hybrid → future) requires no
framework modification. Switching runtimes requires no model modification. Operators compose
freely.

---

## 1. Motivation

### 1.1 The Coupling Problem

Every new LLM architecture generation forces code changes in every serving framework:

```
GPT → LLaMA       requires: grouped-query attention support in KV cache allocator
LLaMA → Mamba     requires: new SSM state manager, constant-length state contract
Mamba → Hybrid    requires: per-layer heterogeneous state (KV + SSM simultaneously)
Dense → MoE       requires: new expert routing in model runner, skip logic in batcher
Qwen3 → Qwen3.5   requires: updated attention patterns and memory profiling again
```

Root cause: frameworks (SGLang, vLLM, llama.cpp, TGI, LMDeploy) are coupled to model internals.
They know about attention head counts, KV cache shapes, expert routing tables — implementation
details they have no architectural reason to know.

This violates four SOLID principles simultaneously:

| Principle | Violation |
|-----------|-----------|
| **Open/Closed** | Every new architecture requires modifying framework internals |
| **Liskov Substitution** | Models are not substitutable — each requires bespoke runtime paths |
| **Interface Segregation** | Runtime consumes model internals it never needs to reason about |
| **Dependency Inversion** | Framework depends on model concretions, not stable abstractions |

It also violates DRY: KV cache management, batching logic, and state scheduling are reimplemented
per architecture family in every framework.

### 1.2 What ARIA Adds to the Mix

Each existing specification solves a real problem well. None address the **stateful
autoregressive decode loop with heterogeneous state types** — not because they got it
wrong, but because it wasn't their problem to solve. ARIA is additive.

| Spec | What it does well | What ARIA adds |
|------|------------------|----------------|
| ONNX | Static graph portability, cross-framework operator coverage | Stateful per-request decode loop; dynamic control flow (MoE routing, recurrent state) |
| WASI-NN | Hardware-agnostic inference backend; WebAssembly portability | Per-request state lifecycle; heterogeneous state type declaration and negotiation |
| GGUF | Single-file model distribution with weights, metadata, tokenizer | The serving-time contract: how a runtime calls into the model at inference time |
| Triton Inference Server backend API | Strong serving contract, production-proven | Standardized stateful decode; model-declared state layout instead of per-backend bespoke KV management |
| HuggingFace `generate()` | Broad model support, sampling library, Python ecosystem | Language-agnostic contract; hides model internals; usable outside Python |

The universal gap none of them fill: no existing spec defines a standard where the model
**declares** its state requirements and the runtime **allocates** them opaquely. For
pure-attention models this is a convenience. For hybrid attention+SSM models it is the
only portable approach.

### 1.3 ARIA's Solution

```
interface ILanguageModel {
    model_info()     -> Result<ModelInfo, ARIAError>
    declare_state(max_batch: uint32, max_tokens: uint32) -> Result<StateSpec, ARIAError>
    prefill(tokens: TokenIds, state: StateHandle, out_logits: Option<TensorView> = None) -> Result<void, ARIAError>
    prefill(batch: List<PrefillRequest>) -> Result<List<Result<void, ARIAError>>, ARIAError>
    decode(token: TokenId, state: StateHandle, out_logits: Option<TensorView> = None) -> Result<void, ARIAError>
    decode(batch: List<DecodeRequest>) -> Result<List<Result<void, ARIAError>>, ARIAError>
}
```

The model tells the runtime what state it needs. The runtime allocates it, owns the handle
lifecycle, and passes handles to the model at call time. The model reads and writes through
the handle. Neither party looks inside the other.

---

## 2. Scope

### 2.1 In Scope — ARIA 0.6-draft

- Decoder-only autoregressive language models
- Discrete token vocabulary (TokenId = uint32, as defined in §4.1)
- Synchronous, per-request generation
- Heterogeneous state types: attention KV cache, recurrent SSM state, and model-defined Custom state
- Basic capability negotiation: ARIA version, vocab size, supported KV dtypes
- Single-node inference (no cross-node communication protocol)

### 2.2 Explicitly Out of Scope — ARIA 0.7-draft

The following are NOT ARIA's concern. Each layer owns its domain:

| Concern | Owner |
|---------|-------|
| Model file format | SafeTensors, GGUF, ONNX |
| Weight quantization format | Model card, nvidia-modelopt, GGUF quant |
| Tokenization | HuggingFace tokenizers, tiktoken, sentencepiece |
| Sampling strategy (temperature, top-p, top-k, beam) | Runtime — above ARIA |
| Consumer API (chat completions, streaming) | OpenAI-compatible layer — above ARIA |
| Hardware dispatch (CUDA kernels, Metal, CPU SIMD) | Model implementation — below ARIA |
| GPU batching and scheduling | Runtime — below ARIA |
| Tensor parallelism / pipeline parallelism | Model implementation — internal |
| LoRA / adapter weight management | Out of scope 0.6 (see §15) |
| Encoder-decoder models | Out of scope 0.6 (see §15) |
| Multimodal embedding inputs | Out of scope 0.6 (see §15) |
| Async execution / streaming protocol | Runtime implementation detail |
| CUDA graph capture | Model implementation detail |

**Design principle**: ARIA does not prevent innovation in any of these areas. It creates a stable
boundary so that innovation in one layer does not require changes in another. See §11.

**Scope extension schedule**: Future versions will expand this list systematically.

| Extension | Target | Notes |
|-----------|--------|-------|
| Encoder-decoder models, `CrossAttentionKV` | ARIA 1.1 | H2 2026 |
| LoRA / adapter interface | ARIA 1.1 | H2 2026 |
| Multimodal embedding inputs (vision, audio) | ARIA 2.0 | 2027 |
| Distributed multi-node inference contract | ARIA 2.0 | 2027 |

These timelines require multi-stakeholder review and at least two independent
implementations before promotion. See `ROADMAP.md` for details.

---

## 3. Definitions

| Term | Definition |
|------|-----------|
| **Model** | An implementation of `ILanguageModel` — weights plus compute logic |
| **Runtime** | A serving system (vLLM, SGLang, llama.cpp, etc.) that calls `ILanguageModel` to generate tokens |
| **Harness** | Synonym for **Runtime**. The component responsible for orchestrating model execution and state management. |
| **State** | All mutable per-request data required to produce correct next-token logits |
| **StateHandle** | An opaque reference to runtime-allocated state storage for one request |
| **Slot** | One contiguous allocation described by a `StateSlot`; accessed via StateHandle |
| **Consumer** | A client of the runtime's serving API (e.g., an OpenAI-compatible endpoint caller) |
| **Binding** | A language-specific expression of the ARIA interface (Python, C, Rust, etc.) |

---

## 3.1 Implementation Principles

ARIA is designed around compiled, type-safe runtimes. The following principles are design
expectations; they are not mandates on implementation language choice, but implementations
that do not uphold them forfeit the safety guarantees the design depends on.

**Encapsulation.** `StateHandle` is opaque. The consumer has no access to slot memory, handle
internals, or model state outside of the defined interface methods. `ModelInfo` MUST be treated
as immutable by the consumer; the model MUST return identical values on repeated calls.

**Immutability at boundaries.** `TensorView` is consumer-immutable: the consumer MUST NOT write
into model-owned tensor memory. The model MUST NOT retain a `TensorView` reference beyond the
scope of the call that produced it.

**Type safety.** The interface contract is expressed in terms of typed structures and enums.
Implementations SHOULD enforce this contract at compile time. Runtime type errors are a
conformance failure, not a recoverable condition.

**Separation of concerns.** The model owns inference. The runtime owns memory management,
scheduling, and hardware dispatch. Neither side should reach across this boundary.

Implementations in interpreted or dynamically-typed languages can conform to the letter of
this specification, but compile-time enforcement of the contract is unavailable. The design
rewards implementations that take full advantage of type-safe, compiled runtimes.

---
