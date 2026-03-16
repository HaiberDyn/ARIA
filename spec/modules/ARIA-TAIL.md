<!-- ARIA module: ARIA-TAIL.md — do not edit directly; edit this file in spec/modules/ -->
## 12. Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Consumer Layer                                 │
│              OpenAI-compatible API (chat, completions)               │
│           Streaming, cancellation, rate limiting, auth               │
│    ← Runtime is FREE to innovate: SLO-aware scheduling, autoscaling  │
├─────────────────────────────────────────────────────────────────────┤
│                       Runtime Layer                                   │
│   SGLang · vLLM · llama.cpp · TGI · LMDeploy · future runtimes      │
│   Batching · scheduling · paging · prefix caching · KV eviction      │
│   Sampling · guided decoding · speculative decoding orchestration     │
│    ← Runtime is FREE to innovate: new schedulers, prefill strategies  │
├═════════════════════════════════════════════════════════════════════╡
║               ── ARIA CONTRACT (this specification) ──               ║
║           model_info · declare_state · prefill · decode              ║
║                    ITokenizer (text ↔ token seam)                    ║
╞═════════════════════════════════════════════════════════════════════╡
│                        Model Layer                                    │
│   Qwen3 · LLaMA · Mamba · Jamba · Mixtral · future architectures     │
│   Weights · compute kernels · internal batching · architecture logic  │
│    ← Model is FREE to innovate: new architectures, new state types    │
├─────────────────────────────────────────────────────────────────────┤
│                       Hardware Layer                                  │
│   NVIDIA CUTLASS · Triton · FlashInfer · Apple Metal · CPU SIMD      │
│   CUDA / ROCm / OneAPI / NPU compilers / custom accelerators          │
│    ← Hardware is FREE to innovate: new dtypes, new memory models      │
└─────────────────────────────────────────────────────────────────────┘
```

The ARIA contract (double line) is the single boundary that all parties agree on.
Every layer above and below it is **completely unconstrained** by the spec.
ARIA standardizes exactly the minimum surface required for interoperability — nothing more.

**For operators**: concerns above the ARIA line (consumer API, streaming, multi-tenancy,
auth, scheduling) remain the runtime's domain — competitive innovation space, not
standardized. ARIA operates below that layer.

---

## 13. Design Rationale

### 13.1 Four Methods: One Metadata Query, Three Generation Primitives

`model_info()` is a load-time metadata query, not part of the generation loop.
`declare_state / prefill / decode` IS the complete autoregressive generation loop:
declare what you need → populate state from prompt → generate one token at a time.

There is no fourth generation primitive. Everything else (sampling, beam search, speculative
decoding, batching) is policy applied above or below this loop. Adding methods for these would
couple the spec to specific generation strategies that are appropriately the runtime's domain.

### 13.2 `declare_state()` vs. a Config File

Config files are parsed at load time with no runtime context. `declare_state()` is called
with `(max_batch, max_tokens)` — the runtime's actual allocation parameters. This lets the
model declare shapes as functions of allocation context (e.g., a model running with
max_batch=1 might allocate differently than one running with max_batch=256). It also makes
the contract verifiable at call time with full type checking, rather than at config parse time
with string matching.

### 13.3 Semantic Tags vs. Custom Memory

A fully opaque state contract would reduce the spec to "give me N bytes per request."
This prevents runtimes from making intelligent policy decisions. An `AttentionKV` slot can
be paged, prefix-shared, and evicted when the request window passes. An `SSMState` slot
must be preserved exactly and cannot be paged across token positions. The runtime needs
to know this difference — not to understand the model's math, but to manage memory correctly.
Semantic tags communicate *role* without exposing *implementation*.

#### On Architecture Disclosure

`declare_state()` reveals layer count and per-layer state types. For organizations with
proprietary model architectures, this may be a concern.

Models MAY aggregate state into fewer slots using `Custom` tags to preserve architectural
confidentiality. Example: instead of per-layer KV slots, declare a single
`Custom{extension_id: "aria.core.attention_state"}` slot with total size requirements. The model
manages internal layout; the runtime allocates the blob and preserves it.

This sacrifices runtime optimization (the runtime cannot page, prefix-share, or evict
within an `Custom` slot) in exchange for confidentiality. It is a supported pattern,
not a workaround.

The ARIA recommendation is: use Standard tags wherever possible for maximum runtime
optimization potential, and fall back to Custom aggregation only where architectural
confidentiality is a hard requirement. See `CONFORMANCE.md` for portability implications.

### 13.4 Per-Request Handles with Batch Overloads

ARIA provides both single-handle and batch overloads for `prefill()` and `decode()`.
The two paths serve different use cases:

- **Single-handle overloads** (`prefill(tokens, state)`, `decode(token, state)`): For
  single-request workloads, edge devices, sequential testing, and language bindings where
  per-call simplicity matters.

- **Batch overloads** (`prefill(batch)`, `decode(batch)`): The production path. The
  runtime assembles requests each iteration and calls the model once. The model executes
  all tokens in a single GPU kernel. This is the correct design for continuous batching,
  iteration-level scheduling, and high-throughput serving.

`max_batch` in `declare_state()` declares capacity — the model uses it to pre-size batch
workspace (intermediate tensors, activation buffers) at load time rather than per-call.

### 13.5 Why Not Extend ONNX or WASI-NN

ONNX's static computational graph cannot express dynamic control flow (MoE routing,
conditional computation) or mutable recurrent state without vendor extensions that break
cross-platform portability. Adding stateful decode would require a ground-up redesign.

WASI-NN is architecturally promising but its inference contract is stateless by design.
Its graph execution model does not have a `declare_state` analogue. Retrofitting stateful
autoregressive decode would require changes as substantial as defining a new interface.
We chose to define the correct abstraction directly.

Neither this analysis nor ARIA's existence is a criticism of ONNX or WASI-NN. They solve
different problems in the same space. ARIA fills the gap both leave open.

### 13.6 Synchronous-First Design

ARIA's interface is synchronous. Async execution and token streaming are above ARIA —
the runtime wraps `decode()` in an async executor. This keeps the spec language-agnostic
and avoids prescribing a specific async model (futures, coroutines, callbacks, CSP channels)
that would conflict with different language ecosystems. Cancellation is the one exception:
`cancel()` is thread-safe and defined at the ARIA boundary because it must cross the
model-runtime boundary to abort in-flight compute — it cannot live purely above ARIA.

### 13.7 ARIA Works With Compiled Graphs

ARIA does not assume eager (interpreter-based) model execution. A conformant model may be
an ahead-of-time compiled computation graph — a TensorRT engine, a Qualcomm QNN graph,
a CoreML model, an Intel SynapseAI binary, or any custom NPU artifact — that implements
`ILanguageModel` through its native execution mechanism.

In this case, `prefill()` and `decode()` are logical operations, not necessarily separate
GPU kernel dispatches. The compiled graph's execution engine calls the appropriate compiled
kernel and writes results into the `StateHandle` slots. The ARIA contract is satisfied at
the interface level regardless of how underlying compute is orchestrated.

This applies to: Qualcomm Hexagon NPU (QNN), Apple Silicon (CoreML), Intel Gaudi
(SynapseAI), NVIDIA TensorRT, and any future hardware that benefits from ahead-of-time
compilation.

---

## 14. Versioning and Capability Negotiation

### 14.1 Version Format

ARIA uses `major.minor.patch` versioning:
- **Major bump**: breaking change (new required method, changed signature, removed method,
  changed invariant)
- **Minor bump**: additive change (new optional semantic tag, new error variant, new optional
  field in an existing struct, new required built-in adapter)
- **Patch bump**: corrections only — no new interface surface, no new required behaviors.
  Patch versions clarify ambiguous text, correct factual errors, fix ABI inconsistencies,
  and complete omissions in existing normative content. Conforming implementations of
  `X.Y.Z` are conforming implementations of `X.Y.0`; no re-implementation is required
  to upgrade from `X.Y.0` to `X.Y.Z`.

A model targets a specific version by declaring `(major, minor)` in `ModelInfo.aria_version`.
Patch versions are editorial; the declared version need not include the patch number.

**Compatibility rules**:
- **Within a major version**: A runtime supporting ARIA `X.N` MUST support any model
  targeting ARIA `X.M` where M ≤ N (full minor-version backward compatibility).
- **Across major versions**: A runtime MAY support multiple major versions but is not
  required to. A runtime MUST return `VersionIncompatible` for any major version it
  does not support.
- **Deprecation**: When a new major version is released, the previous major version
  receives critical fixes for 24 months (per CONTRIBUTING.md).

### 14.2 Conformance

A **conforming model** MUST implement all six methods of `ILanguageModel` (`model_info`,
`declare_state`, `prefill` ×2 overloads, `decode` ×2 overloads), provide a conformant
`ITokenizer` via `ModelInfo.default_tokenizer`, and uphold all model invariants in §10.

A **conforming runtime** MUST call methods in the specified order, allocate all declared slots,
provide all required `StateHandle` accessors, and uphold all runtime invariants in §10.

Partial conformance (implementing a subset of methods) is not valid.

ARIA 0.6-draft defines one conformance level with one **portability profile**:

- **ARIA-Conformant**: Full contract including `Custom` semantic tags, dtypes, and layouts.
  Sufficient for interoperability where both sides agree on Custom definitions.
- **ARIA-Portable** (portability profile): All state uses Standard semantic tags only
  (zero `Custom` slots). An ARIA-Portable model is guaranteed to work with any ARIA-Portable
  runtime without out-of-band agreements. This is a stricter subset of conformant.

See `CONFORMANCE.md` for the full checklists and portability requirements.

### 14.3 Intellectual Property

ARIA specification text is available under
[Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

ShareAlike is intentional: any derivative specification or modified version of this document
MUST be published under the same license. Proprietary forks of the ARIA spec are not permitted.
Implementations of the spec (model code, runtime code) are not derivatives of the spec text
and remain governed by their authors' chosen licenses.

Reference implementations and tooling are available under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

**Patent non-assertion**: The authors make no patent claims on the interface contracts,
semantic tag system, state declaration mechanism, KV dtype negotiation protocol, or any
other concept described in this specification. To the best of the authors' knowledge,
the interfaces described herein are free to implement without patent license.

Contributions to this specification are accepted under the Developer Certificate of Origin
(DCO). By contributing, you certify that you have the right to submit the contribution
under these terms. See `CONTRIBUTING.md` for the complete policy.

---

## 15. Future Extensions

> **Known limitation (0.7)**: `declare_state()` accepts a single `max_tokens` value,
> meaning all handles share the same maximum sequence length allocation. In production
> deployments with mixed request lengths, this may cause memory overallocation for short
> requests. Implementors building memory-efficient runtimes should track actual
> `current_length()` per handle to reclaim unused sequence capacity within their allocator.

ARIA 0.7 is a community-maintained open standard. Future extensions are driven by
community proposal and review. The following items have been identified and are
proposed for community implementation. Each requires a proposal, reference implementation,
and community review per `CONTRIBUTING.md` before merging.

**Interface additions:**
- `truncate(handle, new_length)` — rewind `current_length()` for beam search and resampling
- `mixed_batch(List<MixedRequest>)` — single call combining prefill and decode sequences,
  eliminating double GPU dispatch per scheduling step for continuous batching runtimes
- `declare_state_async()` — non-blocking variant of `declare_state()` with completion
  callback, for hardware that performs ahead-of-time graph compilation at this call site
  (Intel Gaudi / SynapseAI, Qualcomm QNN, Apple CoreML)
- `extract_streaming(token)` — incremental Model Protocol Adapter API for token-by-token
  tool-call boundary detection in streaming generation

**C ABI additions:**
- `aria_execution_context_t` — opaque execution context parameter on inference calls;
  maps to CUDA stream on NVIDIA, Metal command queue on Apple Silicon, HIP stream on AMD
- `aria_load_options_t` — extended load options struct covering: provenance verification
  callback (hash/signature/TEE attestation), progress callback, device index for
  multi-GPU and multi-NPU placement

**Type system additions:**
- `architecture_family` field on `ModelInfo` — structured enum (attention / ssm / hybrid /
  moe / unknown) for tooling, routing, and capability display
- `model_fingerprint: Option<bytes>` on `ModelInfo` — cryptographic hash of model weights
  computed at load time, enabling verifiable model identity for private compute deployments
- `caller_request_id: Option<String>` on `PrefillRequest` / `DecodeRequest` — opaque
  caller-provided identifier for billing, audit, and priority scheduling correlation
- `request_metadata: Option<bytes>` on `PrefillRequest` / `DecodeRequest` — opaque
  passthrough field preserved by the runtime, not interpreted by ARIA; enables
  safety policy context and permission scope to cross the model boundary
- `lazy: bool` on `StateSlot` — defer allocation to first model access; required for
  MoE architectures with large expert counts where most experts are inactive per token

**Protocol additions:**
- Per-slot KV dtype negotiation — expressive alternative to model-wide negotiation
  for MoE models with heterogeneous per-layer dtype requirements
- StateHandle serialization — export / import handle state to bytes for cross-session
  persistence, checkpoint / restore across model restarts, and session migration

**Governance and ecosystem:**
- Audit logging extension profile — optional `aria_audit_callback_t` for enterprise
  compliance logging (inference call events, token counts, timestamps)
- Normative translation: Simplified Chinese — to support adoption in the CN ecosystem
  (Qwen, DeepSeek, 01.AI, Baidu, etc.); translation MUST be maintained in sync with
  the English normative text and submitted via the standard PR process with a designated
  translation maintainer
- Normative translation: French — to support adoption in the FR/EU ecosystem
  (Mistral AI and the broader European open-source community); same maintenance
  requirements as above
- Normative translation: Russian — to support adoption in the RU open-source ecosystem;
  same maintenance requirements as above
- Encoder-decoder support — `CrossAttentionKV` semantic tag, full encoder-decoder scope
- LoRA / adapter weight interface
- Multimodal inputs (vision, audio)
- Distributed multi-node inference contract

To propose an extension, follow the process in `CONTRIBUTING.md` — open an issue or PR,
provide a reference implementation, and allow the 7–14 day community review window.
Semantic tag proposals follow the Tag Registry process (CONTRIBUTING.md §4).

---

## Appendix A: ARIA in Context — Complementary Specifications

ARIA is not a replacement for any existing specification. Each spec solves a different
problem. The table below shows how ARIA is **additive**, not competitive.

| Spec | What it does well | The gap ARIA fills | How they work together |
|------|------------------|-------------------|------------------------|
| **ONNX** | Static model interchange, cross-framework portability, operator coverage | Static graph — cannot express mutable per-request decode state or dynamic control flow (MoE routing) | ONNX exports model weights; ARIA standardizes the serving-time stateful inference contract |
| **WASI-NN** | Hardware-agnostic inference backend interface; WebAssembly portability | Stateless contract — no per-request lifecycle, no state allocation protocol | ARIA's `ILanguageModel` could be adopted as a WASI-NN Stateful Inference Profile (see §13.5) |
| **GGUF** | Single-file distribution with weights, metadata, and tokenizer config in one file | File format — no serving-time contract; how the model runs is undefined | GGUF distributes the model file; ARIA defines how a llama.cpp runtime calls into it at serving time |
| **SafeTensors** | Safe, fast, ML-framework-agnostic weight serialization; zero-copy loading | Weight storage only — no inference contract | SafeTensors stores weights; ARIA defines the inference interface |
| **TIS Backend API** | Triton Inference Server's framework-to-backend serving contract | No standardized stateful decode — each backend implements its own KV state management | Complementary serving contracts at different layers; TIS consumer API above, ARIA model-runtime boundary below |
| **HF `generate()`** | Broad model support, sampling library, Python ecosystem integration | Python-only; exposes model internals (past_key_values); not language-agnostic | HuggingFace models can expose an ARIA interface for use outside Python; generate() remains the sampling-layer API |

**The single gap none of these address**: a language-agnostic, vendor-neutral,
architecture-agnostic contract for the **stateful autoregressive decode loop with
heterogeneous state types**. That gap is what ARIA fills.

These complementary specs are also potential integration partners for demonstrating ARIA's
value: an ONNX-exported model wrapped in an ARIA interface, or an ARIA runtime adapter for
WASI-NN, would demonstrate interoperability rather than competition.

---

## Appendix B: Framework Compatibility Notes

### vLLM

- **PagedAttention**: `AttentionKV` + `Layout::Paged` maps directly to vLLM's physical block
  tables. vLLM chooses its own `page_tokens` granularity; the model declares a preference.
  See §4.5.2 for the two conformant TensorView strategies for paged layouts.
- **Continuous batching**: ARIA handles are per-request. vLLM's iteration-level scheduler
  assembles batches and can call `decode()` across N handles per scheduling step; the model
  implementation batches the GPU kernel internally.
- **Prefix caching**: vLLM's RadixAttention is a runtime optimization on `AttentionKV` pages;
  the model writes through its handle and never observes physical sharing.
- **Speculative decoding**: vLLM drives a draft model as a separate `ILanguageModel` instance;
  the target model verifies via `prefill()` with a full-sequence `out_logits` buffer.

### SGLang

- **Chunked prefill**: `prefill()` may be called multiple times on a `Ready` handle.
  SGLang can chunk any long prompt into GPU-efficient slices without model changes.
- **RadixAttention**: runtime-internal; invisible to the model.
- **Tensor parallelism**: model-internal; the model's `decode()` implementation may use
  NCCL collectives. ARIA's synchronous interface wraps the parallel call.

### llama.cpp

- **KV quantization** (e.g., `-ctk q8_0`): model declares `supported_kv_dtypes`; llama.cpp
  selects a compatible dtype and allocates accordingly. Model reads actual dtype from
  `handle.slot().dtype` and dequantizes in the attention kernel.
- **Context shifting**: llama.cpp's context shifting is a model-level optimization where the
  model overwrites its own KV entries to extend context. In ARIA, this is valid — the model
  controls compute and may rewrite KV positions < `current_length()` if it implements
  context shifting. `AttentionKV`'s append-only invariant is relaxed for implementations
  that explicitly implement context extension algorithms.
- **Metal / CPU backends**: ARIA is hardware-agnostic. llama.cpp selects its backend
  transparently within its model implementation.

### TGI (Text Generation Inference)

- Flash Attention 2 is model-internal. TGI implements ARIA's `prefill/decode` backed by FA2.
- Token streaming: TGI's streaming is above ARIA — it consumes `decode()` results per step.

### LMDeploy (TurboMind engine)

- TurboMind manages its own block tables and persistent batch state. In an ARIA binding,
  TurboMind implements the runtime side — allocating handles backed by its internal block
  manager, exposing ARIA-compliant accessors.

### Intel Gaudi / IPEX-LLM (intel-analytics/ipex-llm)

- Intel Gaudi accelerators use SynapseAI with a graph compiler that pre-fuses ops at
  load time. In an ARIA binding, `declare_state()` triggers graph compilation; subsequent
  `prefill/decode` calls dispatch compiled kernels.
- `FloatCustom` dtypes cover Intel-specific quantization formats where `Float8E4M3` or
  standard enums do not match exactly (e.g., Intel FP8 E4M3FNUZ bias variant).
- IPEX-LLM's CPU optimization paths (AMX, AVX-512) are model-internal; ARIA's interface
  is unchanged.

### Apple Silicon (CoreML / Metal)

- A CoreML-compiled model implements `ILanguageModel` via a Swift/ObjC bridge. `prefill()`
  and `decode()` dispatch Metal compute shaders. State slots map to Metal buffers allocated
  by the runtime in unified memory.
- Memory coherence is via Metal command buffer completion (§4.5.1); the runtime waits on
  the command buffer before reading logits.
- `DType.Float16` maps naturally to Apple's preferred `float16` compute dtype. KV in
  `Int8` is also viable on M-series hardware.

### Qualcomm QNN (Hexagon NPU)

- A QNN-compiled model wraps a Qualcomm graph in an ARIA binding. `declare_state()` returns
  the tensor shapes required by the compiled graph. `prefill/decode` dispatch QNN graph
  execution on the Hexagon NPU.
- State memory is allocated via QNN's memory API and exposed through ARIA's TensorView
  accessors; coherence uses QNN completion callbacks mapped to synchronous return.
- `FloatCustom` covers Qualcomm-specific quantization formats where standard enums don't
  map cleanly.

---

## Appendix C: Reference Implementation Skeleton

```python
# Illustrates a hybrid attention+SSM model's ARIA implementation.
# Not production code — shows the contract shape only.

from aria import (
    ILanguageModel, ModelInfo, StateSpec, StateSlot, StateHandle,
    SemanticTag, DType, Layout, Dimension, ARIAError, Result, ITokenizer, TensorView
)

class HybridModel(ILanguageModel):

    def model_info(self) -> Result[ModelInfo, ARIAError]:
        return ModelInfo(
            aria_version=(0, 6),
            vocab_size=152064,
            model_name="HybridModel-7B",
            max_sequence_length=32768,
            supported_kv_dtypes=[DType.Float16, DType.Float8E4M3, DType.Int8],
            capabilities=["aria.core.context_shift", "aria.core.layout.paged"],
            default_tokenizer=MyTokenizer(),
        )

    def declare_state(self, max_batch: int, max_tokens: int) -> Result[StateSpec, ARIAError]:
        slots = []
        for i in range(self.num_layers):
            if self.layer_type(i) == "attention":
                # GQA: use num_kv_heads, not num_q_heads
                for is_key in [True, False]:
                    slots.append(StateSlot(
                        id=f"kv_{'k' if is_key else 'v'}_{i}",
                        semantic_tag=SemanticTag.AttentionKV(layer_idx=i, is_key=is_key),
                        shape=[
                            Dimension.Sequence,
                            Dimension.ModelParam("num_kv_heads", self.num_kv_heads),
                            Dimension.ModelParam("head_dim", self.head_dim),
                        ],
                        dtype=DType.Float16,          # preferred; negotiated via §8
                        layout=Layout.Paged(preferred_page_tokens=16),
                    ))
            elif self.layer_type(i) == "ssm":
                slots.append(StateSlot(
                    id=f"ssm_{i}",
                    semantic_tag=SemanticTag.SSMState(layer_idx=i),
                    shape=[
                        Dimension.ModelParam("d_state", self.d_state),
                        Dimension.ModelParam("d_inner", self.d_inner),
                    ],
                    dtype=DType.Float16,
                    layout=Layout.Contiguous,
                ))
        return StateSpec(slots=slots)

    def prefill(self, tokens: list[int], state: StateHandle,
                out_logits: TensorView = None) -> Result[None, ARIAError]:
        pos_offset = state.current_length()
        num_tokens = len(tokens)

        # Decide which positions need logits based on output buffer shape
        start_logits_at = num_tokens  # default: none (S10 optimization)
        if out_logits:
            # B1: Write directly to caller-owned buffer
            if out_logits.shape[0] == 1:            # Last position only
                start_logits_at = num_tokens - 1
            elif out_logits.shape[0] == num_tokens: # All positions
                start_logits_at = 0

        for i, token in enumerate(tokens):
            abs_pos = pos_offset + i
            compute_logits = (i >= start_logits_at)

            # Forward pass writes to state immediately
            logits = self._forward(token, abs_pos, state, write_state=True,
                                   return_logits=compute_logits)
            if compute_logits:
                out_idx = i if start_logits_at == 0 else 0
                out_logits[out_idx] = logits

        return None

    def decode(self, token: int, state: StateHandle,
               out_logits: TensorView = None) -> Result[None, ARIAError]:
        pos = state.current_length()
        # Decode always computes logits if buffer provided
        logits = self._forward(token, pos, state, write_state=True,
                               return_logits=(out_logits is not None))
        if out_logits:
            out_logits[0] = logits  # B1: Zero-copy write
        return None

    def _forward(self, token, position, state, write_state, return_logits=True):
        x = self.embed(token)
        for i, layer in enumerate(self.layers):
            if layer.type == "attention":
                # Read KV for all prior positions; write current KV
                kv_k = state.slot(f"kv_k_{i}")   # TensorView into runtime memory
                kv_v = state.slot(f"kv_v_{i}")
                x = layer(x, position, kv_k, kv_v, write=write_state)
            elif layer.type == "ssm":
                # S9: SSM state initialized if new, overwritten if existing
                ssm = state.slot(f"ssm_{i}")
                x = layer(x, ssm, write=write_state)
        if return_logits:
            return self.lm_head(x)
        return None
```
