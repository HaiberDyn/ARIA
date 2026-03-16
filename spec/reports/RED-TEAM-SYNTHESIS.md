# ARIA 0.6.2-draft — Red Team Synthesis
> Per-panelist: green signals first, then red. Full panel: 6 harness makers, 10 model makers, 7 hardware vendors, 1 HCS expert, 4 security leads.

---

## HARNESS / RUNTIME MAKERS

---

**Woojin Lee — vLLM — PagedAttention / Continuous Batching Lead**

Green:
- §4.5.2: Two conformant TensorView strategies (Virtual Contiguity + Block Table Access) map precisely to vLLM's architecture. Block Table Access was clearly designed with PagedAttention in mind. The physical block table model is understood.
- §6.1: AttentionKV + window_size eviction semantics are exactly what sliding window attention needs. The table is precise and actionable.
- §9 batch semantics: two-level `Result<List<Result<>>, E>` is production-correct. Outer Err = catastrophic only. Per-item failures don't abort the batch. This is the right design for continuous batching.

Red:
- §4.5.2 Strategy 2: "exposes the block table structure (page pointers, block size) directly" — but defines no schema. vLLM's block table is `block_tables: Tensor[num_sequences, max_blocks_per_seq]`. The spec leaves this unspecified. Strategy 2 is unimplementable portably — every runtime would expose a different structure. The C ABI (§16) must define `aria_block_table_t`.
- §9 vs §16 — critical inconsistency: The spec defines `Result<List<Result<>>, ARIAError>` for batch calls (§9), but the C ABI (§16) returns a single `aria_error_t` from `prefill_batch` and `decode_batch`. The two-level result is not implemented in the normative ABI. This is a spec-ABI split that makes the batch error semantics unenforceable in C bindings.
- §7 no `truncate()`: beam search and resampling require walking back `current_length()`. There is currently no way to do this. Noted in §15 as 1.0 planned — but this is a real adoption blocker for search-based decoding strategies.
- §4.4 Dimension.Sequence: resolves to `max_tokens` for all handles from a single `declare_state()` call. Different requests in a continuous batch have different natural context lengths. Memory overallocation for short requests is acknowledged in §15, but the `BatchProfile` fix is deferred to 1.0 with no workaround specified for 0.6.2.

---

**Wei Zhu — SGLang — RadixAttention / Chunked Prefill Lead**

Green:
- §7.4: Multi-call prefill is clean. SGLang's chunked prefill maps directly — prefill on Ready handles extends the sequence. "Only new, unprocessed tokens" note is correct.
- §11.2: RadixAttention is runtime-internal, invisible to model. Correct — prefix sharing happens at the block level; the model never sees physical sharing.
- §6.1 prefix sharing column: AttentionKV (no window) = Yes. SGLang can share prefixes transparently behind the handle abstraction.

Red:
- §4.3 Layout::Paged contradiction: "runtime is not required to honor" `preferred_page_tokens` yet "MUST align its page size to the model's compute requirements." These two sentences conflict. When does incompatibility justify `SlotUnsupported` vs silent override? Define the threshold.
- §8 one-shot dtype negotiation: SGLang supports dynamic FP8 KV on H100. A user changing KV dtype at runtime without model reload has no path — negotiation is a single call at `declare_state()` time. Specify a re-negotiation mechanism or explicitly forbid runtime dtype changes and accept the operational limitation.
- §5.1 no mixed prefill+decode batch: SGLang (and every continuous batching runtime) mixes prefill and decode sequences in the same scheduling step. ARIA requires two separate calls (`prefill_batch` then `decode_batch`), meaning two GPU kernel dispatches per step. A `mixed_batch()` method for 1.0 would eliminate this overhead — flag the design gap now.

---

**Georgi Gerganov — llama.cpp — CPU Inference / GGUF / Quantization Lead**

Green:
- §4.2: `FloatCustom{e_bits, m_bits}` covers exotic quantization formats (iq2_xxs, iq4_nl) at the KV level. The note explicitly clarifying that GGUF block quants apply to weights not KV slots is thoughtful and prevents a common misunderstanding.
- §11.1: "use any hardware: CUDA, ROCm, Metal, CPU" — llama.cpp's multi-backend design is explicitly in scope.
- Appendix B llama.cpp: context shifting described correctly. `aria.core.context_shift` capability flag is the right mechanism.

Red:
- §16 C ABI tokenizer is incomplete: `ITokenizer` (§5.5) defines `encode(batch)`, `decode(batch)`, `vocab_size()`, and `special_tokens()`. The C ABI vtable only has `encode` (single) and `decode_token` (single token). Batch methods and `special_tokens()` are missing. llama.cpp uses special tokens extensively (BOS, EOS, system prompt tokens, tool tokens).
- §16 `aria_load_model(path)`: path-only loading bakes in filesystem assumptions. llama.cpp supports loading from URLs, memory buffers, and custom backends. Define a source abstraction or loader callback.
- §7.3 Exhausted state + context_shift conflict: `current_length() == max_tokens()` → Exhausted, no further decode allowed. But context_shift models are never logically exhausted — they overwrite stale KV entries to extend indefinitely. Specify how the lifecycle state machine applies to models declaring `aria.core.context_shift`.

---

**Nicolas Patry — TGI (HuggingFace) — Flash Attention / Production Serving Lead**

Green:
- §9 `ModelError.detail` sanitization: exactly what TGI needs for structured logging and multi-tenant error propagation. TGI surfaces errors to end users; unsanitized model strings are a live injection vector today.
- §14: 24-month critical fix window for previous major version. Production systems cannot upgrade immediately. This commitment matters.
- §9 error taxonomy table: "Raised by / Fatal / Retriable / Handle valid after?" maps directly to TGI's retry and circuit-breaker logic.

Red:
- §5.5 tokenizer substitution: "semantically equivalent" is undefined and unverifiable. A substitute tokenizer MUST produce identical output to the default for a normative test vector set. Without this, "semantically equivalent" is an honor system with no enforcement path. Add a test vector requirement.
- §9 vs §16 ABI inconsistency (same as vLLM above): per-item batch results in the high-level spec but single `aria_error_t` in the C ABI. Fatal issue for TGI's per-request error reporting.
- §12: multi-model runtimes (TGI serves multiple models, dynamic routing) — handles have no namespace or model binding. If two model instances have overlapping `handle.id()` ranges, there's no spec-level way to detect a handle passed to the wrong model instance. `InvalidHandle` would fire, but the root cause is undiagnosable. `handle.id()` should carry a model-instance token.
- §2.2 cancellation out of scope: for long-context models (prefill > 30s), this is larger than the spec acknowledges. Flag as a 1.0 priority with a preliminary design (e.g., `cancel(handle)` returns immediately and marks the handle Released).

---

**Kai Zhang — LMDeploy — TurboMind / CN Production Scale Lead**

Green:
- §4.3 `preferred_page_tokens` hint: TurboMind uses fixed-size block tables. The hint mechanism is appropriate — TurboMind sets its own granularity.
- §10 Runtime Invariant 11: operator resource limits on `declare_state()`. At CN production scale, memory limits are hard constraints. This invariant is essential.

Red:
- §3 Definitions: "Harness" as synonym for "Runtime" conflates LMDeploy's TurboMind engine and its serving layer. Multi-tier architectures need finer vocabulary. Suggest: "Runtime" = inference engine, "Harness" = full serving stack including scheduling and API layer.
- §16 ABI no device placement: `aria_load_model(path)` gives no way to specify GPU index, NUMA node, or device affinity. Multi-GPU deployments require out-of-band configuration that bypasses the spec. Add a `aria_load_options_t` struct.
- No caller-provided `request_id` in `PrefillRequest`/`DecodeRequest`. Priority scheduling and reordering break result correlation when only positional ordering is available. Handle IDs are runtime-assigned — not stable from the caller's perspective.
- §14.1: 24-month fix window is too short for CN enterprise with 3-year hardware lifecycles. Acknowledge that operators MAY negotiate longer support contracts with deploying organizations.

---

**Jeffrey Morgan — Ollama — Developer UX / Local Model Management Lead**

Green:
- §15: LoRA / adapter management explicitly out of scope with correct rationale ("weight management, not state management"). Ollama's modelfile adapter system sits cleanly outside ARIA.
- §13.6: Synchronous-first design. Ollama's HTTP handlers wrap ARIA's sync interface cleanly. No async model prescribed.

Red:
- §4.9 `ModelInfo`: no structured `architecture_family` field. Ollama displays model capabilities in UI and routes requests accordingly. `model_name` is human-readable and unstable. Add `architecture_family: enum { attention | ssm | hybrid | moe | unknown }` or equivalent capability flag.
- §5.5 ITokenizer: no discovery mechanism for tokenizer format. Ollama manages dozens of models with tiktoken, sentencepiece, BPE. Which `ITokenizer` implementation to instantiate is unspecified. `tokenizer_source` in the C ABI (`aria_model_info_t`) is "optional path or JSON string" — too vague to act on.
- §16 `aria_load_model()`: no progress callback. Loading 70B+ models takes 30–60 seconds. Ollama's UX shows load progress. A blocking silent load is unusable for responsive tooling.
- §6.2 MPA conformance status: CONFORMANCE.md says "implementations that include the Model Protocol Adapter layer MUST…" — the word "include" makes MPA optional. If optional, it should be in EXTENSIONS.md. If required for tool-calling models, say so. Clarify normative status.

---

## MODEL MAKERS

---

**Sam Altman / Ilya Sutskever proxy — OpenAI — GPT-4/o Architecture Team**

Green:
- §13.1: "No fourth generation primitive." Minimalism is correct — adding methods for beam search or speculative decoding orchestration would couple the spec to specific strategies that are appropriately above ARIA.
- §8: KV dtype negotiation ordered preference list is well-designed. Expresses preference hierarchy while giving runtimes allocation flexibility.

Red:
- §6.2.5 required adapters: `QwenAdapter` (XML-JSON) and `ClaudeAdapter` (nested XML) are required, but there is no adapter for OpenAI's JSON function-calling format: `{"type":"function","function":{"name":"...","arguments":"..."}}`. OpenAI-compatible tool format is the de facto ecosystem standard — supported by vLLM, SGLang, Ollama, and hundreds of other runtimes. Its absence from the required adapter list is the single largest gap in §6.2.
- §3.1 type safety: "implementations SHOULD enforce at compile time" — SHOULD is too weak. Python bindings (primary prototyping language) cannot have compile-time enforcement. Specify: non-compiled bindings MUST provide runtime type checking as a substitute for compile-time enforcement.
- §4.9: `max_sequence_length` is a single value. Models where different modes (standard, vision, tool-use) have different effective context limits cannot express this. Consider a `context_profiles` map.

---

**Dario Amodei proxy — Anthropic — Claude / Constitutional AI Team**

Green:
- §6.2.3 `extract()` "MUST NOT raise an error for malformed content — return None": defensive extraction is exactly right. Error on malformed XML mid-stream would break every streaming harness.
- §10 Model Invariant 5: "model MUST NOT allocate persistent per-request state outside StateHandle" — the critical multi-tenant isolation property. Essential for Constitutional AI's safety contract.
- §14.3: Patent non-assertion pledge is essential for adoption. Without it, this spec is dead on arrival for any serious implementor.

Red:
- §6.2.5 `ClaudeAdapter`: the format shown is `<tool_call><name>search</name><query>example</query></tool_call>`. Anthropic's actual API tool format uses `<function_calls>\n<invoke name="tool_name">\n<parameter name="param">value\n</invoke>\n</invoke>\n</function_calls>` not the format in the spec. Implementors building a ClaudeAdapter will produce something that doesn't match real Claude output. Either use Anthropic's actual current format or clearly label the example as illustrative-only and point to Anthropic's documentation.
- §10 Model Invariant 8: "Logits MUST be finite for valid inputs." For very long contexts (10M+ tokens), numerical precision issues in attention computations can produce NaN at extreme depths. "Valid inputs" needs quantitative definition — token IDs in range, handle in Ready state, sequence length within declared max. Specify the exact preconditions under which finiteness is guaranteed.
- §5.2: No request metadata field. Constitutional AI safety filtering requires per-request policy context to travel through the ARIA boundary. An opaque `request_metadata: bytes` field in `PrefillRequest`/`DecodeRequest` (preserved, not interpreted by ARIA) would enable this without coupling the spec to specific safety architectures.

---

**Guillaume Lample / Hugo Touvron proxy — Meta AI — LLaMA Team**

Green:
- §4.4 `ModelParam("num_kv_heads", 8)`: excellent for GQA. LLaMA 3.1's `num_kv_heads=8` vs `num_q_heads=32` expressed cleanly and self-documentingly.
- §4.5.3: explicitly notes "GQA/MQA will differ from query head count." LLaMA popularized GQA and it is correctly handled.
- §10 Model Invariant 3: context_shift carve-out is appropriate for YaRN, NTK scaling, and sliding window implementations.

Red:
- §5.1 no mixed prefill+decode batch: LLaMA production deployments (vLLM, SGLang) always mix prefill and decode sequences in the same GPU step. ARIA requires separate `prefill_batch()` and `decode_batch()` calls — two GPU kernel launches per step instead of one. This is the single biggest inference throughput gap. Add `mixed_batch()` to the 1.0 roadmap with a clear design now.
- §4.5.3: two separate KV slots (`kv_k_i`, `kv_v_i`) prevents fused KV layout. LLaMA + FlashAttention uses interleaved K/V for memory efficiency. Declaring them separately forces disjoint allocations. Consider a `FusedAttentionKV` tag variant.
- §6.2.5: no LLaMA tool-call adapter. LLaMA models fine-tuned for tool use have their own format (OpenAI JSON-compatible, not XML). The Qwen and Claude adapters don't cover the LLaMA ecosystem. An `OpenAIJsonAdapter` should be the third required adapter alongside Qwen and Claude.
- §14 implementation gate: "requires 2+ independent 0.6 implementations" is a good gate, but there is no mechanism to track progress toward it. Define how implementations register and how the working group monitors the count. Otherwise the 1.0 gate is invisible and the spec stalls.

---

**Arthur Mensch proxy — Mistral AI — Sliding Window / MoE / Efficiency Lead**

Green:
- §6 `AttentionKV` with `window_size`: designed with Mistral's sliding window attention in mind. The semantic is precise — runtime MAY evict pages beyond window_size.
- §6.1 table: the row for `AttentionKV` + window correctly specifies "Yes (stale pages only)" for intra-handle eviction. Exactly right for Mixtral.

Red:
- §6 `StateSlot`: no "shared across heads" or "shared across experts" mechanism. Mixtral's MoE uses expert routing where different expert groups may share or partially share KV state. No way to declare that two logical layers share a physical allocation.
- §6.2.5: no `MistralAdapter`. Mistral's tool-call format is `[TOOL_CALLS] [{"name": "...", "arguments": {...}}]` — JSON array with a prefix token, neither XML-JSON (Qwen) nor nested XML (Claude). Mistral is one of the most widely deployed open models. Its absence from required adapters is a significant gap.
- §5.1: MoE models have variable compute cost per token. A decode step activating 8 of 8 experts is 4× more expensive than activating 2 of 8. The harness cannot make SLO-aware scheduling decisions without compute cost visibility. An optional `compute_cost_hint` return value from `decode()` would enable smarter scheduling.
- §8: one KV dtype for the entire model. MoE models may want different dtypes for expert layers vs attention layers. Per-slot dtype negotiation would be more expressive.

---

**Jinze Bao proxy — Alibaba/Qwen — Qwen3 Hybrid / MoE / CN Ecosystem Lead**

Green:
- §6 `SSMState`: correct semantics for Qwen3's hybrid Mamba+attention layers. Fixed size, fully overwritten, no paging. The list of examples ("Mamba, GDN, RWKV, etc.") shows awareness of the hybrid landscape.
- §6.2.5 `QwenAdapter`: the spec has a working adapter for Qwen's native format. The JSON-in-XML wrapper is correctly identified.
- §6.1: table correctly differentiates AttentionKV vs SSMState memory management for hybrid model deployment.

Red:
- §6.2.5 `QwenAdapter` versioning: Qwen2 and Qwen3 tool-call formats differ in minor ways. As future Qwen versions are released, adapter behavior may need to change. There is no versioning mechanism for adapters — both would register under "Qwen" prefix. A `register(prefix, adapter, version)` API or prefix namespacing (`"Qwen3"`, `"Qwen2"`) is needed.
- §4.10 `StateSpec`: hybrid models with 28+ layers each declare K slot + V slot + SSM slot = 84+ state slots. The spec places no limit lower than 1024. Serializing 84-slot StateSpecs at `declare_state()` is overhead proportional to model depth. A bulk declaration mechanism ("N layers × this pattern") would improve efficiency.
- §6.2.4 `register()`: "duplicate registrations MUST return an error." In microservices, multiple processes independently call `register()` for the same adapter. Idempotent registration (error only if prefix maps to a DIFFERENT adapter) would be more practical than strict duplicate rejection.

---

**Oriol Vinyals proxy — Google DeepMind — Gemma / Gemini / TPU Architecture Team**

Green:
- §13.7: "ARIA works with compiled graphs" is the key compatibility statement for TPU/XLA. `declare_state()` triggering XLA graph compilation is a clean integration model.
- §2.2: multimodal inputs correctly deferred to 2.0. Appropriate scope restraint.
- §14.1: "minor bump: new optional semantic tag" allows adding `CrossAttentionKV` without breaking existing models. Clean.

Red:
- §4.5.1 memory coherence table: no entry for TPU/XLA. XLA's buffer donation and aliasing mechanism is fundamentally different from CUDA stream sync or Metal command buffer completion. The NPU entry groups Qualcomm and Intel but neither entry covers TPU. Add a TPU/XLA row or generalize to cover all compiled-graph execution models.
- §16 C ABI: JAX/XLA models don't use shared library `dlopen` loading. `aria_load_model(path)` maps to a filesystem artifact, but JAX models are Python objects, SavedModels, or compiled XLA executables. The C ABI assumes a shared library mental model incompatible with JAX's deployment paradigm. Define an alternative loading interface for compiled-graph models.
- §4.3 Layout: TPU HBM requires specific alignment and tiling (128-element alignment for bfloat16 on TPUv5). `Contiguous`, `Paged`, and `Custom` don't address alignment requirements. A `Layout::Tiled{tile_h, tile_w}` option, or an alignment field in `StateSlot`, is needed for TPU efficiency.

---

**Igor Babuschkin proxy — xAI — Grok / MoE / Long Context / Real-Time Data Lead**

Green:
- §7.4 multi-call prefill: clean specification. For Grok's long context (128K+), chunked prefill is essential. The re-prefilling warning is correct.
- §15 known limitation: "max_tokens shared across all handles causes overallocation for short requests" — honest acknowledgment with a practical interim note.

Red:
- §9: no `Timeout` error variant. Real-time data integration means some decode steps involve external calls. If a decode step hangs, the harness cannot distinguish a slow model from a hung one at the ARIA interface. Add `ModelError::Timeout { handle_id: uint64, elapsed_ms: uint64 }` or a general timeout mechanism.
- §6.2 MPA streaming incompatibility: `extract(text)` takes a complete text string. For streaming generation, the harness doesn't have the full text until generation completes. There is no incremental adapter API for detecting tool-call boundaries token by token. MPA is effectively incompatible with low-latency streaming — the harness must buffer the entire response before calling `extract()`. Design an incremental `extract_streaming(token)` interface or acknowledge this limitation explicitly in §6.2.
- §10 Model Invariant 7: "thread-safe on different handles." For MoE, model-internal caches outside StateHandle (expert load balancing statistics, routing history) must also be thread-safe. The spec should clarify that all shared mutable state in the model — not only per-request state — must be thread-safe.

---

**Nick Frosst proxy — Cohere — Command / Enterprise RAG / Tool Use Lead**

Green:
- §9 error taxonomy: "Fatal / Retriable / Handle valid after?" columns map directly to enterprise incident classification and SLA-differentiated retry policies.
- §10 Runtime Invariant 7: "MUST NOT silently suppress or retry errors without consumer's knowledge." Critical for enterprise audit trails.

Red:
- §4.9 `ModelInfo`: no language capability declaration. Cohere's Command-R is multilingual; enterprise deployments route requests by language. Without a declared language capability, routing must be done by model_name heuristics.
- §5.1: no request priority in `PrefillRequest`/`DecodeRequest`. Enterprise SLAs require differentiating interactive from batch requests. ARIA has no priority mechanism — the harness implements priority entirely outside the spec, making priority scheduling invisible at the model boundary.
- §6.2.2: `ToolInvocation.parameters` carries extracted values but no schema validation result. Enterprise integrations need to distinguish "parameter structurally extracted" from "parameter matches tool schema." A `validation_errors: List<String>` field (empty = valid) would enable downstream validation-aware routing.
- §14.2 conformance: GitHub Issue-based conformance claiming is unusable for enterprise vendors with strict contribution policies. Provide an enterprise self-attestation path without requiring public GitHub interaction.

---

**Li Jinyu proxy — 01.AI — Yi Series / Multilingual / Asian Market Lead**

Green:
- §4.2 `FloatCustom{e_bits, m_bits}`: covers CN chip ecosystem custom quantization formats not enumerated in standard enums.
- §10 Runtime Invariant 11: conservative defaults for constrained deployment targets are appropriate.

Red:
- §3 and throughout: English-only spec. For CN market adoption, a normative Chinese translation is required — not a convenience, a market prerequisite.
- §14.1: two-level major.minor versioning only. The CN market moves in smaller increments. A patch level (0.6.2.1) for bug fixes that don't warrant a minor bump would reduce deployment friction.
- §16 ABI: `ARIA_BFLOAT16` — no guidance on whether BF16 requires hardware native support or software emulation is acceptable. Older CN chips (Kunlun, Cambricon) don't natively support BF16. Specify: if `ARIA_BFLOAT16` is declared, does the runtime need hardware support, or is software emulation conformant?

---

**Wenfeng Liang proxy — DeepSeek — MoE / Open Weights / CN AI Lab**

Green:
- §6 `SSMState`: DeepSeek-V3's hybrid MLA+MoE architecture requires heterogeneous state. The spec handles this correctly.
- §13.3: Custom tag aggregation for architectural confidentiality — important for commercial model deployment where internal architecture is proprietary IP.

Red:
- §4.11 `StateSlot`: no lazy/on-demand allocation. DeepSeek-V3's 256 expert groups with 8 activated per token requires declaring all 256 expert states upfront, wasting ~97% of expert state memory. A `lazy: bool` flag on `StateSlot` would allow allocation on first activation.
- §6.2.4 overlapping prefix detection: prefixes "deepseek-r" and "deepseek" both match "deepseek-r1" — longest match resolves it, but there's no conflict detection at registration time. Two registrations that could produce ambiguous matches for some model names should either be rejected or warned on.
- §9: no resource pressure warning. Binary `OutOfMemory` (you're dead) vs no warning at all is too coarse for memory-constrained CN deployments. An `OutOfMemoryWarning` or capacity threshold notification would allow the harness to shed load before allocation fails.

---

## HARDWARE / INFERENCE SILICON

---

**Jensen Huang proxy — NVIDIA — H100/B200/GB200 / CUDA / TensorRT Lead**

Green:
- §4.5.1 memory coherence: "CUDA stream synchronization after call completes" — correct mechanism for NVIDIA's actual pipeline.
- §4.2: `Float8E4M3`, `Float8E5M2`, `Float4E2M1` explicitly named — H100 FP8 tensor cores and B200 NVFP4 are both covered.
- §11.1: CUDA graph capture listed as model-internal freedom — NVIDIA's entire inference performance story depends on this being model-internal.

Red:
- §16 C ABI: no CUDA stream parameter in `prefill()`/`decode()`. Real NVIDIA inference frameworks submit all kernels to an explicit stream for ordering. The current ABI executes on an implicit/default stream, making multi-stream inference (overlapping compute and memory transfer) impossible at the ARIA interface level. Add `aria_stream_t stream` parameter to all inference functions.
- §4.5.2 Strategy 2: the block table structure ("page pointers, block size") is described but not defined in the C ABI (§16). vLLM's block table has a concrete representation (`block_tables: Tensor[num_seq, max_blocks]`). An undefined Strategy 2 structure means every runtime invents a different ABI. Define `aria_block_table_t` in §16.
- §4.2: `Float8E4M3` — spec says "OCP MX / NV E4M3" but NVIDIA H100 uses E4M3FNUZ (NaN and zero encoding differ from OCP MX E4M3FN). These are not interchangeable. Distinguish E4M3FN (OCP) from E4M3FNUZ (NVIDIA) in the DType enum or add a note with behavioral difference.

---

**Victor Peng proxy — AMD — MI300X / ROCm / HIP / CDNA Lead**

Green:
- §4.5.1: "CUDA (NVIDIA, AMD ROCm)" — ROCm correctly grouped for stream synchronization.
- §11.1: ROCm explicitly mentioned in hardware freedom list.

Red:
- §16 ABI: MI300X unified 192GB HBM blurs CPU/GPU memory boundaries. The ABI gives no guidance on where tensor memory must reside — device vs host vs unified. Specify memory residency requirements for `aria_tensor_view_t.data` on unified memory architectures.
- §4.5.1: grouping "CUDA (NVIDIA, AMD ROCm)" under a single entry is inaccurate. ROCm's `hipStreamSynchronize` has different fence semantics in edge cases involving managed/unified memory. ROCm deserves its own entry in the coherence table.
- §16: no GPU selection parameter in `aria_load_model()`. MI300X supports multiple GPU instances. Multi-GPU loading requires device selection, but the load function only takes a path. Add an `aria_load_options_t` struct with device index.

---

**Jay Huang proxy — Huawei Ascend — Ascend 910B/910C / CANN Stack Lead**

Green:
- §4.2 `DType::Custom(string)`: essential for Ascend's INT8 variants with non-standard scaling.
- §13.7: compiled graph approach fits Ascend's CANN graph compiler. `declare_state()` triggering graph compilation is the correct integration pattern.

Red:
- §4.5.1 memory coherence table: no entry for Huawei Ascend/CANN. "NPU (Qualcomm, Intel Gaudi)" is listed but Ascend is not. CANN uses AscendCL event primitives for synchronization — distinct from Qualcomm QNN and Intel SynapseAI. Add an Ascend entry.
- §16.2 calling convention: "cdecl on x86/x64" — Ascend 910B uses AArch64. The spec should explicitly state that on non-x86 architectures, the platform-native C calling convention applies.
- §9: `ModelError { code: uint32 }` — no reserved code range for vendor errors. Ascend-specific security/hardware errors (memory parity error, thermal event) need to surface through ARIA without spec changes. Define a vendor range: e.g., codes ≥ 0x80000000 = vendor-defined.

---

**Norm Jouppi proxy — Google — TPUv5 / Trillium / XLA / JAX Lead**

Green:
- §13.7: ARIA's compatibility with compiled graphs is the key TPU compatibility statement.
- §2.2: multimodal and distributed inference correctly deferred. Appropriate scope.

Red:
- §4.5.1: no TPU/XLA coherence entry. XLA buffer donation is the coherence mechanism — fundamentally different from CUDA stream sync and not covered by the NPU entry.
- §16 ABI: JAX models are Python objects or XLA executables, not `dlopen`-loadable shared libraries. `aria_load_model(path)` is inapplicable to the JAX/TPU deployment model. Define an alternative loading interface or explicitly acknowledge that C ABI loading requires an interop shim layer.
- §4.3 Layout: TPU requires 128-element alignment for bfloat16. `Layout::Contiguous` is underdefined — "contiguous" doesn't specify alignment. Add an `alignment: uint32` field to `Layout::Contiguous` or define `Layout::Aligned{alignment}`.

---

**Raja Koduri proxy — Intel — Gaudi 3 / SynapseAI / AMX / Datacenter Lead**

Green:
- Appendix B Intel Gaudi section: accurate and specific. SynapseAI compilation at `declare_state()` time is correctly modeled.
- §4.2 `FloatCustom`: explicitly covers "Intel FP8 E4M3FNUZ bias variant" in Appendix B. Good precision.
- §11.1: compiled graph freedom — Gaudi's runtime dispatch of pre-compiled kernels fits cleanly.

Red:
- `declare_state()` can block for minutes on Gaudi (SynapseAI compilation for large models). The spec gives no guidance on long-running `declare_state()` calls — no timeout, no async alternative, no progress mechanism. Flag this as a known gap and specify a non-blocking `declare_state_async()` for 1.0.
- §4.5.1 table: the "synapse barrier" entry under "NPU (Qualcomm, Intel Gaudi)" uses a Gaudi-specific term (SynapseAI) in a row that also covers Qualcomm. Qualcomm doesn't use SynapseAI. Separate the entries.
- §5.1 batch with fixed-shape compiled graphs: Gaudi's SynapseAI compiles fixed-shape graphs. Variable `num_sequences` in `prefill_batch` conflicts with compiled static batch sizes. Specify that model implementations MAY pad to `max_batch` and mask unused entries, and that this is the expected pattern for compiled-graph backends.

---

**Johny Srouji proxy — Apple — M4/M4 Max / Metal / CoreML / ANE Lead**

Green:
- Appendix B Apple Silicon: thorough and accurate. CoreML Swift/ObjC bridge, Metal compute shaders, unified memory buffer allocation — correctly described.
- §4.5.1: "Metal command buffer completion event" is correct.
- §11.1: ANE can be used as a model-internal optimization. ARIA doesn't need to know.

Red:
- §16.2: "cdecl on x86/x64" — Apple Silicon is ARM64. The spec should explicitly state ARM64 calling convention applies on AArch64 platforms.
- §16 `aria_load_model()`: CoreML models are `.mlpackage` directories or `.mlmodelc` bundles, not shared libraries exporting C symbols. `dlopen/dlsym` doesn't apply. Apple's deployment model requires an alternative loading path or explicit acknowledgment of the required interop shim.
- §4.5.2 Strategy 1 overhead note: "may incur OS overhead" — on Apple Silicon with unified memory, virtual memory mapping has near-zero cost. There is no separation between CPU and GPU memory; Strategy 1 is effectively free. The warning may discourage a strategy that is optimal on unified memory architectures. Add a note: overhead concern does not apply to unified memory systems.
- §4.5.2 Strategy 2 "paging" semantics: on Apple Silicon, "paging" has OS-level meaning (swap to disk) distinct from GPU memory block management. For unified memory systems, clarify that `Layout::Paged` refers to logical sub-block addressing of the inference buffer, not OS virtual memory paging.

---

**Ziad Asghar proxy — Qualcomm — Snapdragon X Elite / QNN / Hexagon NPU / Edge Lead**

Green:
- Appendix B Qualcomm QNN: accurate. QNN graph execution, state memory via QNN memory API, FloatCustom for Qualcomm formats.
- §4.5.1: QNN completion callbacks covered under NPU entry.
- §13.7: compiled graph model fits QNN's ahead-of-time compilation.

Red:
- §16: no battery/thermal/power constraint signaling. Qualcomm's NPU operates under dynamic power budgets. A model running at full speed on H100 may run at 20% speed under thermal throttling on Snapdragon. ARIA has no mechanism for the harness to request a power-appropriate compute mode or for the model to report current performance headroom.
- §5.3 single-handle framing: §5.3 describes single-handle overloads as "for edge devices, testing, sequential workloads" — implying they are a lesser interface. On Qualcomm edge devices, single-request-at-a-time IS the production workload. The spec's datacenter bias biases adoption documentation against edge use cases. Reframe: both interfaces are first-class; use case determines which is appropriate.
- §16 `aria_load_model()`: no async/non-blocking load. On mobile devices, graph compilation can take minutes. A blocking silent load is incompatible with mobile UX requirements. Add a non-blocking load with completion callback for 1.0.
- §9: no vendor error range. Qualcomm-specific errors (TrustZone access violation, thermal throttle, model tamper detected) need to surface through ARIA. Define vendor code range: ≥ 0x80000000.

---

## STANDARDS / COMPANION INTELLIGENCE

---

**HaiberDyn Agent — HCS Expert — Companion Intelligence Architecture**

*Reviewing in both directions: ARIA → HCS impact, and HCS → ARIA requirements.*

Green (ARIA → HCS):
- §6.2 MPA: directly solves a companion deployment problem. The user selects their model (Qwen, Claude, future model) — the MPA ensures the HCS harness extracts tool invocations consistently regardless of model family. This supports HCS's user-choice principle without requiring harness changes per model.
- §10 Model Invariant 5: "model MUST NOT allocate persistent per-request state outside StateHandle" — a companion privacy property. Models that leak per-request state create cross-session data exposure vectors. ARIA's strict encapsulation is a companion sovereignty guarantee.
- §13.3: Custom tag aggregation for architectural confidentiality enables proprietary companion model deployment while remaining ARIA-conformant. Sovereign companion models can protect their architecture.

Red (ARIA → HCS gaps):
- §2.2 cancellation out of scope: user-initiated cancellation is a first-class companion UX requirement. When a user says "stop" mid-generation, the companion must cancel the in-flight ARIA call immediately. This cannot be an implementation detail — it must be a spec-level primitive for companion deployments where the human's interruption right is non-negotiable.
- §9: no session migration error. Companions maintain long-running relationships across model restarts, upgrades, and memory pressure events. ARIA provides no `StateHandle` migration mechanism. If the model instance behind a companion session is replaced, all session state is lost — there is no path from the old handle to the new model. A `StateHandle` serialization interface (export to bytes, import to new model instance) would enable companion session continuity.
- §6.2.2 ToolInvocation carries no permission context. HCS's Intent Manifest (from the IHS handshake) defines per-session permission scope. Tool invocations must be validated against this manifest before execution. Currently this validation must happen out-of-band since ToolInvocation carries no harness-attached metadata. An opaque `metadata: bytes` field (set by harness, preserved through MPA, passed with ToolInvocation) would enable IHS context attachment without ARIA needing to understand HCS.
- §4.9: no companion capability flag. `ModelInfo` has no field for declaring companion-relevant properties (identity stability, multi-turn memory integration, relational continuity). An `"aria.companion.identity_stable"` capability flag would allow registries and runtimes to surface models appropriate for companion deployment.

HCS → ARIA implications (work product for HCS repo — review before action):
1. HCS ESLA layer requires ARIA-level StateHandle serialization for cross-session state persistence. This is a new ARIA requirement surfaced from HCS that should feed back into ARIA 1.0 planning.
2. HCS IHS authentication must complete BEFORE ARIA `prefill()` is called. ARIA's design allows this (auth is above ARIA), but the spec should acknowledge pre-call identity verification as a valid companion deployment pattern — not just an implementation detail.
3. HCS companion dignity requirements extend to model-controlled error strings. ARIA's sanitization requirement (§9 `ModelError.detail`) is a step toward preventing adversarial model behavior from reaching the user. HCS should reference this requirement explicitly in the Companion Pilot specification.
4. HCS ESLA + ARIA: if ARIA adds StateHandle serialization in 1.0, HCS can define a standard for persisting the inference state alongside the memory state — unifying the physical (KV cache) and conceptual (ESLA) continuity layers.

---

## SECURITY

---

**Becky Weiss proxy — AWS — Cloud Security / Multi-Tenant Inference Lead**

Green:
- §9 `ModelError.detail` sanitization: correct, well-specified, and actionable. "Printable ASCII, no control chars, max 512 bytes" prevents log injection and header injection.
- §6.2.2 `JsonValue` depth/size limits: prevents JSON bomb attacks via adversarial tool parameters.
- §10 Runtime Invariant 11: resource limits on `declare_state()` before allocation. Prevents adversarial model exhausting the allocator.

Red:
- §16 `aria_load_model(path)`: no signature verification, checksum validation, or attestation mechanism. A supply chain attack (malicious model placed at the expected path) loads with full trust. The ABI MUST define a loading path that includes provenance verification: hash check, signature validation, or TEE attestation. This is a Critical security gap.
- §9 sanitization scope: the sanitization requirement applies to `ModelError.detail` only. But `SlotUnsupported.reason`, `SlotUnsupported.slot_id`, and `NoDtypeCompatible.slot_id` are also model-controlled strings from `StateSpec`. An adversarial model could embed injection payloads in slot IDs. Extend the sanitization requirement to ALL model-controlled strings in all `ARIAError` variants.
- §14.2 self-attestation conformance: "self-attestation is accepted in the absence of an automated test suite." For multi-tenant cloud inference handling sensitive data, self-attestation is not a sufficient trust anchor. The conformance test suite must be a gate for 0.7, not an eventual 1.0 item.

---

**Mark Russinovich proxy — Azure — Enterprise Security / Compliance / Data Residency Lead**

Green:
- §10 Runtime Invariant 7: "MUST NOT silently suppress or retry errors without consumer's knowledge." Essential for enterprise audit trails and incident response.
- §9 error taxonomy: "Raised by / Fatal / Retriable" maps directly to enterprise incident classification, SLA tiers, and retry policy automation.
- §3.1 immutability at boundaries: "consumer MUST NOT write into model-owned tensor memory" prevents a class of harness bugs that corrupt model state across tenant boundaries.

Red:
- §4.9 `ModelInfo`: no data classification or sensitivity level declaration. Enterprise deployments segment models by data classification (public, internal, confidential). ARIA gives no mechanism for a model to declare its sensitivity tier or for runtimes to enforce data residency. All classification enforcement must be external.
- §16: no audit logging hooks in the ABI. Compliance requires logging every inference call (caller, model, token count, timestamp). Add an optional `aria_audit_callback_t` function pointer in the vtable that the runtime can set to receive per-call events.
- §5.1 `PrefillRequest`/`DecodeRequest`: no caller-provided `request_id`. Enterprise billing and audit systems need to correlate inference calls to business transactions. Runtime-assigned `handle.id()` (opaque uint64) is not meaningful to external systems. Add an optional caller-provided `request_id: string` field.
- §14.2: GitHub Issue conformance process is unusable for enterprise vendors with strict open-source contribution restrictions. Provide a private attestation path.

---

**Ivan Krstić proxy — Apple — Device Security / Secure Enclave / Private Compute Lead**

Green:
- §10 Model Invariant 5: "model MUST NOT allocate persistent per-request state outside StateHandle" — critical for Private Compute isolation. A model caching user data outside ARIA-managed memory could persist across Secure Enclave boundaries.
- §3.1 encapsulation: "StateHandle is opaque. Consumer has no access to slot memory." Correct for private computation — the harness must not be able to inspect model activations.

Red:
- §16: `aria_load_model()` has no TEE context parameter. Apple's Private Compute Cloud uses Secure Enclave for model loading. Loading "from a path" without attestation context means the loaded model has no cryptographic binding to the TEE. Define a `aria_load_secure()` variant or attestation parameter.
- §9 sanitization is format-only: "printable ASCII, no control chars, max 512 bytes" prevents format attacks but not information leakage. A model could encode user data in a crafted error message that passes format sanitization. Add: model-controlled error strings MUST NOT contain request content. This is a semantic sanitization requirement in addition to the existing format requirement.
- §4.9 `model_name: string`: model identity is self-declared and unverifiable. "GPT-4" can be claimed by any model. Private compute requires verifiable model identity — a hash of model weights or a certificate. Add a `model_fingerprint: Option<bytes>` field (hash of model weights, computed at load time).

---

**Asaf Shen proxy — Qualcomm — Edge Security / TrustZone / TEE / Model IP Protection Lead**

Green:
- §13.3: Custom tag aggregation for architectural confidentiality enables proprietary model IP protection on edge devices deployed in adversarial physical environments.
- §14.3: Apache 2.0 reference implementations — critical for commercial edge deployments where GPL creates legal risk.

Red:
- §16: `aria_load_model()` has no TrustZone/secure-world parameter. Edge models often run in TrustZone secure world. The load function cannot distinguish secure from non-secure loading. Add `trust_level` field or flags.
- §10 Model Invariant 5 note: "static buffers (weight tensors) SHOULD be allocated at model load time." No guidance on memory protection for weight buffers. On edge devices, model weights in unprotected memory are IP exposure vectors. Specify: weight buffers SHOULD be allocated in protected memory on platforms where TEE or encrypted memory is available.
- §9: `ModelError { code: uint32 }` — no vendor range. Qualcomm-specific security errors (TrustZone violation, model tamper detection, thermal shutdown) have no standard channel. Define vendor range: codes ≥ 0x80000000 are vendor-defined.
- §6.2 MPA: `ToolInvocation` has no integrity protection. On edge devices without TrustZone sealing, a malicious harness could modify extracted tool parameters between `extract()` and execution. The model cannot sign its `ToolInvocations` to prove they were not tampered. Define an optional signing mechanism for `ToolInvocation` in high-security deployment profiles.

---

## RANKED FIX LIST

### Critical — spec-breaking, ABI-breaking, or security gaps

1. **C ABI doesn't implement two-level batch Result (§9 vs §16)**: `prefill_batch`/`decode_batch` return single `aria_error_t` but the spec defines `Result<List<Result<>>, E>`. The normative ABI contradicts the spec's error semantics. *(vLLM, TGI)*
2. **Strategy 2 block table structure undefined in §16**: "exposes the block table structure" is unimplementable without a concrete C representation. Define `aria_block_table_t`. *(vLLM, NVIDIA)*
3. **`aria_load_model()` has no provenance verification**: supply chain attack vector. Define a loading path with hash/signature validation. *(AWS, Apple device, Qualcomm edge)*
4. **All model-controlled strings in ARIAError require sanitization, not just `ModelError.detail`**: `SlotUnsupported.slot_id`, `NoDtypeCompatible.slot_id`, and `reason` fields are also model-controlled. *(AWS)*
5. **No OpenAI JSON-compatible adapter in required MPA set**: Qwen+Claude+Passthrough covers 2 of the 3 major ecosystems. OpenAI function-calling JSON format is the de facto standard. Add `OpenAIJsonAdapter` as a required built-in. *(OpenAI, Meta, Ollama)*

### Important — adoption blockers or significant correctness issues

6. **No CUDA stream parameter in C ABI**: real NVIDIA inference uses explicit stream management; implicit default stream blocks multi-stream overlap. *(NVIDIA)*
7. **C ABI ITokenizer incomplete**: `encode(batch)`, `decode(batch)`, `special_tokens()` missing from vtable. *(llama.cpp, TGI)*
8. **No mixed prefill+decode batch**: continuous batching runtimes launch two GPU kernel dispatches per scheduling step instead of one. Critical performance gap for production serving. *(Meta, SGLang)*
9. **MPA streaming incompatibility**: `extract(text)` requires complete response text; no incremental API for token-by-token streaming. *(xAI, Ollama)*
10. **Adapter versioning**: no mechanism for `QwenAdapter-v2`, `QwenAdapter-v3` as model formats evolve. *(Qwen, DeepSeek)*
11. **`truncate()` missing from StateHandle**: blocks beam search and resampling. Mentioned in §15 — move to 0.7 not 1.0. *(vLLM)*
12. **MPA normative status ambiguous**: CONFORMANCE.md says "implementations that include" — makes MPA optional. Clarify: required for tool-calling conformance tier, or make it a defined optional tier. *(Ollama)*
13. **No caller-provided `request_id`**: enterprise/multi-model systems cannot correlate calls without stable caller-provided identifiers. *(Azure, Cohere, LMDeploy)*
14. **Long-running `declare_state()` unaddressed**: Gaudi and Qualcomm graph compilation can take minutes. Spec gives no guidance; define non-blocking `declare_state_async()` for 1.0. *(Gaudi, Qualcomm)*
15. **StateHandle serialization for cross-session persistence (HCS requirement)**: companions need to checkpoint and restore KV state across model restarts. Flag as 1.0 requirement. *(HCS)*
16. **`ClaudeAdapter` format incorrect**: format in spec doesn't match Anthropic's actual API tool-call format. Correct or explicitly label as illustrative. *(Anthropic)*
17. **Tokenizer substitution "semantically equivalent" is unverifiable**: require normative test vectors. *(TGI)*

### Nice-to-have — quality, completeness, adoption friction

18. Structured `architecture_family` field in `ModelInfo` for tooling (Ollama, OpenAI)
19. Float8E4M3 FN vs FNUZ disambiguation in DType enum (NVIDIA)
20. `aria_load_model()` progress callback (Ollama, Qualcomm, Gaudi)
21. TPU/XLA and Ascend entries in memory coherence table (Google, Huawei)
22. Normative Chinese translation (01.AI)
23. Patch-level version support 0.x.y (01.AI, LMDeploy)
24. ARM64 calling convention explicit statement in §16.2 (Apple, Huawei)
25. MoE lazy allocation flag on `StateSlot` (DeepSeek)
26. Per-slot KV dtype negotiation for MoE (Mistral)
27. Companion capability flag in `ModelInfo` (HCS)
28. Audit logging callback in vtable (Azure)
29. `model_fingerprint` field in `ModelInfo` for cryptographic identity (Apple device)
30. `request_metadata: bytes` opaque field in `PrefillRequest` for safety/permission context (Anthropic, HCS)
