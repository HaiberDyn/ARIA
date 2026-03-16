<!-- ARIA module: MOD-C5-NEGOTIATION.md — do not edit directly; edit this file in spec/modules/ -->
## 8. KV Dtype Negotiation

The model declares `supported_kv_dtypes` in `ModelInfo` (ordered by preference) and
declares a `dtype` on each `AttentionKV` slot in `StateSpec`.

The runtime selects an actual allocation dtype from `supported_kv_dtypes` and allocates
`AttentionKV` slots at that dtype. The runtime communicates the selected dtype to the
model via the `TensorView.dtype` property of the slot accessor.

**Rules:**
1. The runtime MUST select a dtype from `ModelInfo.supported_kv_dtypes`. If the list
   is empty, the runtime MUST use the `StateSlot.dtype` as declared.
2. The model MUST handle any dtype in its `supported_kv_dtypes` list. If it cannot,
   that dtype MUST NOT appear in the list.
3. If the runtime cannot allocate at any dtype in `supported_kv_dtypes`, it MUST
   return `NoDtypeCompatible`.
4. `SSMState` and `Custom` slots are allocated at their declared `dtype`. Negotiation
   does not apply to non-KV slots.

**Example (llama.cpp KV quantization)**:
```
# Model declares:
supported_kv_dtypes: [Float16, Float8E4M3, Int8]

# Runtime (llama.cpp, user requested -ctk q8_0) allocates KV at Int8.
# Model checks handle.slot("kv_k_0").dtype == Int8, dequantizes appropriately.
```

---
