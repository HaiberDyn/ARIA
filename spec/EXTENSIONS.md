# ARIA Standard Extensions Registry

This document defines the official registry of extension strings used in:
1.  `ModelInfo.capabilities` (feature flags)
2.  `StateSlot` with `SemanticTag::Custom` (custom state roles)

Extensions allow models to declare optional features or behavioral modifications without breaking the core spec.

## Core Extensions (Reserved)

| Extension String | Description | Since |
|------------------|-------------|-------|
| `aria.core.context_shift` | Model supports modifying past KV slots (e.g., sliding window, RoPE scaling). Runtime MAY allow overwrites. | 0.5 |
| `aria.core.layout.paged` | Model supports `Layout::Paged` memory. Runtime MAY provide non-contiguous KV blocks. | 0.5 |
| `aria.core.layout.mamba` | Model supports `SSMState` slots. | 0.5 |

## Vendor Extensions

Vendor extensions must follow the format `aria.vendor.{vendor_name}.{feature}`.

*   No vendor extensions currently registered.

## Process
To register a new extension, submit a PR adding it to this file with a clear description of the behavioral contract.
