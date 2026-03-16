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

## Adapter Extensions

Model Protocol Adapter registrations (see §6.2) follow the format `aria.adapter.{vendor}.{family}`.

| Extension String | Description | Since |
|------------------|-------------|-------|
| `aria.adapter.qwen.qwen3` | QwenAdapter — JSON-in-XML tool-call format (`<tool_call>{...}</tool_call>`) | 0.6.1 |
| `aria.adapter.anthropic.claude` | ClaudeAdapter — nested XML tool-call format (`<function_calls><invoke name="..."><parameter name="...">...</parameter></invoke></function_calls>`) | 0.6.1 |
| `aria.adapter.openai.function` | OpenAIJsonAdapter — JSON function-calling format (`{"type":"function","function":{"name":"...","arguments":"..."}}`) | 0.6.2 |
| `aria.passthrough` | PassthroughAdapter — fallback; `extract()` always returns None | 0.6.1 |

## Companion Intelligence Extensions

Capability flags for companion and relational AI deployments (see §4.9 `ModelInfo.capabilities`).

| Extension String | Description | Since |
|------------------|-------------|-------|
| `aria.companion.identity_stable` | Model maintains consistent persona and identity characteristics across multi-turn sessions within a request sequence | 0.6.2 |
| `aria.companion.memory_integration` | Model can integrate external structured memory inputs (relationship history, user preferences) via the token context | 0.6.2 |

## Vendor Extensions

Vendor extensions must follow the format `aria.vendor.{vendor_name}.{feature}`.

*   No vendor extensions currently registered.

## Process
To register a new extension, submit a PR adding it to this file with a clear description of the behavioral contract.
