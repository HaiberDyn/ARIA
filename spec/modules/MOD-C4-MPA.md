<!-- ARIA module: MOD-C4-MPA.md — do not edit directly; edit this file in spec/modules/ -->
## 6.2 Model Protocol Adapter

The `IModelAdapter` interface extends ARIA into the agentic layer — one level above the
inference contract defined in §5. It addresses the same structural coupling problem at the
tool-calling format boundary that ARIA addresses at the state management boundary.

**Normative status**: The Model Protocol Adapter layer is a **required component** for any
implementation claiming ARIA conformance for tool-calling models. Implementations that
serve only models without tool-calling capabilities (no `extract()` usage in their
deployment) MAY omit the MPA. The `CONFORMANCE.md` file specifies the exact requirements
under the "Model Protocol Adapter Conformance (§6.2)" section. Implementations MUST NOT
claim full ARIA 0.6.2-draft conformance while omitting the MPA if they serve
tool-calling-capable models.

### 6.2.1 The Format Coupling Problem

Models that support tool calling emit tool invocations in model-family-specific formats.
Every harness that routes these outputs to downstream executors must independently implement
a format shim per model family:

```
Qwen:    <tool_call>{"name":"search","arguments":{"query":"..."}}</tool_call>
Claude:  <function_calls><invoke name="search"><parameter name="query">...</parameter></invoke></function_calls>
OpenAI:  {"type":"function","function":{"name":"search","arguments":"{\"query\":\"...\"}"}}
Mistral: [TOOL_CALLS] [{"name":"search","arguments":{"query":"..."}}]
Future:  [model-specific format]
```

Without a canonical normalization layer, adding a new model family requires changes to
every harness — the same O(M×N) coupling problem ARIA eliminates at the state management
layer.

### 6.2.2 ToolInvocation

```
struct ToolInvocation {
    tool_name:       String                  // canonical tool name extracted from native format
    parameters:      Map<String, JsonValue>  // extracted key-value arguments; see below
    raw_text_before: String                  // model-generated text preceding the tool call (MAY be empty)
    raw_text_after:  String                  // model-generated text following the tool call (MAY be empty)
}

// JsonValue: a JSON-compatible value type. Implementations MUST restrict to:
//   string | number | boolean | null | array of JsonValue | object (Map<String, JsonValue>)
// Implementations MUST enforce:
//   - Maximum nesting depth: 16 levels
//   - Maximum total serialized parameter size: 64KB
// Values exceeding these limits MUST cause extract() to return None.
type JsonValue = string | number | boolean | null | List<JsonValue> | Map<String, JsonValue>
```

`ToolInvocation` is a normalization type only. It represents a parsed, format-agnostic
tool invocation. Execution semantics are outside ARIA's scope (see §6.2.6).

### 6.2.3 IModelAdapter

```
interface IModelAdapter {

    // Extract a tool invocation from the model's raw generation output.
    // Returns Some(ToolInvocation) if a well-formed tool call is present.
    // Returns None if no tool call is detected.
    // MUST NOT raise an error for unrecognized or malformed content — return None.
    extract(text: String) -> Option<ToolInvocation>

    // The adapter's model family identifier.
    // MUST match the prefix registered via AdapterRegistry.register().
    // Format: "{vendor}.{family}" — e.g., "qwen.qwen3", "anthropic.claude"
    model_family() -> String

}
```

### 6.2.4 AdapterRegistry

```
interface AdapterRegistry {

    // Register an adapter for a model name prefix.
    // prefix: case-insensitive model name prefix — e.g., "Qwen3", "claude-"
    // Duplicate registrations for the SAME prefix mapping to a DIFFERENT adapter MUST
    // return an error. Idempotent re-registration of the same prefix→adapter pair is
    // implementation-defined (MAY succeed silently).
    //
    // Versioning: use specific prefixes to distinguish format versions. For example,
    // register "Qwen3" for Qwen3 format and "Qwen2" for Qwen2 format. Longest-match
    // resolution ensures "Qwen3-14B" selects "Qwen3" over "Qwen". Implementors SHOULD
    // prefer version-specific prefixes over generic family prefixes when formats differ.
    register(prefix: String, adapter: IModelAdapter) -> Result<void, ARIAError>

    // Resolve the adapter for a given model name.
    // Matches registered prefixes case-insensitively; longest match wins.
    // If no prefix matches, MUST return PassthroughAdapter — MUST NOT return an error.
    resolve(model_name: String) -> IModelAdapter

}
```

`resolve()` MUST always return an adapter. Unknown model names return `PassthroughAdapter`
(§6.2.5), not an error. On ambiguous prefix matches, the longest matching prefix wins.

### 6.2.5 Required Built-In Adapters

Conforming implementations MUST include the following four adapters:

**QwenAdapter** — Qwen model family (JSON-in-XML format):
```
Input:   <tool_call>{"name":"search","arguments":{"query":"example"}}</tool_call>
Output:  ToolInvocation { tool_name: "search", parameters: {"query": "example"}, ... }
```
Default registration prefix: `"Qwen"` (case-insensitive)

**ClaudeAdapter** — Anthropic Claude family (nested XML format):
```
Input:
  <function_calls>
  <invoke name="search">
  <parameter name="query">example query</parameter>
  </invoke>
  </function_calls>

Output:  ToolInvocation { tool_name: "search", parameters: {"query": "example query"}, ... }
```
Default registration prefix: `"claude"` (case-insensitive)

**OpenAIJsonAdapter** — OpenAI function-calling format and compatible models (JSON object):
```
Input:   {"type":"function","function":{"name":"search","arguments":"{\"query\":\"example\"}"}}
Output:  ToolInvocation { tool_name: "search", parameters: {"query": "example"}, ... }
```
The `arguments` field is a JSON-encoded string; the adapter MUST parse it. This format is
used by GPT-4/o, LLaMA tool-use fine-tunes, and any model declaring OpenAI API compatibility.
Default registration prefix: `"gpt"` (case-insensitive); implementors SHOULD also register
common compatible prefixes (`"llama"`, `"meta-llama"`, `"mistral"`) or allow applications
to register them via `AdapterRegistry.register()`.

**PassthroughAdapter** — fallback for unregistered model families:
- `extract()` MUST return `None` for all inputs
- `model_family()` MUST return `"aria.passthrough"`
- Returned by `AdapterRegistry.resolve()` when no registered prefix matches

The PassthroughAdapter ensures that harnesses serving novel or fine-tuned models degrade
gracefully: the raw generation text passes through unchanged rather than causing a
registry error.

**Streaming limitation**: `extract(text: String)` requires the complete generation text.
For streaming generation, the harness MUST buffer the full response before calling
`extract()`. An incremental `extract_streaming(token: String)` API for token-by-token
detection of tool-call boundaries is planned for ARIA 1.0. Until then, implementations
that need low-latency streaming SHOULD use heuristic prefix detection to identify
tool-call start markers and begin buffering at that point.

### 6.2.6 Rationale

**Same structural problem, one layer up.** The O(M×N) coupling that ARIA eliminates at the
inference boundary reappears at the tool-call format boundary. `IModelAdapter` applies the
same solution: one adapter per model family, registered once, reused across all harnesses.

**Execution is out of scope.** `ToolInvocation` carries a parsed representation, not an
execution directive. What happens after extraction — executor dispatch, parameter
validation, result injection — is the harness's domain. ARIA standardizes extraction and
canonical representation only.

**PassthroughAdapter, not error.** Returning `None` from an unknown family's `extract()`
is correct behavior. The harness receives the raw text and can handle it as it sees fit.
An error would break harnesses serving models whose names don't match any registered prefix.

**Vendor extensions.** Additional adapters MAY be registered using the
`aria.adapter.{vendor}.{family}` capability identifier. See `spec/EXTENSIONS.md` for the
registry entry format.

---
