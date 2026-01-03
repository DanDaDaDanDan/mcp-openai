# mcp-openai

MCP server providing Claude Code access to OpenAI's GPT-5.2 models, web search, and deep research.

## Philosophy

1. **Fail fast** - Surface errors immediately with clear messages. Don't silently swallow failures or return partial results.
2. **Don't guess, research** - When API behavior is unclear, check the docs. Model IDs and parameters change; verify against OpenAI documentation.
3. **Eager initialization** - Create provider instances at startup. Fail at init, not use-time.
4. **Structured errors** - Categorize errors (AUTH_ERROR, RATE_LIMIT, SAFETY_BLOCK, TIMEOUT) for actionable feedback.

## SDK

Uses the `openai` npm package with the **Responses API** (the recommended primitive for new projects).

Do NOT use the deprecated Assistants API.

## Models

### Text Generation (GPT-5.2 Family)

| Friendly Name | API Model ID | Context | Max Output | Notes |
|---------------|--------------|---------|------------|-------|
| gpt-5.2 | `gpt-5.2` | 400K | 128K | Default. Best general choice for coding + agentic tasks |
| gpt-5.2-pro | `gpt-5.2-pro` | 400K | 128K | Max accuracy. Can take minutes. Reasoning: medium/high/xhigh only |
| gpt-5.2-chat-latest | `gpt-5.2-chat-latest` | 128K | 16K | ChatGPT snapshot. Use gpt-5.2 for most API work |

### Deep Research

| Friendly Name | API Model ID | Context | Max Output | Notes |
|---------------|--------------|---------|------------|-------|
| o3-deep-research | `o3-deep-research` | 200K | 100K | Best deep research. 5-30 minutes |
| o4-mini-deep-research | `o4-mini-deep-research` | 200K | 100K | Faster/cheaper deep research |

## Reasoning Configuration

### Reasoning Effort

For `gpt-5.2`, reasoning effort supports:
- `none` (default)
- `low`
- `medium`
- `high`
- `xhigh`

For `gpt-5.2-pro`, supported efforts are **only**: `medium`, `high`, `xhigh`.

### Parameter Compatibility

When reasoning effort is **not** `none`, `temperature` is **not allowed**. The server validates this and rejects incompatible combinations.

### Verbosity Control

```typescript
text: { verbosity: "low" | "medium" | "high" }
```

Controls output length. Works alongside reasoning effort.

### Reasoning Summaries

```typescript
reasoning: { summary: "concise" | "detailed" }
```

Optional user-visible reasoning summaries.

## Structured Outputs

Request JSON output conforming to a JSON Schema. Only supported by `gpt-5.2` and `gpt-5.2-chat-latest` (NOT `gpt-5.2-pro`).

### Configuration

```typescript
text: {
  format: {
    type: "json_schema",
    name: "extract_entities",
    strict: true,  // Default
    schema: {
      type: "object",
      properties: {
        entities: {
          type: "array",
          items: { type: "string" }
        }
      },
      required: ["entities"],
      additionalProperties: false
    }
  }
}
```

### Tool Parameter

```typescript
json_schema: {
  name: "schema_name",       // Required
  description: "...",        // Optional
  schema: { /* JSON Schema */ },  // Required
  strict: true               // Default: true
}
```

### Best Practices

1. Use `additionalProperties: false` in schemas for strict validation
2. Mark all required fields in the `required` array
3. Keep schemas simple for reliable output
4. Do NOT use with `gpt-5.2-pro` (will return validation error)

## Architecture

```
src/
├── index.ts              # MCP server, tool routing
├── types.ts              # Shared types, model constants
├── logger.ts             # Logging (stderr + optional file)
├── retry.ts              # Exponential backoff, timeout wrapper
└── providers/
    ├── text-provider.ts       # GPT-5.2 text generation
    ├── web-search-provider.ts # Web search via built-in tool
    └── deep-research-provider.ts  # Deep research (async polling)
```

## API Usage

### Responses API (text generation)

```typescript
const response = await client.responses.create({
  model: "gpt-5.2",
  input: "Your prompt here",
  reasoning: { effort: "high" },  // optional
  text: { verbosity: "medium" },  // optional
  max_output_tokens: 8192,
});

console.log(response.output_text);
```

### Web Search

```typescript
const response = await client.responses.create({
  model: "gpt-5.2",
  input: "What changed recently?",
  tools: [{ type: "web_search" }],
  include: ["web_search_call.action.sources"],
});
```

### Deep Research (background mode)

```typescript
// Start research
const response = await client.responses.create({
  model: "o3-deep-research",
  input: [...],
  tools: [{ type: "web_search_preview" }],
  background: true,
});

// Poll for completion
const result = await client.responses.retrieve(response.id);
```

## Tools

| Tool | Description | Models |
|------|-------------|--------|
| `generate_text` | Text generation with reasoning controls and structured outputs | gpt-5.2, gpt-5.2-pro, gpt-5.2-chat-latest |
| `web_search` | Web search with source citations | gpt-5.2, gpt-5.2-chat-latest |
| `deep_research` | Autonomous web research (5-30 min) | o3-deep-research, o4-mini-deep-research |
| `list_models` | List available models | Static |

## Error Categories

| Category | HTTP Status | Meaning |
|----------|-------------|---------|
| AUTH_ERROR | 401 | Invalid or missing API key |
| RATE_LIMIT | 429 | API quota exceeded |
| SAFETY_BLOCK | 400 | Blocked by OpenAI safety filters |
| CONTENT_BLOCKED | 400 | Content policy violation |
| TIMEOUT | - | Request exceeded timeout |
| VALIDATION_ERROR | 422 | Invalid input parameters |
| API_ERROR | 4xx/5xx | Other API errors |

## Environment Variables

- `OPENAI_API_KEY` (required) - from https://platform.openai.com/api-keys
- `OPENAI_ORG_ID` (optional) - organization ID
- `MCP_DEBUG` - debug logging enabled by default; set to "false" to disable
- `MCP_LOG_DIR` - defaults to `./logs`; set to "none" to disable file logging

## Development

```bash
npm install       # Install dependencies
npm run build     # Compile TypeScript
npm run dev       # Watch mode
npm start         # Run server
```

## Testing Changes

After modifying providers, verify:
1. Build succeeds: `npm run build`
2. Model IDs are current (check OpenAI docs)
3. Error categories match API responses
4. Reasoning effort validation works correctly

## Adding New Models

1. Add to model constants in `types.ts`
2. Add model ID mapping
3. Add `getModelInfo()` entry in provider
4. Update `list_models` handler if needed
5. Check if model needs special parameter handling
