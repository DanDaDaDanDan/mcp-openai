# mcp-openai

MCP server for Claude Code providing access to OpenAI's GPT-5.2 models, web search, and deep research capabilities.

## Features

- **Text Generation**: GPT-5.2 family with reasoning effort, verbosity controls, and structured JSON outputs
- **Web Search**: Built-in web search with source citations
- **Deep Research**: Long-form research using o3/o4-mini deep research models

## Installation

```bash
npm install
npm run build
```

## Configuration

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
OPENAI_ORG_ID=org-...
MCP_DEBUG=true
MCP_LOG_DIR=./logs
```

### Claude Code MCP Settings

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "mcp-openai": {
      "command": "node",
      "args": ["D:\\Personal\\mcp-openai\\dist\\index.js"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## Available Tools

### generate_text

Generate text using GPT-5.2 family with optional reasoning controls.

**Parameters:**
- `prompt` (required): The prompt to send
- `system_prompt`: Optional system instructions
- `model`: `gpt-5.2` (default), `gpt-5.2-pro`, `gpt-5.2-chat-latest`
- `reasoning_effort`: `none` (default), `low`, `medium`, `high`, `xhigh`
- `verbosity`: `low`, `medium` (default), `high`
- `reasoning_summary`: `off` (default), `concise`, `detailed`
- `max_output_tokens`: Default 8192
- `temperature`: Only when reasoning_effort=none
- `json_schema`: Structured output configuration (gpt-5.2 and gpt-5.2-chat-latest only)
  - `name`: Schema name (required)
  - `description`: Optional description
  - `schema`: JSON Schema object (required)
  - `strict`: Enable strict validation (default: true)

### web_search

Search the web with source citations.

**Parameters:**
- `query` (required): Search query
- `model`: `gpt-5.2` (default), `gpt-5.2-chat-latest`
- `allowed_domains`: Optional domain allowlist (max 100)
- `include_sources`: Include source URLs (default: true)

### deep_research

Perform comprehensive research (5-30 minutes).

**Parameters:**
- `query` (required): Research question
- `model`: `o3-deep-research` (default), `o4-mini-deep-research`
- `timeout_minutes`: Default 60, max 60

### list_models

List all available models and their capabilities.

## Supported Models

### Text Generation
| Model | Description |
|-------|-------------|
| `gpt-5.2` | Best general choice (400K context) |
| `gpt-5.2-pro` | Max accuracy, slower (400K context) |
| `gpt-5.2-chat-latest` | ChatGPT snapshot (128K context) |

### Deep Research
| Model | Description |
|-------|-------------|
| `o3-deep-research` | Most thorough research |
| `o4-mini-deep-research` | Faster, more affordable |

## Development

```bash
npm install       # Install dependencies
npm run build     # Compile TypeScript
npm run dev       # Watch mode
npm start         # Run server
```

## License

MIT
