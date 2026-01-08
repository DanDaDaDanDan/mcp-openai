#!/usr/bin/env node

/**
 * MCP Server: mcp-openai
 *
 * Provides text generation, web search, and deep research capabilities
 * using OpenAI's GPT-5.2 family and deep research models.
 *
 * Models:
 *   - gpt-5.2: Best general choice for coding + agentic tasks (default)
 *   - gpt-5.2-pro: Max accuracy for hardest problems
 *   - gpt-5.2-chat-latest: ChatGPT snapshot testing
 *   - o3-deep-research: Best deep research specialist
 *   - o4-mini-deep-research: Faster/cheaper deep research
 *
 * Tools:
 *   - generate_text: Generate text using GPT-5.2 family with reasoning controls
 *   - web_search: Search the web with source citations
 *   - deep_research: Long-form research using deep research models
 *   - list_models: List available models and their capabilities
 *
 * Environment Variables:
 *   - OPENAI_API_KEY: Required for all model access
 *   - OPENAI_ORG_ID: Optional organization ID
 *   - MCP_DEBUG: Set to "false" to disable debug logging (default: true)
 *   - MCP_LOG_DIR: Directory for log files (default: ./logs, set to "none" to disable)
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type CallToolRequest,
} from "@modelcontextprotocol/sdk/types.js";
import { OpenAITextProvider } from "./providers/text-provider.js";
import { OpenAIWebSearchProvider } from "./providers/web-search-provider.js";
import { OpenAIDeepResearchProvider } from "./providers/deep-research-provider.js";
import {
  TEXT_MODELS,
  WEB_SEARCH_MODELS,
  DEEP_RESEARCH_MODELS,
  REASONING_EFFORTS,
  VERBOSITY_LEVELS,
  REASONING_SUMMARIES,
  isTextModel,
  isWebSearchModel,
  isDeepResearchModel,
  type TextModel,
  type WebSearchModel,
  type DeepResearchModel,
  type ReasoningEffort,
  type VerbosityLevel,
  type ReasoningSummary,
} from "./types.js";
import { logger } from "./logger.js";
import { costTracker } from "./cost-tracker.js";

// Configuration from environment - fail fast if missing
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  const errorMsg =
    "FATAL: OPENAI_API_KEY environment variable is required. " +
    "Set it in your MCP server configuration or export it in your shell.";
  logger.error(errorMsg);
  console.error(errorMsg); // Also to stderr for immediate visibility
  process.exit(1);
}

const OPENAI_ORG_ID = process.env.OPENAI_ORG_ID;

// Initialize providers eagerly at startup - fail fast
const textProvider = new OpenAITextProvider(OPENAI_API_KEY, OPENAI_ORG_ID);
const webSearchProvider = new OpenAIWebSearchProvider(OPENAI_API_KEY, OPENAI_ORG_ID);
const deepResearchProvider = new OpenAIDeepResearchProvider(OPENAI_API_KEY, OPENAI_ORG_ID);

// Create MCP server
const server = new Server(
  {
    name: "mcp-openai",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Tool definitions
const TOOLS = [
  {
    name: "generate_text",
    description:
      "Generate text using GPT-5.2 family with optional reasoning + verbosity controls. " +
      "Use this for complex reasoning, writing, analysis, or any text generation task.",
    inputSchema: {
      type: "object" as const,
      properties: {
        prompt: {
          type: "string",
          description: "The complete prompt to send to the model, including all necessary context",
        },
        system_prompt: {
          type: "string",
          description:
            "Optional system instructions that set the model's behavior and role (e.g., 'You are a professional writer')",
        },
        model: {
          type: "string",
          enum: [...TEXT_MODELS],
          description:
            "Model to use: 'gpt-5.2' (default, best general), 'gpt-5.2-pro' (max accuracy, slow), 'gpt-5.2-chat-latest' (ChatGPT snapshot)",
          default: "gpt-5.2",
        },
        reasoning_effort: {
          type: "string",
          enum: [...REASONING_EFFORTS],
          description:
            "Reasoning depth: 'none' (default), 'low', 'medium', 'high', 'xhigh'. Note: gpt-5.2-pro only supports medium/high/xhigh.",
          default: "none",
        },
        verbosity: {
          type: "string",
          enum: [...VERBOSITY_LEVELS],
          description: "Output verbosity: 'low', 'medium' (default), 'high'",
          default: "medium",
        },
        reasoning_summary: {
          type: "string",
          enum: [...REASONING_SUMMARIES],
          description:
            "Optional user-visible reasoning summary: 'off' (default), 'concise', 'detailed'",
          default: "off",
        },
        max_output_tokens: {
          type: "number",
          description: "Maximum number of tokens to generate (default: 8192)",
          default: 8192,
        },
        temperature: {
          type: "number",
          description:
            "Sampling temperature from 0 to 2. Only allowed when reasoning_effort='none'. Higher = more creative.",
          minimum: 0,
          maximum: 2,
        },
        json_schema: {
          type: "object",
          description:
            "Structured output configuration. Only supported by gpt-5.2 and gpt-5.2-chat-latest (NOT gpt-5.2-pro). " +
            "When provided, the model will output valid JSON matching the schema.",
          properties: {
            name: {
              type: "string",
              description: "Name for the schema (e.g., 'extract_entities', 'analyze_sentiment')",
            },
            description: {
              type: "string",
              description: "Optional description of what the schema represents",
            },
            schema: {
              type: "object",
              description: "JSON Schema object defining the output structure",
            },
            strict: {
              type: "boolean",
              description: "Enable strict schema validation (default: true)",
              default: true,
            },
          },
          required: ["name", "schema"],
        },
      },
      required: ["prompt"],
    },
  },
  {
    name: "web_search",
    description:
      "Search the web using OpenAI's built-in web_search tool (Responses API). " +
      "Returns synthesized answers with source citations.",
    inputSchema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description: "The search query",
        },
        model: {
          type: "string",
          enum: [...WEB_SEARCH_MODELS],
          description: "Model to use: 'gpt-5.2' (default), 'gpt-5.2-chat-latest'",
          default: "gpt-5.2",
        },
        allowed_domains: {
          type: "array",
          items: { type: "string" },
          description:
            "Domain allowlist for filtering results (no scheme, max 100). Example: ['wikipedia.org', 'github.com']",
        },
        include_sources: {
          type: "boolean",
          description: "Include source URLs in response (default: true)",
          default: true,
        },
      },
      required: ["query"],
    },
  },
  {
    name: "deep_research",
    description:
      "Perform comprehensive research using OpenAI's deep research models. " +
      "The agent searches the web, analyzes multiple sources, and produces detailed research reports. " +
      "This is a long-running operation that typically takes 5-60 minutes to complete. " +
      "If it times out, use check_research with the returned response_id to retrieve results.",
    inputSchema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description:
            "The research question or topic to investigate. Be specific and detailed for best results.",
        },
        model: {
          type: "string",
          enum: [...DEEP_RESEARCH_MODELS],
          description:
            "Model to use: 'o3-deep-research' (default, most thorough), 'o4-mini-deep-research' (faster)",
          default: "o3-deep-research",
        },
        timeout_minutes: {
          type: "number",
          description:
            "Maximum time to wait for research completion in minutes (default: 60, max: 120)",
          default: 60,
          minimum: 5,
          maximum: 120,
        },
      },
      required: ["query"],
    },
  },
  {
    name: "check_research",
    description:
      "Check the status of a running deep research task or retrieve results after a timeout. " +
      "Use this with the response_id returned from deep_research if it times out or to poll for completion.",
    inputSchema: {
      type: "object" as const,
      properties: {
        response_id: {
          type: "string",
          description: "The response ID returned from a previous deep_research call",
        },
      },
      required: ["response_id"],
    },
  },
  {
    name: "list_models",
    description: "List all available OpenAI models and their capabilities",
    inputSchema: {
      type: "object" as const,
      properties: {},
      required: [],
    },
  },
  {
    name: "get_cost_summary",
    description:
      "Get cumulative cost summary for all API calls made during this session. " +
      "Shows total costs broken down by model and operation type.",
    inputSchema: {
      type: "object" as const,
      properties: {
        reset: {
          type: "boolean",
          description: "If true, reset the cost counter after returning summary",
          default: false,
        },
      },
      required: [],
    },
  },
];

// Handle list tools request
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOLS };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest) => {
  const { name, arguments: args } = request.params;

  // List models tool
  if (name === "list_models") {
    const models = [];

    // Add text models
    for (const model of TEXT_MODELS) {
      models.push({
        ...textProvider.getModelInfo(model),
        available: true,
      });
    }

    // Add web search model info
    models.push({
      ...webSearchProvider.getModelInfo(),
      available: true,
    });

    // Add deep research models
    for (const model of DEEP_RESEARCH_MODELS) {
      models.push({
        ...deepResearchProvider.getModelInfo(model),
        available: true,
      });
    }

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({ models }, null, 2),
        },
      ],
    };
  }

  // Generate text tool
  if (name === "generate_text") {
    const {
      prompt,
      system_prompt: systemPrompt,
      model,
      reasoning_effort: reasoningEffort,
      verbosity,
      reasoning_summary: reasoningSummary,
      max_output_tokens: maxOutputTokens,
      temperature,
      json_schema: jsonSchema,
    } = args as {
      prompt: string;
      system_prompt?: string;
      model?: string;
      reasoning_effort?: string;
      verbosity?: string;
      reasoning_summary?: string;
      max_output_tokens?: number;
      temperature?: number;
      json_schema?: {
        name: string;
        description?: string;
        schema: Record<string, unknown>;
        strict?: boolean;
      };
    };

    // Validate prompt
    if (!prompt || prompt.trim().length === 0) {
      return {
        content: [
          {
            type: "text",
            text: "Error: Prompt cannot be empty",
          },
        ],
        isError: true,
      };
    }

    // Validate model if provided
    if (model && !isTextModel(model)) {
      return {
        content: [
          {
            type: "text",
            text: `Error: Unknown text model "${model}". Supported models: ${TEXT_MODELS.join(", ")}`,
          },
        ],
        isError: true,
      };
    }

    try {
      const result = await textProvider.generate({
        prompt,
        systemPrompt,
        model: model as TextModel | undefined,
        reasoningEffort: reasoningEffort as ReasoningEffort | undefined,
        verbosity: verbosity as VerbosityLevel | undefined,
        reasoningSummary: reasoningSummary as ReasoningSummary | undefined,
        maxOutputTokens,
        temperature,
        jsonSchema,
      });

      // Return successful result
      return {
        content: [
          {
            type: "text",
            text: result.text || "",
          },
        ],
        // Include metadata about the generation
        _meta: {
          model: result.model,
          usage: result.usage,
          cost: result.cost,
        },
      };
    } catch (error: any) {
      const errorMessage = error.message || "Unknown error during generation";
      logger.error("Text generation failed", { error: errorMessage });

      return {
        content: [
          {
            type: "text",
            text: `Error: ${errorMessage}`,
          },
        ],
        isError: true,
      };
    }
  }

  // Web search tool
  if (name === "web_search") {
    const {
      query,
      model,
      allowed_domains: allowedDomains,
      include_sources: includeSources,
    } = args as {
      query: string;
      model?: string;
      allowed_domains?: string[];
      include_sources?: boolean;
    };

    // Validate query
    if (!query || query.trim().length === 0) {
      return {
        content: [
          {
            type: "text",
            text: "Error: Query cannot be empty",
          },
        ],
        isError: true,
      };
    }

    // Validate model if provided
    if (model && !isWebSearchModel(model)) {
      return {
        content: [
          {
            type: "text",
            text: `Error: Unknown web search model "${model}". Supported models: ${WEB_SEARCH_MODELS.join(", ")}`,
          },
        ],
        isError: true,
      };
    }

    try {
      const result = await webSearchProvider.search({
        query,
        model: model as WebSearchModel | undefined,
        allowedDomains,
        includeSources,
      });

      // Format response with sources if available
      let responseText = result.text;
      if (result.sources && result.sources.length > 0) {
        responseText += "\n\n---\n**Sources:**\n";
        for (const source of result.sources) {
          responseText += `- [${source.title || source.url}](${source.url})\n`;
        }
      }

      return {
        content: [
          {
            type: "text",
            text: responseText,
          },
        ],
        _meta: {
          model: result.model,
          usage: result.usage,
          cost: result.cost,
          sourceCount: result.sources?.length || 0,
        },
      };
    } catch (error: any) {
      const errorMessage = error.message || "Unknown error during web search";
      logger.error("Web search failed", { error: errorMessage });

      return {
        content: [
          {
            type: "text",
            text: `Error: ${errorMessage}`,
          },
        ],
        isError: true,
      };
    }
  }

  // Deep research tool
  if (name === "deep_research") {
    const {
      query,
      model,
      timeout_minutes: timeoutMinutes,
    } = args as {
      query: string;
      model?: string;
      timeout_minutes?: number;
    };

    // Validate query
    if (!query || query.trim().length === 0) {
      return {
        content: [
          {
            type: "text",
            text: "Error: Query cannot be empty",
          },
        ],
        isError: true,
      };
    }

    // Validate model if provided
    if (model && !isDeepResearchModel(model)) {
      return {
        content: [
          {
            type: "text",
            text: `Error: Unknown deep research model "${model}". Supported models: ${DEEP_RESEARCH_MODELS.join(", ")}`,
          },
        ],
        isError: true,
      };
    }

    // Convert timeout from minutes to milliseconds (default 60 minutes)
    const timeoutMs = (timeoutMinutes || 60) * 60 * 1000;

    try {
      logger.info("Starting deep research", {
        queryLength: query.length,
        model: model || "o3-deep-research",
        timeoutMinutes: timeoutMinutes || 60,
      });

      const result = await deepResearchProvider.research({
        query,
        model: model as DeepResearchModel | undefined,
        timeoutMs,
      });

      // Return successful result
      return {
        content: [
          {
            type: "text",
            text: result.text,
          },
        ],
        _meta: {
          model: result.model,
          responseId: result.responseId,
          durationMs: result.durationMs,
          durationMinutes: Math.round((result.durationMs / 1000 / 60) * 10) / 10,
          usage: result.usage,
          cost: result.cost,
        },
      };
    } catch (error: any) {
      const errorMessage = error.message || "Unknown error during deep research";
      logger.error("Deep research failed", { error: errorMessage });

      return {
        content: [
          {
            type: "text",
            text: `Error: ${errorMessage}`,
          },
        ],
        isError: true,
      };
    }
  }

  // Check research status tool
  if (name === "check_research") {
    const { response_id: responseId } = args as {
      response_id: string;
    };

    // Validate response ID
    if (!responseId || responseId.trim().length === 0) {
      return {
        content: [
          {
            type: "text",
            text: "Error: response_id is required",
          },
        ],
        isError: true,
      };
    }

    try {
      logger.info("Checking research status", { responseId });

      const result = await deepResearchProvider.checkResearch(responseId);

      // Return result (could be completed, in_progress, or failed)
      return {
        content: [
          {
            type: "text",
            text: result.text,
          },
        ],
        _meta: {
          model: result.model,
          responseId: result.responseId,
          status: result.status,
          durationMs: result.durationMs,
          usage: result.usage,
        },
      };
    } catch (error: any) {
      const errorMessage = error.message || "Unknown error checking research status";
      logger.error("Check research failed", { error: errorMessage, responseId });

      return {
        content: [
          {
            type: "text",
            text: `Error: ${errorMessage}`,
          },
        ],
        isError: true,
      };
    }
  }

  // Get cost summary tool
  if (name === "get_cost_summary") {
    const { reset } = args as { reset?: boolean };

    const summary = costTracker.getSummary();

    if (reset) {
      costTracker.reset();
    }

    // Format costs for display
    const formatCost = (n: number) => `$${n.toFixed(6)}`;

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(
            {
              summary: {
                totalCost: formatCost(summary.totalCost),
                callCount: summary.callCount,
                estimatedCosts: formatCost(summary.estimatedCosts),
                since: summary.since,
                byModel: Object.fromEntries(
                  Object.entries(summary.byModel).map(([k, v]) => [k, formatCost(v)])
                ),
                byOperation: Object.fromEntries(
                  Object.entries(summary.byOperation).map(([k, v]) => [k, formatCost(v)])
                ),
              },
              wasReset: reset || false,
            },
            null,
            2
          ),
        },
      ],
    };
  }

  // Unknown tool
  return {
    content: [
      {
        type: "text",
        text: `Error: Unknown tool "${name}"`,
      },
    ],
    isError: true,
  };
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();

  // Log startup
  logger.info("Starting MCP server", {
    version: "1.0.0",
    openaiConfigured: !!OPENAI_API_KEY,
    orgId: OPENAI_ORG_ID ? "configured" : "not set",
    debugMode: process.env.MCP_DEBUG !== "false",
    logDir: process.env.MCP_LOG_DIR || "logs",
  });

  await server.connect(transport);

  logger.info("Server running and ready for connections");
}

main().catch((error) => {
  logger.error("Fatal error", { error: error.message });
  process.exit(1);
});
