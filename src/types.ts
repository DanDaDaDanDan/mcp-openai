/**
 * Shared types for the MCP OpenAI server
 */

// ============================================================================
// Model Constants
// ============================================================================

// Text generation models (GPT-5.2 family)
export const TEXT_MODELS = ["gpt-5.2", "gpt-5.2-pro", "gpt-5.2-chat-latest"] as const;
export type TextModel = (typeof TEXT_MODELS)[number];

// Deep research models
export const DEEP_RESEARCH_MODELS = ["o3-deep-research", "o4-mini-deep-research"] as const;
export type DeepResearchModel = (typeof DEEP_RESEARCH_MODELS)[number];

// Web search models (subset of text models that support web search)
export const WEB_SEARCH_MODELS = ["gpt-5.2", "gpt-5.2-chat-latest"] as const;
export type WebSearchModel = (typeof WEB_SEARCH_MODELS)[number];

// All supported models
export const ALL_MODELS = [...TEXT_MODELS, ...DEEP_RESEARCH_MODELS] as const;

// Model ID mapping (for API calls)
export const MODEL_IDS: Record<string, string> = {
  "gpt-5.2": "gpt-5.2",
  "gpt-5.2-pro": "gpt-5.2-pro",
  "gpt-5.2-chat-latest": "gpt-5.2-chat-latest",
  "o3-deep-research": "o3-deep-research",
  "o4-mini-deep-research": "o4-mini-deep-research",
};

// Type guards
export function isTextModel(model: string): model is TextModel {
  return TEXT_MODELS.includes(model as TextModel);
}

export function isDeepResearchModel(model: string): model is DeepResearchModel {
  return DEEP_RESEARCH_MODELS.includes(model as DeepResearchModel);
}

export function isWebSearchModel(model: string): model is WebSearchModel {
  return WEB_SEARCH_MODELS.includes(model as WebSearchModel);
}

// ============================================================================
// Reasoning Effort
// ============================================================================

export const REASONING_EFFORTS = ["none", "low", "medium", "high", "xhigh"] as const;
export type ReasoningEffort = (typeof REASONING_EFFORTS)[number];

// gpt-5.2-pro only supports medium, high, xhigh
export const PRO_REASONING_EFFORTS = ["medium", "high", "xhigh"] as const;
export type ProReasoningEffort = (typeof PRO_REASONING_EFFORTS)[number];

export function isValidProReasoningEffort(effort: ReasoningEffort): effort is ProReasoningEffort {
  return PRO_REASONING_EFFORTS.includes(effort as ProReasoningEffort);
}

// ============================================================================
// Verbosity
// ============================================================================

export const VERBOSITY_LEVELS = ["low", "medium", "high"] as const;
export type VerbosityLevel = (typeof VERBOSITY_LEVELS)[number];

// ============================================================================
// Reasoning Summary
// ============================================================================

export const REASONING_SUMMARIES = ["off", "concise", "detailed"] as const;
export type ReasoningSummary = (typeof REASONING_SUMMARIES)[number];

// ============================================================================
// Input Types (Tool Parameters)
// ============================================================================

// Structured output configuration
export interface StructuredOutputSchema {
  name: string;
  description?: string;
  schema: Record<string, unknown>; // JSON Schema object
  strict?: boolean; // Default: true
}

// Models that support structured outputs
export const STRUCTURED_OUTPUT_MODELS = ["gpt-5.2", "gpt-5.2-chat-latest"] as const;
export type StructuredOutputModel = (typeof STRUCTURED_OUTPUT_MODELS)[number];

export function supportsStructuredOutput(model: string): model is StructuredOutputModel {
  return STRUCTURED_OUTPUT_MODELS.includes(model as StructuredOutputModel);
}

// Text generation options
export interface TextGenerateOptions {
  prompt: string;
  systemPrompt?: string;
  model?: TextModel;
  reasoningEffort?: ReasoningEffort;
  verbosity?: VerbosityLevel;
  reasoningSummary?: ReasoningSummary;
  maxOutputTokens?: number;
  temperature?: number; // Only allowed when reasoningEffort="none"
  jsonSchema?: StructuredOutputSchema; // For structured JSON output (not supported by gpt-5.2-pro)
}

// Web search options
export interface WebSearchOptions {
  query: string;
  model?: WebSearchModel;
  allowedDomains?: string[]; // Max 100, domain only (no scheme)
  includeSources?: boolean;
}

// Deep research options
export interface DeepResearchOptions {
  query: string;
  model?: DeepResearchModel;
  timeoutMs?: number; // Max time to wait for research completion
  pollIntervalMs?: number; // How often to check status
}

// ============================================================================
// Result Types
// ============================================================================

// Cost information returned with results
export interface CostInfo {
  inputCost: number;
  outputCost: number;
  totalCost: number;
  currency: "USD";
  estimated: boolean;
}

// Common result structure
export interface GenerateResult {
  text: string;
  model: string;
  usage?: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
    reasoningTokens?: number;
  };
  cost?: CostInfo;
}

// Web search result with sources
export interface WebSearchResult extends GenerateResult {
  sources?: Array<{
    url: string;
    title?: string;
  }>;
}

// Deep research result
export interface DeepResearchResult {
  text: string;
  model: string;
  responseId: string;
  status: "completed" | "failed";
  durationMs: number;
  usage?: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
  };
  cost?: CostInfo;
}

// ============================================================================
// Provider Interfaces
// ============================================================================

// Model information
export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  type: "text" | "web-search" | "research";
  contextWindow?: number;
  maxOutput?: number;
  supportsReasoning?: boolean;
  description: string;
}

// Provider interfaces
export interface TextProvider {
  generate(options: TextGenerateOptions): Promise<GenerateResult>;
  getModelInfo(model?: TextModel): ModelInfo;
}

export interface WebSearchProvider {
  search(options: WebSearchOptions): Promise<WebSearchResult>;
  getModelInfo(): ModelInfo;
}

export interface DeepResearchProvider {
  research(options: DeepResearchOptions): Promise<DeepResearchResult>;
  getModelInfo(model?: DeepResearchModel): ModelInfo;
}

// ============================================================================
// Error Types
// ============================================================================

export type ErrorCategory =
  | "AUTH_ERROR"
  | "RATE_LIMIT"
  | "CONTENT_BLOCKED"
  | "SAFETY_BLOCK"
  | "TIMEOUT"
  | "API_ERROR"
  | "VALIDATION_ERROR";

export class MCPError extends Error {
  constructor(
    public category: ErrorCategory,
    message: string,
    public statusCode?: number
  ) {
    super(`${category}: ${message}`);
    this.name = "MCPError";
  }
}

/**
 * Categorize an error from the OpenAI API
 */
export function categorizeError(error: unknown): MCPError {
  const message = error instanceof Error ? error.message : String(error);
  const status = (error as any)?.status || (error as any)?.statusCode;

  if (status === 401 || message.includes("API key") || message.includes("unauthorized") || message.includes("Incorrect API key")) {
    return new MCPError("AUTH_ERROR", "Invalid or missing OpenAI API key", status);
  }

  if (status === 429 || message.includes("rate") || message.includes("quota") || message.includes("too many requests")) {
    return new MCPError("RATE_LIMIT", "OpenAI API rate limit or quota exceeded. Please wait and retry.", status);
  }

  if (message.includes("safety") || message.includes("content_policy")) {
    return new MCPError("SAFETY_BLOCK", "Content was blocked by OpenAI safety filters", status);
  }

  if (message.includes("blocked") || message.includes("content policy") || message.includes("moderation")) {
    return new MCPError("CONTENT_BLOCKED", "Request blocked due to content policy", status);
  }

  if (message.includes("TIMEOUT") || message.includes("timed out")) {
    return new MCPError("TIMEOUT", message);
  }

  return new MCPError("API_ERROR", message, status);
}
