/**
 * OpenAI Web Search provider using the Responses API built-in web_search tool
 *
 * Features:
 * - Built-in web search via Responses API
 * - Source extraction using include parameter
 * - Domain filtering (allowlist)
 */

import OpenAI from "openai";
import type {
  WebSearchOptions,
  WebSearchResult,
  ModelInfo,
  WebSearchProvider,
} from "../types.js";
import { MODEL_IDS, MCPError } from "../types.js";
import { logger } from "../logger.js";
import { withRetry, withTimeout } from "../retry.js";
import { calculateCost } from "../pricing.js";
import { costTracker } from "../cost-tracker.js";

// Default timeout for web search requests (60 minutes)
const DEFAULT_TIMEOUT_MS = 60 * 60 * 1000;

// Maximum allowed domains for filtering
const MAX_ALLOWED_DOMAINS = 100;

export class OpenAIWebSearchProvider implements WebSearchProvider {
  private client: OpenAI;

  constructor(apiKey: string, orgId?: string) {
    if (!apiKey) {
      throw new Error("OpenAI API key is required");
    }
    this.client = new OpenAI({
      apiKey,
      organization: orgId,
    });
    logger.info("OpenAI web search provider initialized");
  }

  async search(options: WebSearchOptions): Promise<WebSearchResult> {
    const {
      query,
      model = "gpt-5.2",
      allowedDomains = [],
      includeSources = true,
    } = options;

    const startTime = Date.now();

    // Validate allowed domains
    if (allowedDomains.length > MAX_ALLOWED_DOMAINS) {
      throw new MCPError(
        "VALIDATION_ERROR",
        `Maximum ${MAX_ALLOWED_DOMAINS} allowed domains permitted. Got: ${allowedDomains.length}`
      );
    }

    // Validate domain format (no scheme)
    for (const domain of allowedDomains) {
      if (domain.includes("://")) {
        throw new MCPError(
          "VALIDATION_ERROR",
          `Domains must not include scheme (https://). Got: ${domain}`
        );
      }
    }

    logger.debugLog("Starting web search", {
      queryLength: query.length,
      model,
      allowedDomainsCount: allowedDomains.length,
      includeSources,
    });

    try {
      // Build web search tool configuration
      const webSearchTool: any = { type: "web_search" };

      // Add domain filtering if specified
      if (allowedDomains.length > 0) {
        webSearchTool.web_search = {
          allowed_domains: allowedDomains,
        };
      }

      // Build request options
      const requestOptions: any = {
        model: MODEL_IDS[model] || model,
        input: query,
        tools: [webSearchTool],
      };

      // Include sources in response if requested
      if (includeSources) {
        requestOptions.include = ["web_search_call.action.sources"];
      }

      logger.debugLog("Web search API request", {
        model: requestOptions.model,
        queryPreview: query.substring(0, 200) + (query.length > 200 ? "..." : ""),
        hasAllowedDomains: allowedDomains.length > 0,
        includeSources,
      });

      // Use retry wrapper for transient errors and timeout protection
      const response = await withRetry(
        () =>
          withTimeout(
            () => this.client.responses.create(requestOptions),
            DEFAULT_TIMEOUT_MS,
            "web-search"
          ),
        {
          maxRetries: 2,
          retryableErrors: ["RATE_LIMIT", "429", "503", "502", "ECONNRESET", "ETIMEDOUT"],
          context: "web-search",
        }
      );

      logger.debugLog("Web search API response", {
        model,
        hasOutputText: !!(response as any).output_text,
        responseId: (response as any).id,
      });

      // Extract text from response
      const text = (response as any).output_text || "";

      // Extract sources from response output
      const sources: Array<{ url: string; title?: string }> = [];

      if (includeSources && (response as any).output) {
        for (const item of (response as any).output) {
          if (item.type === "web_search_call" && item.action?.sources) {
            for (const source of item.action.sources) {
              sources.push({
                url: source.url,
                title: source.title,
              });
            }
          }
        }
      }

      // Get usage metadata
      const usage = (response as any).usage
        ? {
            promptTokens: (response as any).usage.input_tokens,
            completionTokens: (response as any).usage.output_tokens,
            totalTokens:
              ((response as any).usage.input_tokens || 0) +
              ((response as any).usage.output_tokens || 0),
          }
        : undefined;

      const durationMs = Date.now() - startTime;

      // Calculate cost
      const cost = calculateCost(
        model,
        usage?.promptTokens || 0,
        usage?.completionTokens || 0
      );

      // Track cumulative cost
      costTracker.trackCost({
        timestamp: new Date().toISOString(),
        model,
        operation: "web_search",
        inputCost: cost.inputCost,
        outputCost: cost.outputCost,
        totalCost: cost.totalCost,
        estimated: cost.estimated,
        promptTokens: usage?.promptTokens,
        completionTokens: usage?.completionTokens,
      });

      // Log usage statistics
      logger.logUsage({
        timestamp: new Date().toISOString(),
        provider: "openai",
        model,
        operation: "web_search",
        durationMs,
        success: true,
        metrics: usage,
      });

      return {
        text,
        model,
        usage,
        cost,
        sources: sources.length > 0 ? sources : undefined,
      };
    } catch (error: any) {
      const durationMs = Date.now() - startTime;
      let errorType = "SEARCH_ERROR";
      let errorMessage = error.message || "Unknown error during web search";

      logger.error("Web search API error", {
        model,
        errorName: error.name,
        errorMessage: error.message,
        errorStack: error.stack?.split("\n").slice(0, 5).join("\n"),
        durationMs,
      });

      // Handle specific OpenAI API errors
      if (error.message?.includes("API key") || error.message?.includes("Incorrect API key") || error.status === 401) {
        errorType = "AUTH_ERROR";
        errorMessage = "Invalid or missing OpenAI API key";
      } else if (
        error.message?.includes("quota") ||
        error.message?.includes("rate") ||
        error.status === 429
      ) {
        errorType = "RATE_LIMIT";
        errorMessage = "OpenAI API rate limit or quota exceeded. Please wait and retry.";
      } else if (error.message?.includes("TIMEOUT")) {
        errorType = "TIMEOUT";
        errorMessage = "Web search timed out. Try a simpler query.";
      }

      // Log failed usage
      logger.logUsage({
        timestamp: new Date().toISOString(),
        provider: "openai",
        model,
        operation: "web_search",
        durationMs,
        success: false,
        error: `${errorType}: ${errorMessage}`,
      });

      throw new Error(`${errorType}: ${errorMessage}`);
    }
  }

  getModelInfo(): ModelInfo {
    return {
      id: "web-search",
      name: "Web Search (GPT-5.2)",
      provider: "openai",
      type: "web-search",
      description:
        "Search the web using OpenAI's built-in web_search tool. Returns synthesized answers with source citations.",
    };
  }
}
