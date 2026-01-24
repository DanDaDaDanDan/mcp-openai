/**
 * OpenAI GPT-5.2 text provider using the Responses API
 *
 * Supports:
 * - gpt-5.2 (default): Best general choice for coding + agentic tasks
 * - gpt-5.2-pro: Max accuracy, supports reasoning effort medium/high/xhigh only
 * - gpt-5.2-chat-latest: ChatGPT snapshot testing
 *
 * Features:
 * - Reasoning effort control (none, low, medium, high, xhigh)
 * - Verbosity control (low, medium, high)
 * - Reasoning summaries (concise, detailed)
 */

import OpenAI from "openai";
import type {
  TextGenerateOptions,
  GenerateResult,
  ModelInfo,
  TextProvider,
  TextModel,
} from "../types.js";
import {
  TEXT_MODELS,
  MODEL_IDS,
  isValidProReasoningEffort,
  supportsStructuredOutput,
  MCPError,
} from "../types.js";
import { logger } from "../logger.js";
import { withRetry, withTimeout } from "../retry.js";
import { calculateCost } from "../pricing.js";
import { costTracker } from "../cost-tracker.js";

// Default timeout for generation requests (60 minutes for extended thinking)
const DEFAULT_TIMEOUT_MS = 60 * 60 * 1000;

export class OpenAITextProvider implements TextProvider {
  private client: OpenAI;

  constructor(apiKey: string, orgId?: string) {
    if (!apiKey) {
      throw new Error("OpenAI API key is required");
    }
    this.client = new OpenAI({
      apiKey,
      organization: orgId,
    });
    logger.info("OpenAI text provider initialized", { models: TEXT_MODELS });
  }

  async generate(options: TextGenerateOptions): Promise<GenerateResult> {
    const {
      prompt,
      systemPrompt,
      model = "gpt-5.2",
      reasoningEffort = "none",
      verbosity = "medium",
      reasoningSummary = "off",
      maxOutputTokens = 8192,
      temperature,
      jsonSchema,
    } = options;

    const startTime = Date.now();

    // Validation: gpt-5.2-pro only supports medium/high/xhigh reasoning effort
    if (model === "gpt-5.2-pro" && !isValidProReasoningEffort(reasoningEffort)) {
      throw new MCPError(
        "VALIDATION_ERROR",
        `gpt-5.2-pro only supports reasoning effort: medium, high, xhigh. Got: ${reasoningEffort}`
      );
    }

    // Validation: temperature is only allowed when reasoningEffort="none"
    if (temperature !== undefined && reasoningEffort !== "none") {
      throw new MCPError(
        "VALIDATION_ERROR",
        `Temperature is only allowed when reasoning_effort="none". Either remove temperature or set reasoning_effort to "none".`
      );
    }

    // Validation: structured outputs only supported by gpt-5.2 and gpt-5.2-chat-latest
    if (jsonSchema && !supportsStructuredOutput(model)) {
      throw new MCPError(
        "VALIDATION_ERROR",
        `Structured outputs (json_schema) are not supported by ${model}. Use gpt-5.2 or gpt-5.2-chat-latest.`
      );
    }

    logger.debugLog("Starting text generation", {
      promptLength: prompt.length,
      hasSystemPrompt: !!systemPrompt,
      model,
      reasoningEffort,
      verbosity,
      reasoningSummary,
      maxOutputTokens,
      temperature,
      hasJsonSchema: !!jsonSchema,
    });

    // Build the input - either string or message array
    let input: string | Array<{ role: string; content: string | Array<{ type: string; text: string }> }>;

    if (systemPrompt) {
      input = [
        {
          role: "developer",
          content: [{ type: "input_text", text: systemPrompt }],
        },
        {
          role: "user",
          content: [{ type: "input_text", text: prompt }],
        },
      ];
    } else {
      input = prompt;
    }

    try {
      // Build request options
      const requestOptions: any = {
        model: MODEL_IDS[model] || model,
        input,
        max_output_tokens: maxOutputTokens,
      };

      // Add text configuration (verbosity and optional structured output)
      requestOptions.text = { verbosity };

      // Add structured output format if json_schema is provided
      if (jsonSchema) {
        requestOptions.text.format = {
          type: "json_schema",
          name: jsonSchema.name,
          strict: jsonSchema.strict !== false, // Default to true
          schema: jsonSchema.schema,
        };
        if (jsonSchema.description) {
          requestOptions.text.format.description = jsonSchema.description;
        }
      }

      // Add reasoning effort (only if not "none")
      if (reasoningEffort !== "none") {
        requestOptions.reasoning = { effort: reasoningEffort };

        // Add reasoning summary if requested
        if (reasoningSummary !== "off") {
          requestOptions.reasoning.summary = reasoningSummary;
        }
      } else if (temperature !== undefined) {
        // Temperature only when effort=none
        requestOptions.temperature = temperature;
      }

      logger.debugLog("Text generation API request", {
        model: requestOptions.model,
        hasSystemPrompt: !!systemPrompt,
        reasoningEffort,
        verbosity,
        reasoningSummary,
        maxOutputTokens,
        temperature: requestOptions.temperature,
      });

      // Use retry wrapper for transient errors and timeout protection
      const response = await withRetry(
        () =>
          withTimeout(
            () => this.client.responses.create(requestOptions),
            DEFAULT_TIMEOUT_MS,
            "text-generation"
          ),
        {
          maxRetries: 2,
          retryableErrors: ["RATE_LIMIT", "429", "503", "502", "ECONNRESET", "ETIMEDOUT"],
          context: "text-generation",
        }
      );

      logger.debugLog("Text generation API response", {
        model,
        hasOutputText: !!(response as any).output_text,
        responseId: (response as any).id,
      });

      // Extract text from response
      const text = (response as any).output_text || "";

      // Get usage metadata
      const usage = (response as any).usage
        ? {
            promptTokens: (response as any).usage.input_tokens,
            completionTokens: (response as any).usage.output_tokens,
            totalTokens:
              ((response as any).usage.input_tokens || 0) +
              ((response as any).usage.output_tokens || 0),
            reasoningTokens: (response as any).usage.reasoning_tokens,
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
        operation: "generate_text",
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
        operation: "generate_text",
        durationMs,
        success: true,
        metrics: usage,
      });

      return {
        text,
        model,
        usage,
        cost,
      };
    } catch (error: any) {
      const durationMs = Date.now() - startTime;
      let errorType = "GENERATION_ERROR";
      let errorMessage = error.message || "Unknown error during generation";

      logger.error("Text generation API error", {
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
      } else if (error.message?.includes("content_policy") || error.message?.includes("moderation")) {
        errorType = "SAFETY_BLOCK";
        errorMessage = "Content was blocked by OpenAI safety filters";
      } else if (error.message?.includes("TIMEOUT")) {
        errorType = "TIMEOUT";
        errorMessage = "Request timed out. The prompt may be too complex or the service is slow.";
      }

      // Log failed usage
      logger.logUsage({
        timestamp: new Date().toISOString(),
        provider: "openai",
        model,
        operation: "generate_text",
        durationMs,
        success: false,
        error: `${errorType}: ${errorMessage}`,
      });

      throw new Error(`${errorType}: ${errorMessage}`);
    }
  }

  getModelInfo(model: TextModel = "gpt-5.2"): ModelInfo {
    const modelInfoMap: Record<TextModel, ModelInfo> = {
      "gpt-5.2": {
        id: "gpt-5.2",
        name: "GPT-5.2",
        provider: "openai",
        type: "text",
        contextWindow: 400000,
        maxOutput: 128000,
        supportsReasoning: true,
        description:
          "Best general choice for coding + agentic tasks. Supports reasoning effort: none, low, medium, high, xhigh.",
      },
      "gpt-5.2-pro": {
        id: "gpt-5.2-pro",
        name: "GPT-5.2 Pro",
        provider: "openai",
        type: "text",
        contextWindow: 400000,
        maxOutput: 128000,
        supportsReasoning: true,
        description:
          "Max accuracy for hardest problems. Can take minutes. Supports reasoning effort: medium, high, xhigh only.",
      },
      "gpt-5.2-chat-latest": {
        id: "gpt-5.2-chat-latest",
        name: "GPT-5.2 Chat Latest",
        provider: "openai",
        type: "text",
        contextWindow: 128000,
        maxOutput: 16384,
        supportsReasoning: true,
        description:
          "Points to the GPT-5.2 snapshot used in ChatGPT. Recommended to use gpt-5.2 for most API work.",
      },
    };

    return modelInfoMap[model];
  }
}
