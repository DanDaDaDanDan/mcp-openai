/**
 * OpenAI text provider using the Responses API
 *
 * Supports:
 * - gpt-5.4 (default): Best general choice, 1.05M context
 * - gpt-5.4-pro: Max accuracy, reasoning medium/high/xhigh only, 1.05M context
 *
 * Features:
 * - Reasoning effort control (none, low, medium, high, xhigh)
 * - Verbosity control (low, medium, high)
 * - Reasoning summaries (concise, detailed)
 */

import { readFileSync, existsSync } from "fs";
import { basename, extname } from "path";
import OpenAI from "openai";
import type {
  TextGenerateOptions,
  GenerateResult,
  ModelInfo,
  TextProvider,
  TextModel,
  Attachment,
} from "../types.js";
import {
  TEXT_MODELS,
  MODEL_IDS,
  isValidProReasoningEffort,
  supportsStructuredOutput,
  isImageMediaType,
  isSupportedMediaType,
  MCPError,
} from "../types.js";
import { logger } from "../logger.js";
import { withRetry, withTimeout } from "../retry.js";
import { calculateCost } from "../pricing.js";
import { costTracker } from "../cost-tracker.js";

// Default timeout for generation requests (120 minutes for extended thinking)
const DEFAULT_TIMEOUT_MS = 120 * 60 * 1000;

// Map file extension to MIME type
const EXT_TO_MEDIA_TYPE: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".pdf": "application/pdf",
};

/**
 * Resolve a single attachment to a base64 data URI and metadata.
 * Validates that exactly one source is provided and the media type is supported.
 */
function resolveAttachment(
  attachment: Attachment,
  index: number
): { dataUri: string; mediaType: string; filename?: string; url?: string } {
  const sources = [attachment.path, attachment.data, attachment.url].filter(Boolean);
  if (sources.length !== 1) {
    throw new MCPError(
      "VALIDATION_ERROR",
      `Attachment ${index}: exactly one of 'path', 'data', or 'url' must be provided (got ${sources.length})`
    );
  }

  // URL source — only for images, passed directly
  if (attachment.url) {
    // We can't know the media type for sure from a URL, but it must be an image
    return { dataUri: "", mediaType: "image/unknown", url: attachment.url };
  }

  let base64Data: string;
  let mediaType: string | undefined = attachment.media_type;
  let filename = attachment.filename;

  if (attachment.path) {
    // Read file from disk
    const filePath = attachment.path;
    if (!existsSync(filePath)) {
      throw new MCPError(
        "VALIDATION_ERROR",
        `Attachment ${index}: file not found: ${filePath}`
      );
    }

    const ext = extname(filePath).toLowerCase();
    if (!mediaType) {
      mediaType = EXT_TO_MEDIA_TYPE[ext];
      if (!mediaType) {
        throw new MCPError(
          "VALIDATION_ERROR",
          `Attachment ${index}: cannot infer media type from extension '${ext}'. Supported: ${Object.keys(EXT_TO_MEDIA_TYPE).join(", ")}`
        );
      }
    }

    if (!filename) {
      filename = basename(filePath);
    }

    const fileBuffer = readFileSync(filePath);
    base64Data = fileBuffer.toString("base64");
  } else {
    // Inline data
    if (!mediaType) {
      throw new MCPError(
        "VALIDATION_ERROR",
        `Attachment ${index}: 'media_type' is required when using 'data'`
      );
    }

    // Handle both raw base64 and data URI formats
    if (attachment.data!.startsWith("data:")) {
      // Already a data URI — extract the base64 part
      const match = attachment.data!.match(/^data:([^;]+);base64,(.+)$/);
      if (!match) {
        throw new MCPError(
          "VALIDATION_ERROR",
          `Attachment ${index}: invalid data URI format`
        );
      }
      base64Data = match[2];
    } else {
      base64Data = attachment.data!;
    }
  }

  if (!isSupportedMediaType(mediaType)) {
    throw new MCPError(
      "VALIDATION_ERROR",
      `Attachment ${index}: unsupported media type '${mediaType}'. Supported: image/png, image/jpeg, image/gif, image/webp, application/pdf`
    );
  }

  const dataUri = `data:${mediaType};base64,${base64Data}`;
  return { dataUri, mediaType, filename };
}

/**
 * Convert attachments to OpenAI Responses API content parts.
 */
function attachmentsToContentParts(
  attachments: Attachment[]
): Array<Record<string, any>> {
  const parts: Array<Record<string, any>> = [];

  for (let i = 0; i < attachments.length; i++) {
    const resolved = resolveAttachment(attachments[i], i);

    if (resolved.url) {
      // URL image — pass directly
      parts.push({
        type: "input_image",
        image_url: resolved.url,
      });
    } else if (isImageMediaType(resolved.mediaType)) {
      // Base64 image
      parts.push({
        type: "input_image",
        image_url: resolved.dataUri,
      });
    } else {
      // File (PDF etc.)
      parts.push({
        type: "input_file",
        filename: resolved.filename || `attachment_${i}`,
        file_data: resolved.dataUri,
      });
    }
  }

  return parts;
}

export class OpenAITextProvider implements TextProvider {
  private client: OpenAI;

  constructor(apiKey: string, orgId?: string) {
    if (!apiKey) {
      throw new Error("OpenAI API key is required");
    }
    this.client = new OpenAI({
      apiKey,
      organization: orgId,
      timeout: DEFAULT_TIMEOUT_MS,
    });
    logger.info("OpenAI text provider initialized", { models: TEXT_MODELS });
  }

  async generate(options: TextGenerateOptions): Promise<GenerateResult> {
    const {
      prompt,
      systemPrompt,
      model = "gpt-5.4",
      reasoningEffort = "none",
      verbosity = "medium",
      reasoningSummary = "off",
      maxOutputTokens = 8192,
      temperature,
      jsonSchema,
      attachments,
    } = options;

    const startTime = Date.now();

    // Validation: gpt-5.4-pro only supports medium/high/xhigh reasoning effort
    if (model === "gpt-5.4-pro" && !isValidProReasoningEffort(reasoningEffort)) {
      throw new MCPError(
        "VALIDATION_ERROR",
        `gpt-5.4-pro only supports reasoning effort: medium, high, xhigh. Got: ${reasoningEffort}`
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
      attachmentCount: attachments?.length || 0,
    });

    // Convert attachments to content parts if present
    const attachmentParts = attachments?.length
      ? attachmentsToContentParts(attachments)
      : [];

    // Build the input - use array form when we have system prompt or attachments
    let input: any;

    if (systemPrompt || attachmentParts.length > 0) {
      const userContent: Array<Record<string, any>> = [
        { type: "input_text", text: prompt },
        ...attachmentParts,
      ];

      input = [];
      if (systemPrompt) {
        input.push({
          role: "developer",
          content: [{ type: "input_text", text: systemPrompt }],
        });
      }
      input.push({
        role: "user",
        content: userContent,
      });
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

  getModelInfo(model: TextModel = "gpt-5.4"): ModelInfo {
    const modelInfoMap: Record<TextModel, ModelInfo> = {
      "gpt-5.4": {
        id: "gpt-5.4",
        name: "GPT-5.4",
        provider: "openai",
        type: "text",
        contextWindow: 1050000,
        maxOutput: 128000,
        supportsReasoning: true,
        description:
          "Best general choice for coding + agentic tasks. 1.05M context. Supports reasoning effort: none, low, medium, high, xhigh.",
      },
      "gpt-5.4-pro": {
        id: "gpt-5.4-pro",
        name: "GPT-5.4 Pro",
        provider: "openai",
        type: "text",
        contextWindow: 1050000,
        maxOutput: 128000,
        supportsReasoning: true,
        description:
          "Max accuracy for hardest problems. 1.05M context. Can take minutes. Supports reasoning effort: medium, high, xhigh only.",
      },
    };

    return modelInfoMap[model];
  }
}
