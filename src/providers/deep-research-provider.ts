/**
 * OpenAI Deep Research provider using the Responses API in background mode
 *
 * Supports:
 * - o3-deep-research (default): Best deep research, most thorough
 * - o4-mini-deep-research: Faster/cheaper, still deep-research optimized
 *
 * Features:
 * - Background mode with polling
 * - Web search and code interpreter tools
 * - Long-form research reports with citations
 */

import OpenAI from "openai";
import type {
  DeepResearchOptions,
  DeepResearchResult,
  ModelInfo,
  DeepResearchProvider,
  DeepResearchModel,
} from "../types.js";
import { MODEL_IDS, DEEP_RESEARCH_MODELS } from "../types.js";
import { logger } from "../logger.js";
import { calculateCost } from "../pricing.js";
import { costTracker } from "../cost-tracker.js";

// Default timeout: 30 minutes (research typically takes 5-30 min)
const DEFAULT_TIMEOUT_MS = 30 * 60 * 1000;
const DEFAULT_POLL_INTERVAL_MS = 10 * 1000;

export class OpenAIDeepResearchProvider implements DeepResearchProvider {
  private client: OpenAI;

  constructor(apiKey: string, orgId?: string) {
    if (!apiKey) {
      throw new Error("OpenAI API key is required");
    }
    this.client = new OpenAI({
      apiKey,
      organization: orgId,
    });
    logger.info("OpenAI deep research provider initialized", { models: DEEP_RESEARCH_MODELS });
  }

  /**
   * Start a deep research task and poll until completion
   */
  async research(options: DeepResearchOptions): Promise<DeepResearchResult> {
    const {
      query,
      model = "o3-deep-research",
      timeoutMs = DEFAULT_TIMEOUT_MS,
      pollIntervalMs = DEFAULT_POLL_INTERVAL_MS,
    } = options;

    const startTime = Date.now();

    logger.debugLog("Starting deep research", {
      queryLength: query.length,
      model,
      timeoutMs,
      pollIntervalMs,
    });

    // Start the research task in background mode
    const responseId = await this.startResearch(query, model);
    logger.info("Deep research started", { responseId, model });

    // Poll for completion
    const result = await this.pollForCompletion(responseId, timeoutMs, pollIntervalMs, startTime);

    const durationMs = Date.now() - startTime;
    logger.info("Deep research completed", {
      responseId,
      model,
      status: result.status,
      durationMs,
      resultLength: result.text.length,
    });

    // Calculate cost
    const cost = calculateCost(
      model,
      result.usage?.promptTokens || 0,
      result.usage?.completionTokens || 0
    );

    // Track cumulative cost
    costTracker.trackCost({
      timestamp: new Date().toISOString(),
      model,
      operation: "deep_research",
      inputCost: cost.inputCost,
      outputCost: cost.outputCost,
      totalCost: cost.totalCost,
      estimated: cost.estimated || !result.usage, // Mark as estimated if no usage data
      promptTokens: result.usage?.promptTokens,
      completionTokens: result.usage?.completionTokens,
    });

    // Log usage
    logger.logUsage({
      timestamp: new Date().toISOString(),
      provider: "openai",
      model,
      operation: "deep_research",
      durationMs,
      success: result.status === "completed",
      metrics: result.usage,
    });

    return {
      ...result,
      model,
      responseId,
      durationMs,
      usage: result.usage,
      cost,
    };
  }

  /**
   * Start a new research task in background mode
   */
  private async startResearch(query: string, model: DeepResearchModel): Promise<string> {
    const requestOptions: any = {
      model: MODEL_IDS[model] || model,
      input: [
        {
          role: "developer",
          content: [
            {
              type: "input_text",
              text: "You are a research analyst. Conduct thorough research on the given topic. " +
                "Write a well-structured, comprehensive report with citations and sources. " +
                "Include key findings, analysis, and relevant data.",
            },
          ],
        },
        {
          role: "user",
          content: [{ type: "input_text", text: query }],
        },
      ],
      tools: [
        { type: "web_search_preview" },
        { type: "code_interpreter", container: { type: "auto" } },
      ],
      background: true, // Run in background mode
    };

    logger.debugLog("Deep research API request", {
      model: requestOptions.model,
      queryPreview: query.substring(0, 200) + (query.length > 200 ? "..." : ""),
      background: true,
    });

    try {
      const response = await this.client.responses.create(requestOptions);

      logger.debugLog("Deep research started successfully", {
        responseId: (response as any).id,
        status: (response as any).status,
      });

      const responseId = (response as any).id;
      if (!responseId) {
        throw new Error("API_ERROR: No response ID returned from API");
      }

      return responseId;
    } catch (error: any) {
      logger.error("Deep research start error", {
        model,
        errorName: error.name,
        errorMessage: error.message,
      });

      // Handle specific errors
      if (error.message?.includes("API key") || error.message?.includes("Incorrect API key") || error.status === 401) {
        throw new Error("AUTH_ERROR: Invalid or missing OpenAI API key");
      } else if (error.status === 429) {
        throw new Error("RATE_LIMIT: OpenAI API rate limit exceeded. Please wait and retry.");
      }

      throw new Error(`API_ERROR: ${error.message}`);
    }
  }

  /**
   * Poll for research completion
   */
  private async pollForCompletion(
    responseId: string,
    timeoutMs: number,
    pollIntervalMs: number,
    startTime: number
  ): Promise<{ text: string; status: "completed" | "failed"; usage?: { promptTokens?: number; completionTokens?: number; totalTokens?: number } }> {
    while (true) {
      const elapsed = Date.now() - startTime;

      if (elapsed > timeoutMs) {
        const minutes = Math.round(elapsed / 1000 / 60);
        throw new Error(
          `TIMEOUT: Research timed out after ${minutes} minutes. ` +
            `The research may still be running - response ID: ${responseId}`
        );
      }

      logger.debugLog("Polling research status", {
        responseId,
        elapsedMs: elapsed,
        elapsedMinutes: Math.round((elapsed / 1000 / 60) * 10) / 10,
      });

      try {
        const response = await this.client.responses.retrieve(responseId);

        logger.debugLog("Research poll response", {
          responseId,
          status: (response as any).status,
          elapsedMs: elapsed,
          hasOutput: !!(response as any).output_text,
        });

        const status = (response as any).status;

        if (status === "completed") {
          const text = (response as any).output_text || "";

          if (!text) {
            throw new Error("API_ERROR: Research completed but no output text found");
          }

          // Extract usage if available
          const rawUsage = (response as any).usage;
          const usage = rawUsage
            ? {
                promptTokens: rawUsage.input_tokens,
                completionTokens: rawUsage.output_tokens,
                totalTokens: (rawUsage.input_tokens || 0) + (rawUsage.output_tokens || 0),
              }
            : undefined;

          return {
            text,
            status: "completed",
            usage,
          };
        }

        if (status === "failed") {
          const errorMessage = (response as any).error?.message || "Research failed with unknown error";
          throw new Error(`RESEARCH_FAILED: ${errorMessage}`);
        }

        // Status is still "in_progress" or similar - wait and poll again
        await this.sleep(pollIntervalMs);
      } catch (error: any) {
        // If it's our own error, re-throw
        if (error.message?.startsWith("TIMEOUT:") ||
            error.message?.startsWith("RESEARCH_FAILED:") ||
            error.message?.startsWith("API_ERROR:")) {
          throw error;
        }

        // Otherwise, it's a poll error - log and retry
        logger.warn("Research poll error, retrying", {
          responseId,
          error: error.message,
        });

        await this.sleep(pollIntervalMs);
      }
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  getModelInfo(model: DeepResearchModel = "o3-deep-research"): ModelInfo {
    const modelInfoMap: Record<DeepResearchModel, ModelInfo> = {
      "o3-deep-research": {
        id: "o3-deep-research",
        name: "O3 Deep Research",
        provider: "openai",
        type: "research",
        contextWindow: 200000,
        maxOutput: 100000,
        description:
          "Deep research specialist. Conducts thorough web research and produces comprehensive reports. " +
          "Takes 5-30 minutes. Best for complex research questions.",
      },
      "o4-mini-deep-research": {
        id: "o4-mini-deep-research",
        name: "O4 Mini Deep Research",
        provider: "openai",
        type: "research",
        contextWindow: 200000,
        maxOutput: 100000,
        description:
          "Faster, more affordable deep research. Still produces thorough reports but optimized for speed. " +
          "Takes 5-20 minutes. Good for simpler research tasks.",
      },
    };

    return modelInfoMap[model];
  }
}
