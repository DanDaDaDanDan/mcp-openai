/**
 * OpenAI Model Pricing and Cost Calculation
 *
 * Pricing source: https://openai.com/api/pricing/
 * Last updated: January 2026
 */

// ============================================================================
// Types
// ============================================================================

export interface TokenPricing {
  input: number; // USD per 1M input tokens
  output: number; // USD per 1M output tokens
}

export interface CostInfo {
  inputCost: number;
  outputCost: number;
  totalCost: number;
  currency: "USD";
  estimated: boolean;
}

// ============================================================================
// Pricing Constants (USD per 1M tokens)
// ============================================================================

export const OPENAI_PRICING: Record<string, TokenPricing> = {
  // GPT-5.2 family
  "gpt-5.2": {
    input: 1.75,
    output: 14.0,
  },
  "gpt-5.2-pro": {
    input: 21.0,
    output: 168.0,
  },
  "gpt-5.2-chat-latest": {
    input: 1.75,
    output: 14.0,
  },
  // Deep research models
  "o3-deep-research": {
    input: 10.0,
    output: 40.0,
  },
  "o4-mini-deep-research": {
    input: 2.0,
    output: 8.0,
  },
};

// Default pricing for unknown models (uses gpt-5.2-pro as conservative estimate)
export const DEFAULT_PRICING: TokenPricing = {
  input: 21.0,
  output: 168.0,
};

// ============================================================================
// Cost Calculation
// ============================================================================

/**
 * Calculate cost for a request based on token usage
 *
 * @param model - The model ID
 * @param promptTokens - Number of input tokens
 * @param completionTokens - Number of output tokens (includes reasoning tokens)
 * @returns Cost breakdown with estimated flag
 */
export function calculateCost(
  model: string,
  promptTokens: number = 0,
  completionTokens: number = 0
): CostInfo {
  const pricing = OPENAI_PRICING[model];
  const estimated = !pricing;
  const effectivePricing = pricing || DEFAULT_PRICING;

  const inputCost = (promptTokens / 1_000_000) * effectivePricing.input;
  const outputCost = (completionTokens / 1_000_000) * effectivePricing.output;
  const totalCost = inputCost + outputCost;

  return {
    inputCost: roundToMicro(inputCost),
    outputCost: roundToMicro(outputCost),
    totalCost: roundToMicro(totalCost),
    currency: "USD",
    estimated,
  };
}

/**
 * Round to 6 decimal places (micro-dollar precision)
 */
function roundToMicro(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

/**
 * Get pricing for a specific model
 */
export function getPricing(model: string): TokenPricing & { estimated: boolean } {
  const pricing = OPENAI_PRICING[model];
  return {
    ...(pricing || DEFAULT_PRICING),
    estimated: !pricing,
  };
}
