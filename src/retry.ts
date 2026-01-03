/**
 * Retry and timeout utilities for API calls
 */

import { logger } from "./logger.js";

export interface RetryOptions {
  maxRetries?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
  backoffMultiplier?: number;
  retryableErrors?: string[];
  /** Optional context for logging */
  context?: string;
}

const DEFAULT_RETRY_OPTIONS: Required<Omit<RetryOptions, "retryableErrors" | "context">> & {
  retryableErrors: string[];
} = {
  maxRetries: 3,
  initialDelayMs: 1000,
  maxDelayMs: 30000,
  backoffMultiplier: 2,
  retryableErrors: [
    "RATE_LIMIT",
    "429",
    "503",
    "502",
    "ECONNRESET",
    "ETIMEDOUT",
    "ENOTFOUND",
    "temporarily unavailable",
    "too many requests",
  ],
};

function isRetryable(error: unknown, retryableErrors: string[]): boolean {
  const message = error instanceof Error ? error.message : String(error);
  const retryable = retryableErrors.some((pattern) =>
    message.toLowerCase().includes(pattern.toLowerCase())
  );
  logger.debugLog("Checking if error is retryable", {
    errorMessage: message.substring(0, 200),
    isRetryable: retryable,
    matchedPatterns: retryableErrors.filter((pattern) =>
      message.toLowerCase().includes(pattern.toLowerCase())
    ),
  });
  return retryable;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function withRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const opts = { ...DEFAULT_RETRY_OPTIONS, ...options };
  const context = opts.context || "operation";
  let lastError: unknown;
  let delay = opts.initialDelayMs;

  logger.debugLog(`Starting ${context} with retry`, {
    maxRetries: opts.maxRetries,
    initialDelayMs: opts.initialDelayMs,
    maxDelayMs: opts.maxDelayMs,
    backoffMultiplier: opts.backoffMultiplier,
  });

  for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
    try {
      if (attempt > 0) {
        logger.debugLog(`${context}: Attempt ${attempt + 1} starting`, {
          previousAttempts: attempt,
        });
      }
      const result = await fn();
      if (attempt > 0) {
        logger.info(`${context}: Succeeded after ${attempt + 1} attempts`);
      }
      return result;
    } catch (error) {
      lastError = error;
      const errorMessage = error instanceof Error ? error.message : String(error);
      const errorStack = error instanceof Error ? error.stack : undefined;

      logger.debugLog(`${context}: Attempt ${attempt + 1} failed`, {
        attempt: attempt + 1,
        maxRetries: opts.maxRetries,
        error: errorMessage,
        stack: errorStack,
      });

      if (attempt === opts.maxRetries || !isRetryable(error, opts.retryableErrors)) {
        logger.warn(`${context}: Retry exhausted or non-retryable error`, {
          attempt: attempt + 1,
          maxRetries: opts.maxRetries,
          isRetryable: isRetryable(error, opts.retryableErrors),
          error: errorMessage,
          willRetry: false,
        });
        throw error;
      }

      logger.info(`${context}: Retrying after transient error`, {
        attempt: attempt + 1,
        maxRetries: opts.maxRetries,
        delayMs: delay,
        nextDelayMs: Math.min(delay * opts.backoffMultiplier, opts.maxDelayMs),
        error: errorMessage,
      });

      await sleep(delay);
      delay = Math.min(delay * opts.backoffMultiplier, opts.maxDelayMs);
    }
  }

  throw lastError;
}

export async function withTimeout<T>(
  fn: () => Promise<T>,
  timeoutMs: number,
  context?: string
): Promise<T> {
  const opContext = context || "operation";
  logger.debugLog(`${opContext}: Starting with timeout`, { timeoutMs });

  const startTime = Date.now();

  return Promise.race([
    fn().then((result) => {
      const elapsed = Date.now() - startTime;
      logger.debugLog(`${opContext}: Completed within timeout`, {
        elapsedMs: elapsed,
        timeoutMs,
        remainingMs: timeoutMs - elapsed,
      });
      return result;
    }),
    new Promise<T>((_, reject) =>
      setTimeout(() => {
        logger.warn(`${opContext}: Timed out`, { timeoutMs });
        reject(new Error(`TIMEOUT: Operation timed out after ${timeoutMs}ms`));
      }, timeoutMs)
    ),
  ]);
}
