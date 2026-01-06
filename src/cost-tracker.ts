/**
 * Cost Tracker - Tracks cumulative costs across tool calls
 *
 * Features:
 * - In-memory tracking for fast queries
 * - File persistence to costs.jsonl
 * - Load historical data on startup
 */

import { appendFileSync, readFileSync, existsSync, mkdirSync } from "fs";
import { join, dirname } from "path";

// ============================================================================
// Types
// ============================================================================

export interface CostEntry {
  timestamp: string;
  model: string;
  operation: string;
  inputCost: number;
  outputCost: number;
  totalCost: number;
  estimated: boolean;
  promptTokens?: number;
  completionTokens?: number;
}

export interface CostSummary {
  totalCost: number;
  byModel: Record<string, number>;
  byOperation: Record<string, number>;
  callCount: number;
  estimatedCosts: number;
  since: string;
}

// ============================================================================
// Cost Tracker
// ============================================================================

class CostTracker {
  private entries: CostEntry[] = [];
  private costsFile: string | null = null;

  constructor() {
    const logDir = process.env.MCP_LOG_DIR ?? "logs";
    if (logDir !== "none") {
      try {
        if (!existsSync(logDir)) {
          mkdirSync(logDir, { recursive: true });
        }
        this.costsFile = join(logDir, "costs.jsonl");
        this.loadHistoricalCosts();
      } catch {
        // Silently fail if we can't set up file persistence
        this.costsFile = null;
      }
    }
  }

  /**
   * Load historical costs from file on startup
   */
  private loadHistoricalCosts(): void {
    if (!this.costsFile || !existsSync(this.costsFile)) {
      return;
    }

    try {
      const content = readFileSync(this.costsFile, "utf-8");
      const lines = content.trim().split("\n").filter((line) => line);

      for (const line of lines) {
        try {
          const entry = JSON.parse(line) as CostEntry;
          this.entries.push(entry);
        } catch {
          // Skip malformed lines
        }
      }
    } catch {
      // File read failed, start fresh
    }
  }

  /**
   * Track a cost entry and persist to file
   */
  trackCost(entry: CostEntry): void {
    this.entries.push(entry);

    // Persist to file
    if (this.costsFile) {
      try {
        appendFileSync(this.costsFile, JSON.stringify(entry) + "\n");
      } catch {
        // Silently fail file writes
      }
    }
  }

  /**
   * Get cumulative cost summary
   */
  getSummary(): CostSummary {
    const byModel: Record<string, number> = {};
    const byOperation: Record<string, number> = {};
    let totalCost = 0;
    let estimatedCosts = 0;

    for (const entry of this.entries) {
      totalCost += entry.totalCost;

      byModel[entry.model] = (byModel[entry.model] || 0) + entry.totalCost;
      byOperation[entry.operation] = (byOperation[entry.operation] || 0) + entry.totalCost;

      if (entry.estimated) {
        estimatedCosts += entry.totalCost;
      }
    }

    // Round all values
    const round = (n: number) => Math.round(n * 1_000_000) / 1_000_000;

    return {
      totalCost: round(totalCost),
      byModel: Object.fromEntries(Object.entries(byModel).map(([k, v]) => [k, round(v)])),
      byOperation: Object.fromEntries(Object.entries(byOperation).map(([k, v]) => [k, round(v)])),
      callCount: this.entries.length,
      estimatedCosts: round(estimatedCosts),
      since: this.entries[0]?.timestamp || new Date().toISOString(),
    };
  }

  /**
   * Reset in-memory costs (does not delete file)
   */
  reset(): void {
    this.entries = [];
  }

  /**
   * Get raw entries for debugging
   */
  getEntries(): CostEntry[] {
    return [...this.entries];
  }
}

// Export singleton instance
export const costTracker = new CostTracker();
