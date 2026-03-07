/**
 * Built-in file tools for agentic text generation.
 * Provides read_file, list_directory, and grep_search tools
 * that models can call during generation to explore the filesystem.
 */

import { readFileSync, existsSync, readdirSync, statSync, openSync, readSync, closeSync } from "fs";
import { join, resolve, relative } from "path";
import { logger } from "./logger.js";

const MAX_OUTPUT_CHARS = 50000;
const MAX_GREP_RESULTS = 100;
const MAX_DIR_ENTRIES = 500;

const SKIP_DIRS = new Set([
  "node_modules", ".git", "dist", "build", ".next", "__pycache__",
  ".venv", "venv", ".cache", "coverage", ".nyc_output",
]);

export interface FileToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, any>;
}

export const FILE_TOOL_DEFINITIONS: FileToolDefinition[] = [
  {
    name: "read_file",
    description:
      "Read contents of a file from the local filesystem. Returns numbered lines. " +
      "For large files, use offset and limit to read specific sections.",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "File path (absolute or relative to working directory)",
        },
        offset: {
          type: "number",
          description: "Starting line number (1-based, default: 1)",
        },
        limit: {
          type: "number",
          description: "Maximum number of lines to return (default: 500)",
        },
      },
      required: ["path"],
    },
  },
  {
    name: "list_directory",
    description:
      "List files and directories at the given path. " +
      "Returns names with / suffix for directories, sorted with directories first.",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Directory path (absolute or relative to working directory)",
        },
      },
      required: ["path"],
    },
  },
  {
    name: "grep_search",
    description:
      "Search for a regex pattern in files. " +
      "Returns matching lines with file paths and line numbers. " +
      "Skips binary files and common non-source directories (node_modules, .git, dist, etc.).",
    parameters: {
      type: "object",
      properties: {
        pattern: {
          type: "string",
          description: "Search pattern (JavaScript regex syntax)",
        },
        path: {
          type: "string",
          description: "Directory or file to search in (default: current working directory)",
        },
        include: {
          type: "string",
          description: "File glob pattern to filter (e.g., '*.ts', '*.py'). Simple wildcards only.",
        },
      },
      required: ["pattern"],
    },
  },
];

function truncate(output: string): string {
  if (output.length <= MAX_OUTPUT_CHARS) return output;
  return (
    output.substring(0, MAX_OUTPUT_CHARS) +
    `\n\n[truncated — ${output.length - MAX_OUTPUT_CHARS} chars remaining]`
  );
}

function matchGlob(filename: string, pattern: string): boolean {
  const regex = pattern
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*/g, ".*")
    .replace(/\?/g, ".");
  return new RegExp(`^${regex}$`, "i").test(filename);
}

function isBinaryFile(filePath: string): boolean {
  try {
    const buffer = Buffer.alloc(512);
    const fd = openSync(filePath, "r");
    const bytesRead = readSync(fd, buffer, 0, 512, 0);
    closeSync(fd);
    for (let i = 0; i < bytesRead; i++) {
      if (buffer[i] === 0) return true;
    }
    return false;
  } catch {
    return false;
  }
}

export function executeFileTool(name: string, args: Record<string, any>): string {
  try {
    switch (name) {
      case "read_file":
        return executeReadFile(args);
      case "list_directory":
        return executeListDirectory(args);
      case "grep_search":
        return executeGrepSearch(args);
      default:
        return `Error: Unknown tool "${name}"`;
    }
  } catch (error: any) {
    return `Error: ${error.message}`;
  }
}

function executeReadFile(args: Record<string, any>): string {
  const filePath = resolve(args.path);
  if (!existsSync(filePath)) {
    return `Error: File not found: ${args.path}`;
  }
  const stat = statSync(filePath);
  if (stat.isDirectory()) {
    return `Error: Path is a directory, not a file: ${args.path}. Use list_directory instead.`;
  }
  if (isBinaryFile(filePath)) {
    return `Error: File appears to be binary: ${args.path}`;
  }

  const content = readFileSync(filePath, "utf-8");
  const lines = content.split("\n");
  const offset = Math.max(0, (args.offset || 1) - 1);
  const limit = args.limit || 500;
  const selected = lines.slice(offset, offset + limit);

  let result = selected.map((line, i) => `${offset + i + 1}: ${line}`).join("\n");
  if (offset + limit < lines.length) {
    result += `\n\n[${lines.length - offset - limit} more lines. Total: ${lines.length} lines]`;
  }
  return truncate(result);
}

function executeListDirectory(args: Record<string, any>): string {
  const dirPath = resolve(args.path);
  if (!existsSync(dirPath)) {
    return `Error: Directory not found: ${args.path}`;
  }
  const stat = statSync(dirPath);
  if (!stat.isDirectory()) {
    return `Error: Path is not a directory: ${args.path}. Use read_file instead.`;
  }

  const entries = readdirSync(dirPath, { withFileTypes: true });
  const sorted = entries
    .sort((a, b) => {
      if (a.isDirectory() !== b.isDirectory()) return a.isDirectory() ? -1 : 1;
      return a.name.localeCompare(b.name);
    })
    .slice(0, MAX_DIR_ENTRIES);

  const result = sorted
    .map((entry) => (entry.isDirectory() ? `${entry.name}/` : entry.name))
    .join("\n");

  if (entries.length > MAX_DIR_ENTRIES) {
    return result + `\n\n[${entries.length - MAX_DIR_ENTRIES} more entries not shown]`;
  }
  return result;
}

function executeGrepSearch(args: Record<string, any>): string {
  const searchPath = resolve(args.path || ".");
  let regex: RegExp;
  try {
    regex = new RegExp(args.pattern, "i");
  } catch {
    return `Error: Invalid regex pattern: ${args.pattern}`;
  }

  if (!existsSync(searchPath)) {
    return `Error: Path not found: ${args.path || "."}`;
  }

  const results: string[] = [];

  function searchInFile(filePath: string) {
    if (results.length >= MAX_GREP_RESULTS) return;
    if (isBinaryFile(filePath)) return;

    try {
      const content = readFileSync(filePath, "utf-8");
      const lines = content.split("\n");
      for (let i = 0; i < lines.length; i++) {
        if (results.length >= MAX_GREP_RESULTS) return;
        if (regex.test(lines[i])) {
          const relPath = relative(process.cwd(), filePath) || filePath;
          results.push(`${relPath}:${i + 1}: ${lines[i]}`);
        }
      }
    } catch {
      // Skip unreadable files
    }
  }

  function searchDir(dir: string) {
    if (results.length >= MAX_GREP_RESULTS) return;
    try {
      const entries = readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        if (results.length >= MAX_GREP_RESULTS) return;
        const fullPath = join(dir, entry.name);
        if (entry.isDirectory()) {
          if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
          searchDir(fullPath);
        } else if (entry.isFile()) {
          if (args.include && !matchGlob(entry.name, args.include)) continue;
          searchInFile(fullPath);
        }
      }
    } catch {
      // Skip inaccessible directories
    }
  }

  const stat = statSync(searchPath);
  if (stat.isFile()) {
    searchInFile(searchPath);
  } else {
    searchDir(searchPath);
  }

  if (results.length === 0) return "No matches found.";

  let output = results.join("\n");
  if (results.length >= MAX_GREP_RESULTS) {
    output += `\n\n[Results limited to ${MAX_GREP_RESULTS} matches]`;
  }
  return truncate(output);
}
