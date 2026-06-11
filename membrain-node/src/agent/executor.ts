/**
 * LLM-agnostic task executor.
 *
 * Executes individual steps from a task plan, optionally using registered
 * tool functions.
 */

import type { LLMCallable } from "../conversation";
import type { ChatMessage, TaskResult, TaskStep } from "../types";

export type ToolFunction = (...args: string[]) => string | Promise<string>;

export class MembrainExecutor {
  private readonly llmCallable: LLMCallable;
  private readonly tools: Map<string, ToolFunction> = new Map();

  constructor(llmCallable: LLMCallable) {
    this.llmCallable = llmCallable;
  }

  registerTool(name: string, fn: ToolFunction): void {
    this.tools.set(name, fn);
  }

  get availableTools(): string[] {
    return Array.from(this.tools.keys());
  }

  async execute(task: TaskStep): Promise<TaskResult> {
    let toolDescriptions = "";
    if (this.tools.size > 0) {
      const toolList = Array.from(this.tools.keys()).join(", ");
      toolDescriptions =
        `\n\nAvailable tools: ${toolList}\n` +
        "To use a tool, respond with: TOOL_CALL: tool_name(arg1, arg2)\n" +
        "Otherwise, respond with the task output directly.";
    }

    const messages: ChatMessage[] = [
      {
        role: "system",
        content:
          "You are a task executor. Complete the given task and " +
          "return the result." +
          toolDescriptions,
      },
      { role: "user", content: task.description },
    ];

    try {
      const response = await this.llmCallable(messages);
      const output = await this.handleToolCalls(response);
      return { task, output, success: true, error: null };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return { task, output: "", success: false, error: message };
    }
  }

  private async handleToolCalls(response: string): Promise<string> {
    const trimmed = response.trim();
    if (!trimmed.startsWith("TOOL_CALL:")) {
      return response;
    }

    const toolLine = trimmed.slice("TOOL_CALL:".length).trim();
    const parenIndex = toolLine.indexOf("(");
    if (parenIndex === -1) return response;

    const toolName = toolLine.slice(0, parenIndex).trim();
    const tool = this.tools.get(toolName);
    if (!tool) {
      return `Unknown tool: ${toolName}. Raw response: ${response}`;
    }

    const argsRaw = toolLine.slice(parenIndex + 1).replace(/\)\s*$/, "");
    const args = argsRaw
      .split(",")
      .map((arg) => arg.trim().replace(/^['"]|['"]$/g, ""))
      .filter((arg) => arg.length > 0);

    return await tool(...args);
  }
}
