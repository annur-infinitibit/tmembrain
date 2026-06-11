/**
 * LLM-agnostic task planner with case-based context injection.
 *
 * Retrieves similar past cases, builds an in-context learning prompt, then
 * asks an LLM to decompose a query into a minimal task plan.
 */

import type { CasePromptBuilder } from "../cbr/prompt_builder";
import type { CaseRetriever } from "../cbr/retriever";
import type { LLMCallable } from "../conversation";
import type { ChatMessage, TaskPlan, TaskStep } from "../types";

export const SYSTEM_PROMPT = `You are a task planner. Given a user query and optional context from past experiences, decompose the query into a minimal sequence of steps.

Return ONLY a JSON object in this format:
{"plan": [{"id": 1, "description": "..."}, {"id": 2, "description": "..."}]}

Rules:
- Each step must be a concrete, actionable task.
- Use the fewest steps possible.
- If past successful approaches are provided, learn from them.
- If approaches to avoid are provided, do not repeat those mistakes.
- Return valid JSON only, no markdown or explanation.`;

export interface PlannerOptions {
  caseLimit?: number;
}

export class MembrainPlanner {
  private readonly retriever: CaseRetriever;
  private readonly promptBuilder: CasePromptBuilder;
  private readonly llmCallable: LLMCallable;
  private readonly caseLimit: number;

  constructor(
    retriever: CaseRetriever,
    promptBuilder: CasePromptBuilder,
    llmCallable: LLMCallable,
    options: PlannerOptions = {}
  ) {
    this.retriever = retriever;
    this.promptBuilder = promptBuilder;
    this.llmCallable = llmCallable;
    this.caseLimit = options.caseLimit ?? 5;
  }

  async plan(query: string): Promise<TaskPlan> {
    const cases = await this.retriever.retrieve(query, this.caseLimit);
    const context = this.promptBuilder.buildContext(cases);

    let userContent = query;
    if (context) {
      userContent =
        `## Past Experience Context\n\n${context}\n\n` +
        `## Current Query\n\n${query}`;
    }

    const messages: ChatMessage[] = [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userContent },
    ];

    const rawResponse = await this.llmCallable(messages);
    return parsePlan(rawResponse);
  }
}

export function parsePlan(response: string): TaskPlan {
  let cleaned = response.trim();
  if (cleaned.startsWith("```")) {
    const lines = cleaned
      .split("\n")
      .filter((line) => !line.trim().startsWith("```"));
    cleaned = lines.join("\n");
  }

  let data: unknown;
  try {
    data = JSON.parse(cleaned);
  } catch {
    return {
      steps: [{ id: 1, description: cleaned }],
      raw_response: response,
    };
  }

  const steps: TaskStep[] = [];
  const planList =
    data && typeof data === "object" && "plan" in data
      ? (data as { plan?: unknown }).plan
      : [];

  if (Array.isArray(planList)) {
    for (const item of planList) {
      if (item && typeof item === "object") {
        const record = item as Record<string, unknown>;
        const id =
          typeof record.id === "number" ? record.id : steps.length + 1;
        const description =
          typeof record.description === "string"
            ? record.description
            : JSON.stringify(item);
        steps.push({ id, description });
      }
    }
  }

  if (steps.length === 0) {
    steps.push({ id: 1, description: cleaned });
  }

  return { steps, raw_response: response };
}
