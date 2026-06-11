/**
 * Prompt builder for in-context learning with retrieved cases.
 *
 * Formats positive and negative cases into sections that can be prepended to
 * an LLM prompt so the model learns from past experience.
 */

import type { CaseEntry, CaseSearchResults } from "../types";

export interface CasePromptBuilderOptions {
  maxPositiveExamples?: number;
  maxNegativeExamples?: number;
}

export class CasePromptBuilder {
  private readonly maxPositive: number;
  private readonly maxNegative: number;

  constructor(options: CasePromptBuilderOptions = {}) {
    this.maxPositive = options.maxPositiveExamples ?? 3;
    this.maxNegative = options.maxNegativeExamples ?? 2;
  }

  buildContext(cases: CaseSearchResults): string {
    const sections: string[] = [];

    const positive = cases.positive_cases.slice(0, this.maxPositive);
    if (positive.length > 0) {
      sections.push(renderSection("Successful Past Approaches", "Example", positive));
    }

    const negative = cases.negative_cases.slice(0, this.maxNegative);
    if (negative.length > 0) {
      sections.push(
        renderSection("Approaches to Avoid", "Counter-Example", negative)
      );
    }

    return sections.join("\n");
  }
}

function renderSection(
  heading: string,
  itemLabel: string,
  cases: CaseEntry[]
): string {
  const lines: string[] = [`## ${heading}`, ""];
  cases.forEach((entry, index) => {
    lines.push(`### ${itemLabel} ${index + 1}`);
    lines.push(`**Problem:** ${entry.problem}`);
    lines.push(`**Plan:**\n${formatPlan(entry.plan)}`);
    lines.push(`**Outcome:** ${entry.outcome}`);
    lines.push("");
  });
  return lines.join("\n");
}

export function formatPlan(plan: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(plan);
  } catch {
    return plan;
  }

  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    const record = parsed as Record<string, unknown>;
    if ("plan" in record) {
      parsed = record.plan;
    }
  }

  if (!Array.isArray(parsed)) {
    return plan;
  }

  return parsed
    .map((step, index) => {
      if (step && typeof step === "object" && !Array.isArray(step)) {
        const description = (step as Record<string, unknown>).description;
        const text =
          typeof description === "string" ? description : JSON.stringify(step);
        return `${index + 1}. ${text}`;
      }
      return `${index + 1}. ${String(step)}`;
    })
    .join("\n");
}
