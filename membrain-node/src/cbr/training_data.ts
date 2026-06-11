/**
 * Training data collection and management for the neural retriever.
 *
 * Accumulates training pairs during agent execution and serialises them to
 * JSONL. The JSONL schema matches the Python {@link TrainingDataCollector}
 * byte-for-byte so training data can be shared across implementations.
 */

import { promises as fs } from "node:fs";
import * as path from "node:path";

import type { CaseEntry, TrainingPair } from "../types";

interface JsonlRecord {
  query: string;
  case: string;
  case_label: "positive" | "negative";
  plan: string;
  truth_label: boolean;
}

export class TrainingDataCollector {
  private buffer: TrainingPair[] = [];

  get size(): number {
    return this.buffer.length;
  }

  record(
    query: string,
    retrievedCases: CaseEntry[],
    isCorrect: boolean,
    positiveRewardThreshold: number = 0.5
  ): void {
    for (const entry of retrievedCases) {
      const caseLabel: "positive" | "negative" =
        entry.reward >= positiveRewardThreshold ? "positive" : "negative";
      const caseText =
        `[CASE]\n` +
        `Problem: ${entry.problem}\n` +
        `Outcome: ${entry.outcome}\n` +
        `Reward: ${entry.reward}`;
      this.buffer.push({
        query,
        case_text: caseText,
        case_label: caseLabel,
        plan: entry.plan,
        truth_label: isCorrect,
      });
    }
  }

  async flush(filePath: string): Promise<number> {
    const parent = path.dirname(filePath);
    await fs.mkdir(parent, { recursive: true });

    const lines = this.buffer.map((pair) => {
      const record: JsonlRecord = {
        query: pair.query,
        case: pair.case_text,
        case_label: pair.case_label,
        plan: pair.plan,
        truth_label: pair.truth_label,
      };
      return JSON.stringify(record);
    });

    const count = this.buffer.length;
    if (count > 0) {
      await fs.appendFile(filePath, lines.join("\n") + "\n", "utf8");
    }
    this.buffer = [];
    return count;
  }

  static async load(filePath: string): Promise<TrainingPair[]> {
    const content = await fs.readFile(filePath, "utf8");
    const pairs: TrainingPair[] = [];
    for (const rawLine of content.split("\n")) {
      const line = rawLine.trim();
      if (!line) continue;
      const record = JSON.parse(line) as JsonlRecord;
      pairs.push({
        query: record.query,
        case_text: record.case,
        case_label: record.case_label,
        plan: record.plan,
        truth_label: record.truth_label,
      });
    }
    return pairs;
  }
}
