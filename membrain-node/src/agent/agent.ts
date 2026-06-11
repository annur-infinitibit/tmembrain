/**
 * Top-level Membrain agent combining planning, execution, and learning.
 *
 * {@link MembrainAgent} ties the planner, executor, and experience replay
 * loop into a single `run()` method. `runBatch()` drives multiple queries
 * with a user-supplied judge callable for reward assignment.
 */

import type { MembrainClient } from "../client";
import { CasePromptBuilder } from "../cbr/prompt_builder";
import { ExperienceReplay } from "../cbr/experience_replay";
import { CaseRetriever, NonParametricRetriever } from "../cbr/retriever";
import type { LLMCallable } from "../conversation";
import type { CaseEntry, JudgeCallable, TaskResult } from "../types";
import { MembrainExecutor } from "./executor";
import { MembrainPlanner } from "./planner";

export interface MembrainAgentOptions {
  retriever?: CaseRetriever;
  promptBuilder?: CasePromptBuilder;
  trainingDataPath?: string;
  caseLimit?: number;
  maxCycles?: number;
}

export class MembrainAgent {
  private readonly client: MembrainClient;
  private readonly maxCycles: number;
  private readonly planner: MembrainPlanner;
  private readonly _executor: MembrainExecutor;
  private readonly _replay: ExperienceReplay;

  constructor(
    client: MembrainClient,
    llmCallable: LLMCallable,
    options: MembrainAgentOptions = {}
  ) {
    this.client = client;
    this.maxCycles = options.maxCycles ?? 3;

    const retriever = options.retriever ?? new NonParametricRetriever(client);
    const promptBuilder = options.promptBuilder ?? new CasePromptBuilder();

    this.planner = new MembrainPlanner(retriever, promptBuilder, llmCallable, {
      caseLimit: options.caseLimit ?? 5,
    });
    this._executor = new MembrainExecutor(llmCallable);
    this._replay = new ExperienceReplay(client, {
      retriever,
      trainingDataPath: options.trainingDataPath,
    });
  }

  get executor(): MembrainExecutor {
    return this._executor;
  }

  get replay(): ExperienceReplay {
    return this._replay;
  }

  async run(query: string): Promise<string> {
    let output = "";

    for (let cycle = 0; cycle < this.maxCycles; cycle += 1) {
      const plan = await this.planner.plan(query);
      const results: TaskResult[] = [];

      let allSucceeded = true;
      for (const step of plan.steps) {
        const result = await this._executor.execute(step);
        results.push(result);
        if (!result.success) {
          allSucceeded = false;
          break;
        }
      }

      output = results
        .map((result) => result.output)
        .filter((line) => line.length > 0)
        .join("\n");

      if (allSucceeded) {
        return output;
      }
    }

    return output;
  }

  async runBatch(
    queries: string[],
    judgeCallable: JudgeCallable
  ): Promise<string[]> {
    const outputs: string[] = [];
    for (const query of queries) {
      const cases = await this._replay.retriever.retrieve(query);
      const retrievedCases: CaseEntry[] = [
        ...cases.positive_cases,
        ...cases.negative_cases,
      ];

      const output = await this.run(query);
      outputs.push(output);

      const isCorrect = judgeCallable(query, output, []);
      const reward = isCorrect ? 1.0 : 0.0;

      this._replay.recordExecution({
        problem: query,
        plan: output,
        outcome: isCorrect ? "correct" : "incorrect",
        reward,
        query,
        retrievedCases,
        isCorrect,
      });
    }
    return outputs;
  }
}
