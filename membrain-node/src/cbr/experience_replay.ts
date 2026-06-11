/**
 * Experience replay loop tying storage, retrieval, and training.
 *
 * Orchestrates the store-retrieve-train cycle. Retraining of the neural
 * retriever (BERT fine-tuning with PyTorch) is only available via the Python
 * package; {@link ExperienceReplay.retrain} throws on Node.js with a pointer
 * to `membrain-py`.
 */

import type { MembrainClient } from "../client";
import { MembrainError } from "../ffi";
import type { CaseEntry } from "../types";
import { CaseRetriever, NonParametricRetriever } from "./retriever";
import { TrainingDataCollector } from "./training_data";

export interface ExperienceReplayOptions {
  retriever?: CaseRetriever;
  trainingDataPath?: string;
  positiveRewardThreshold?: number;
}

export interface RecordExecutionInput {
  problem: string;
  plan: string;
  outcome: string;
  reward: number;
  query: string;
  retrievedCases: CaseEntry[];
  isCorrect: boolean;
}

export class ExperienceReplay {
  private readonly client: MembrainClient;
  private _retriever: CaseRetriever;
  private readonly trainingDataPath: string;
  private readonly positiveRewardThreshold: number;
  private readonly collector: TrainingDataCollector;
  private executionCount = 0;

  constructor(client: MembrainClient, options: ExperienceReplayOptions = {}) {
    this.client = client;
    this._retriever = options.retriever ?? new NonParametricRetriever(client);
    this.trainingDataPath = options.trainingDataPath ?? "training_data.jsonl";
    this.positiveRewardThreshold = options.positiveRewardThreshold ?? 0.5;
    this.collector = new TrainingDataCollector();
  }

  get retriever(): CaseRetriever {
    return this._retriever;
  }

  set retriever(value: CaseRetriever) {
    this._retriever = value;
  }

  get pendingTrainingPairs(): number {
    return this.collector.size;
  }

  get executionsSinceFlush(): number {
    return this.executionCount;
  }

  recordExecution(input: RecordExecutionInput): string | null {
    const result = this.client.storeCase(
      input.problem,
      input.plan,
      input.outcome,
      input.reward
    );

    this.collector.record(
      input.query,
      input.retrievedCases,
      input.isCorrect,
      this.positiveRewardThreshold
    );

    this.executionCount += 1;
    return result.success ? result.id : null;
  }

  async flushTrainingData(): Promise<number> {
    const count = await this.collector.flush(this.trainingDataPath);
    this.executionCount = 0;
    return count;
  }

  async retrain(_outputDir: string): Promise<never> {
    await this.flushTrainingData();
    throw new MembrainError(
      "Neural retraining is Python-only; use membrain-py " +
        "RetrieverTrainer to fine-tune the relevance classifier."
    );
  }
}
