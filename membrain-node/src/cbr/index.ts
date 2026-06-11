/**
 * Case-based reasoning module for Membrain.
 *
 * Provides retrieval of past experience cases, prompt building for in-context
 * learning, training data collection, and the experience replay loop. Neural
 * classifier training and parametric retrievers remain Python-only.
 */

export { CasePromptBuilder, formatPlan } from "./prompt_builder";
export type { CasePromptBuilderOptions } from "./prompt_builder";
export { CaseRetriever, NonParametricRetriever } from "./retriever";
export { TrainingDataCollector } from "./training_data";
export { ExperienceReplay } from "./experience_replay";
export type {
  ExperienceReplayOptions,
  RecordExecutionInput,
} from "./experience_replay";
