/**
 * Anthropic LLM-based reranker.
 */

import type { RerankerConfig, SearchResults, RerankResults } from "../types";

import {
  BaseReranker,
  DEFAULT_TIMEOUT,
  LLM_SYSTEM_PROMPT,
  httpPost,
  buildRerankResults,
  buildLlmUserPrompt,
  parseLlmScores,
} from "./base";

const ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-5-20250929";

export class AnthropicReranker extends BaseReranker {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly topK: number;
  private readonly endpoint: string;
  private readonly timeout: number;

  /**
   * LLM-based reranker using the Anthropic Messages API.
   *
   * Supports: claude-sonnet-4-5-20250929, claude-haiku-4-5-20251001, etc.
   */
  constructor(config: RerankerConfig) {
    super();
    this.apiKey = config.apiKey;
    this.model = config.model ?? ANTHROPIC_DEFAULT_MODEL;
    this.topK = config.topK ?? 5;
    this.endpoint = config.endpoint ?? ANTHROPIC_MESSAGES_URL;
    this.timeout = config.timeout ?? DEFAULT_TIMEOUT;
  }

  async rerank(
    query: string,
    results: SearchResults,
    topK?: number,
  ): Promise<RerankResults> {
    const effectiveTopK = topK ?? this.topK;
    const memories = results.memories;
    if (memories.length === 0) {
      return { memories: [], model: this.model, provider: "anthropic", duration_ms: 0 };
    }

    const documents = memories.map((memory) => memory.content);
    const userPrompt = buildLlmUserPrompt(query, documents);

    const start = performance.now();
    const response = await httpPost(
      this.endpoint,
      {
        "x-api-key": this.apiKey,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
      },
      {
        model: this.model,
        max_tokens: 1024,
        system: LLM_SYSTEM_PROMPT,
        messages: [{ role: "user", content: userPrompt }],
        temperature: 0.0,
      },
      this.timeout,
    );
    const durationMs = Math.round(performance.now() - start);

    let responseText = "[]";
    const contentBlocks = response.content ?? [];
    for (const block of contentBlocks) {
      if (block.type === "text") {
        responseText = block.text ?? "[]";
        break;
      }
    }

    const scoredIndices = parseLlmScores(
      responseText,
      documents.length,
      effectiveTopK,
    );

    return buildRerankResults(
      memories,
      scoredIndices,
      this.model,
      "anthropic",
      durationMs,
    );
  }
}
