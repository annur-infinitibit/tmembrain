/**
 * OpenAI LLM-based reranker.
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

const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions";
const OPENAI_DEFAULT_MODEL = "gpt-4o-mini";

export class OpenAIReranker extends BaseReranker {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly topK: number;
  private readonly endpoint: string;
  private readonly timeout: number;

  /**
   * LLM-based reranker using the OpenAI Chat Completions API.
   *
   * Supports any chat model: gpt-4o-mini, gpt-4o, etc.
   */
  constructor(config: RerankerConfig) {
    super();
    this.apiKey = config.apiKey;
    this.model = config.model ?? OPENAI_DEFAULT_MODEL;
    this.topK = config.topK ?? 5;
    this.endpoint = config.endpoint ?? OPENAI_CHAT_URL;
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
      return { memories: [], model: this.model, provider: "openai", duration_ms: 0 };
    }

    const documents = memories.map((memory) => memory.content);
    const userPrompt = buildLlmUserPrompt(query, documents);

    const start = performance.now();
    const response = await httpPost(
      this.endpoint,
      {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      {
        model: this.model,
        messages: [
          { role: "system", content: LLM_SYSTEM_PROMPT },
          { role: "user", content: userPrompt },
        ],
        temperature: 0.0,
      },
      this.timeout,
    );
    const durationMs = Math.round(performance.now() - start);

    const responseText =
      response.choices?.[0]?.message?.content ?? "[]";
    const scoredIndices = parseLlmScores(
      responseText,
      documents.length,
      effectiveTopK,
    );

    return buildRerankResults(
      memories,
      scoredIndices,
      this.model,
      "openai",
      durationMs,
    );
  }
}
