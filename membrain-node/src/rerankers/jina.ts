/**
 * Jina cross encoder reranker.
 */

import type { RerankerConfig, SearchResults, RerankResults } from "../types";

import {
  BaseReranker,
  DEFAULT_TIMEOUT,
  httpPost,
  buildRerankResults,
} from "./base";

const JINA_RERANK_URL = "https://api.jina.ai/v1/rerank";
const JINA_DEFAULT_MODEL = "jina-reranker-v2-base-multilingual";

export class JinaReranker extends BaseReranker {
  private readonly apiKey: string;
  private readonly model: string;
  private readonly topK: number;
  private readonly endpoint: string;
  private readonly timeout: number;

  /**
   * Cross encoder reranker using the Jina Rerank API.
   *
   * Supported models: jina-reranker-v2-base-multilingual, jina-colbert-v2.
   */
  constructor(config: RerankerConfig) {
    super();
    this.apiKey = config.apiKey;
    this.model = config.model ?? JINA_DEFAULT_MODEL;
    this.topK = config.topK ?? 5;
    this.endpoint = config.endpoint ?? JINA_RERANK_URL;
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
      return { memories: [], model: this.model, provider: "jina", duration_ms: 0 };
    }

    const documents = memories.map((memory) => memory.content);

    const start = performance.now();
    const response = await httpPost(
      this.endpoint,
      {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      {
        query,
        documents,
        model: this.model,
        top_n: effectiveTopK,
      },
      this.timeout,
    );
    const durationMs = Math.round(performance.now() - start);

    const scoredIndices: Array<[number, number]> = [];
    for (const item of response.results ?? []) {
      scoredIndices.push([item.index, item.relevance_score]);
    }
    scoredIndices.sort((a, b) => b[1] - a[1]);

    return buildRerankResults(
      memories,
      scoredIndices,
      this.model,
      "jina",
      durationMs,
    );
  }
}
