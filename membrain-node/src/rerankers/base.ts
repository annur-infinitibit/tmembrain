/**
 * Base reranker class, error type, and shared helpers.
 */

import type {
  MemoryEntry,
  RerankResult,
  RerankResults,
  SearchResults,
} from "../types";

export class RerankerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RerankerError";
  }
}

const DEFAULT_TIMEOUT = 30_000;
const MAX_RETRIES = 3;

export abstract class BaseReranker {
  /**
   * Rerank search results by semantic relevance to the query.
   *
   * @param query - The original search query.
   * @param results - SearchResults from membrain.search().
   * @param topK - Number of top results to keep after reranking.
   * @returns RerankResults ordered by descending relevance_score.
   */
  abstract rerank(
    query: string,
    results: SearchResults,
    topK?: number,
  ): Promise<RerankResults>;
}

export { DEFAULT_TIMEOUT };

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function httpPost(
  url: string,
  headers: Record<string, string>,
  body: Record<string, unknown>,
  timeout: number,
): Promise<Record<string, any>> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timer);

      if (response.ok) {
        return (await response.json()) as Record<string, any>;
      }

      const status = response.status;
      if (status === 429 || status >= 500) {
        const wait = 2 ** attempt * 1000;
        lastError = new RerankerError(
          `Reranker API returned HTTP ${status}`,
        );
        await sleep(wait);
        continue;
      }

      const responseBody = await response.text();
      throw new RerankerError(
        `Reranker API returned HTTP ${status}: ${responseBody}`,
      );
    } catch (error: unknown) {
      clearTimeout(timer);
      if (error instanceof RerankerError) {
        throw error;
      }
      lastError =
        error instanceof Error ? error : new Error(String(error));
      const wait = 2 ** attempt * 1000;
      await sleep(wait);
    }
  }

  throw new RerankerError(
    `Reranker API failed after ${MAX_RETRIES} retries: ${lastError?.message ?? "unknown error"}`,
  );
}

export function buildRerankResults(
  memories: MemoryEntry[],
  scoredIndices: Array<[number, number]>,
  model: string,
  provider: string,
  durationMs: number,
): RerankResults {
  const reranked: RerankResult[] = [];
  for (const [index, relevanceScore] of scoredIndices) {
    const memory = memories[index];
    if (memory === undefined) {
      continue;
    }
    reranked.push({
      id: memory.id,
      content: memory.content,
      score: memory.score,
      relevance_score: relevanceScore,
      memory_type: memory.memory_type,
    });
  }
  return {
    memories: reranked,
    model,
    provider,
    duration_ms: durationMs,
  };
}

// ---------------------------------------------------------------------------
// LLM prompt helpers
// ---------------------------------------------------------------------------

export const LLM_SYSTEM_PROMPT =
  "You are a relevance scoring assistant. Given a query and a list of " +
  "documents, score each document's relevance to the query on a scale " +
  "of 0 to 10. Respond with ONLY a JSON array of objects: " +
  '[{"index": 0, "score": 7}, ...]';

export function buildLlmUserPrompt(query: string, documents: string[]): string {
  const parts: string[] = [`Query: ${query}\n\nDocuments:`];
  for (let index = 0; index < documents.length; index++) {
    parts.push(`[${index}] ${documents[index]}`);
  }
  parts.push("\nScore each document's relevance to the query (0-10).");
  return parts.join("\n");
}

export function parseLlmScores(
  responseText: string,
  documentCount: number,
  topK: number,
): Array<[number, number]> {
  let scores: unknown;
  try {
    scores = JSON.parse(responseText);
  } catch {
    throw new RerankerError(
      `LLM reranker returned invalid JSON: ${responseText.slice(0, 200)}`,
    );
  }

  if (!Array.isArray(scores)) {
    throw new RerankerError(
      `LLM reranker returned non-array JSON: ${typeof scores}`,
    );
  }

  const scoredIndices: Array<[number, number]> = [];
  for (const item of scores) {
    const index = item?.index;
    const score = item?.score;
    if (
      typeof index === "number" &&
      typeof score === "number" &&
      index >= 0 &&
      index < documentCount
    ) {
      const normalized = Math.max(0.0, Math.min(1.0, score / 10.0));
      scoredIndices.push([index, normalized]);
    }
  }

  scoredIndices.sort((a, b) => b[1] - a[1]);
  return scoredIndices.slice(0, topK);
}
