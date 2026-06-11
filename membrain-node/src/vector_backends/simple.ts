/**
 * Simple in-memory vector backend for development and testing.
 */

import { VectorBackend, VectorBackendCapabilities, VectorSearchResult } from "./base";

export class SimpleInMemoryBackend extends VectorBackend {
  private vectors: Map<string, { embedding: number[]; metadata: Record<string, unknown> }> =
    new Map();

  async store(
    memoryId: string,
    embedding: number[],
    metadata: Record<string, unknown>,
  ): Promise<void> {
    this.vectors.set(memoryId, { embedding, metadata });
  }

  async search(
    queryEmbedding: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorSearchResult[]> {
    const results: VectorSearchResult[] = [];

    for (const [memoryId, { embedding, metadata }] of this.vectors) {
      if (filters && !this.matchesFilters(metadata, filters)) {
        continue;
      }
      const score = this.cosineSimilarity(queryEmbedding, embedding);
      results.push({ memoryId, score, metadata });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, limit);
  }

  async delete(memoryId: string): Promise<boolean> {
    return this.vectors.delete(memoryId);
  }

  async count(): Promise<number> {
    return this.vectors.size;
  }

  async healthCheck(): Promise<boolean> {
    return true;
  }

  getCapabilities(): VectorBackendCapabilities {
    return {
      supportsMetadataFiltering: true,
      supportsHybridSearch: false,
      supportsBatchOperations: false,
      maxDimension: 10000,
      backendName: "simple_in_memory",
    };
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error(`Vector dimensions don't match: ${a.length} vs ${b.length}`);
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      const ai = a[i]!;
      const bi = b[i]!;
      dotProduct += ai * bi;
      normA += ai * ai;
      normB += bi * bi;
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (normA * normB);
  }

  private matchesFilters(
    metadata: Record<string, unknown>,
    filters: Record<string, unknown>,
  ): boolean {
    for (const [key, value] of Object.entries(filters)) {
      if (metadata[key] !== value) return false;
    }
    return true;
  }
}
