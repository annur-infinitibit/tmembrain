/**
 * Qdrant vector database backend.
 *
 * Requires: npm install @qdrant/js-client-rest
 */

import { VectorBackend, VectorBackendCapabilities, VectorSearchResult } from "./base";

export interface QdrantConfig {
  url?: string;
  collectionName?: string;
  apiKey?: string;
  dimension?: number;
}

export class QdrantBackend extends VectorBackend {
  private client: any;
  private collectionName: string;
  private dimension: number;

  constructor(config: QdrantConfig = {}) {
    super();
    this.collectionName = config.collectionName ?? "membrain";
    this.dimension = config.dimension ?? 1536;

    let QdrantClient: any;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      QdrantClient = require("@qdrant/js-client-rest").QdrantClient;
    } catch {
      throw new Error(
        "@qdrant/js-client-rest is required for QdrantBackend. " +
        "Install it with: npm install @qdrant/js-client-rest",
      );
    }

    this.client = new QdrantClient({
      url: config.url ?? "http://localhost:6333",
      apiKey: config.apiKey,
    });
  }

  async ensureCollection(): Promise<void> {
    try {
      await this.client.getCollection(this.collectionName);
    } catch {
      await this.client.createCollection(this.collectionName, {
        vectors: { size: this.dimension, distance: "Cosine" },
      });
    }
  }

  async store(
    memoryId: string,
    embedding: number[],
    metadata: Record<string, unknown>,
  ): Promise<void> {
    await this.client.upsert(this.collectionName, {
      wait: true,
      points: [{ id: memoryId, vector: embedding, payload: metadata }],
    });
  }

  async search(
    queryEmbedding: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorSearchResult[]> {
    const queryFilter = filters
      ? {
          must: Object.entries(filters).map(([key, value]) => ({
            key,
            match: { value },
          })),
        }
      : undefined;

    const results = await this.client.search(this.collectionName, {
      vector: queryEmbedding,
      limit,
      filter: queryFilter,
      with_payload: true,
    });

    return results.map((result: any) => ({
      memoryId: String(result.id),
      score: result.score,
      metadata: result.payload ?? {},
    }));
  }

  async delete(memoryId: string): Promise<boolean> {
    try {
      await this.client.delete(this.collectionName, {
        wait: true,
        points: [memoryId],
      });
      return true;
    } catch {
      return false;
    }
  }

  async count(): Promise<number> {
    const info = await this.client.getCollection(this.collectionName);
    return info.points_count ?? 0;
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.client.getCollection(this.collectionName);
      return true;
    } catch {
      return false;
    }
  }

  getCapabilities(): VectorBackendCapabilities {
    return {
      supportsMetadataFiltering: true,
      supportsHybridSearch: true,
      supportsBatchOperations: true,
      maxDimension: 65536,
      backendName: "qdrant",
    };
  }

  async close(): Promise<void> {
    // Qdrant JS client doesn't require explicit close
  }
}
