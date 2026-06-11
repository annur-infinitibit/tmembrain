/**
 * Pinecone vector database backend.
 *
 * Requires: npm install @pinecone-database/pinecone
 */

import { VectorBackend, VectorBackendCapabilities, VectorSearchResult } from "./base";

export interface PineconeConfig {
  apiKey: string;
  indexName?: string;
  namespace?: string;
  dimension?: number;
  metric?: string;
}

export class PineconeBackend extends VectorBackend {
  private index: any;
  private namespace: string;

  constructor(config: PineconeConfig) {
    super();
    this.namespace = config.namespace ?? "default";

    let Pinecone: any;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      Pinecone = require("@pinecone-database/pinecone").Pinecone;
    } catch {
      throw new Error(
        "@pinecone-database/pinecone is required for PineconeBackend. " +
        "Install it with: npm install @pinecone-database/pinecone",
      );
    }

    const client = new Pinecone({ apiKey: config.apiKey });
    this.index = client.index(config.indexName ?? "membrain");
  }

  async store(
    memoryId: string,
    embedding: number[],
    metadata: Record<string, unknown>,
  ): Promise<void> {
    await this.index.namespace(this.namespace).upsert([
      { id: memoryId, values: embedding, metadata },
    ]);
  }

  async search(
    queryEmbedding: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorSearchResult[]> {
    const pineconeFilter = filters
      ? Object.fromEntries(
          Object.entries(filters).map(([key, value]) => [key, { $eq: value }]),
        )
      : undefined;

    const results = await this.index.namespace(this.namespace).query({
      vector: queryEmbedding,
      topK: limit,
      includeMetadata: true,
      filter: pineconeFilter,
    });

    return (results.matches ?? []).map((match: any) => ({
      memoryId: match.id,
      score: match.score ?? 0,
      metadata: match.metadata ?? {},
    }));
  }

  async delete(memoryId: string): Promise<boolean> {
    try {
      await this.index.namespace(this.namespace).deleteOne(memoryId);
      return true;
    } catch {
      return false;
    }
  }

  async count(): Promise<number> {
    const stats = await this.index.describeIndexStats();
    return stats.totalRecordCount ?? 0;
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.index.describeIndexStats();
      return true;
    } catch {
      return false;
    }
  }

  getCapabilities(): VectorBackendCapabilities {
    return {
      supportsMetadataFiltering: true,
      supportsHybridSearch: false,
      supportsBatchOperations: true,
      maxDimension: 20000,
      backendName: "pinecone",
    };
  }
}
