/**
 * Milvus vector database backend.
 *
 * Requires: npm install @zilliz/milvus2-sdk-node
 */

import { VectorBackend, VectorBackendCapabilities, VectorSearchResult } from "./base";

export interface MilvusConfig {
  address?: string;
  collectionName?: string;
  dimension?: number;
  metricType?: string;
  token?: string;
}

export class MilvusBackend extends VectorBackend {
  private client: any;
  private collectionName: string;
  private dimension: number;

  constructor(config: MilvusConfig = {}) {
    super();
    this.collectionName = config.collectionName ?? "membrain";
    this.dimension = config.dimension ?? 1536;

    let MilvusClient: any;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      MilvusClient = require("@zilliz/milvus2-sdk-node").MilvusClient;
    } catch {
      throw new Error(
        "@zilliz/milvus2-sdk-node is required for MilvusBackend. " +
        "Install it with: npm install @zilliz/milvus2-sdk-node",
      );
    }

    this.client = new MilvusClient({
      address: config.address ?? "localhost:19530",
      token: config.token,
    });
  }

  async ensureCollection(): Promise<void> {
    const has = await this.client.hasCollection({ collection_name: this.collectionName });
    if (!has.value) {
      await this.client.createCollection({
        collection_name: this.collectionName,
        fields: [
          { name: "id", data_type: "VarChar", is_primary_key: true, max_length: 36 },
          { name: "vector", data_type: "FloatVector", dim: this.dimension },
        ],
        enable_dynamic_field: true,
      });
    }
  }

  async store(
    memoryId: string,
    embedding: number[],
    metadata: Record<string, unknown>,
  ): Promise<void> {
    await this.client.insert({
      collection_name: this.collectionName,
      data: [{ id: memoryId, vector: embedding, ...metadata }],
    });
  }

  async search(
    queryEmbedding: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorSearchResult[]> {
    let filterExpression: string | undefined;
    if (filters) {
      const conditions = Object.entries(filters).map(([key, value]) => {
        if (typeof value === "string") return `${key} == "${value}"`;
        return `${key} == ${value}`;
      });
      filterExpression = conditions.join(" && ");
    }

    const results = await this.client.search({
      collection_name: this.collectionName,
      data: [queryEmbedding],
      limit,
      filter: filterExpression,
      output_fields: ["*"],
    });

    return (results.results ?? []).map((hit: any) => {
      const score = hit.score <= 1.0 ? hit.score : 1.0 / (1.0 + hit.score);
      const metadata: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(hit)) {
        if (key !== "id" && key !== "vector" && key !== "score") {
          metadata[key] = value;
        }
      }
      return { memoryId: hit.id, score, metadata };
    });
  }

  async delete(memoryId: string): Promise<boolean> {
    try {
      await this.client.delete({
        collection_name: this.collectionName,
        filter: `id == "${memoryId}"`,
      });
      return true;
    } catch {
      return false;
    }
  }

  async count(): Promise<number> {
    const stats = await this.client.getCollectionStatistics({
      collection_name: this.collectionName,
    });
    return stats.data?.row_count ?? 0;
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.client.listCollections();
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
      maxDimension: 32768,
      backendName: "milvus",
    };
  }

  async close(): Promise<void> {
    if (this.client.close) {
      await this.client.close();
    }
  }
}
