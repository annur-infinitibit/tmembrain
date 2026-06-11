/**
 * ChromaDB vector database backend.
 *
 * Requires: npm install chromadb
 */

import { VectorBackend, VectorBackendCapabilities, VectorSearchResult } from "./base";

export interface ChromaConfig {
  path?: string;
  collectionName?: string;
}

export class ChromaBackend extends VectorBackend {
  private collection: any;
  private clientInstance: any;

  constructor(private config: ChromaConfig = {}) {
    super();
  }

  async initialize(): Promise<void> {
    let chromadb: any;
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      chromadb = require("chromadb");
    } catch {
      throw new Error(
        "chromadb is required for ChromaBackend. " +
        "Install it with: npm install chromadb",
      );
    }

    const collectionName = this.config.collectionName ?? "membrain";

    if (this.config.path) {
      this.clientInstance = new chromadb.ChromaClient({ path: this.config.path });
    } else {
      this.clientInstance = new chromadb.ChromaClient();
    }

    this.collection = await this.clientInstance.getOrCreateCollection({
      name: collectionName,
      metadata: { description: "Membrain memory storage" },
    });
  }

  async store(
    memoryId: string,
    embedding: number[],
    metadata: Record<string, unknown>,
  ): Promise<void> {
    await this.collection.upsert({
      ids: [memoryId],
      embeddings: [embedding],
      metadatas: [metadata],
    });
  }

  async search(
    queryEmbedding: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorSearchResult[]> {
    const results = await this.collection.query({
      queryEmbeddings: [queryEmbedding],
      nResults: limit,
      where: filters ?? undefined,
      include: ["metadatas", "distances"],
    });

    const output: VectorSearchResult[] = [];
    if (results.ids?.[0]) {
      for (let i = 0; i < results.ids[0].length; i++) {
        const distance = results.distances?.[0]?.[i] ?? 0;
        const score = 1.0 / (1.0 + distance);
        output.push({
          memoryId: results.ids[0][i],
          score,
          metadata: results.metadatas?.[0]?.[i] ?? {},
        });
      }
    }
    return output;
  }

  async delete(memoryId: string): Promise<boolean> {
    try {
      await this.collection.delete({ ids: [memoryId] });
      return true;
    } catch {
      return false;
    }
  }

  async count(): Promise<number> {
    return await this.collection.count();
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.collection.count();
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
      maxDimension: 4096,
      backendName: "chromadb",
    };
  }
}
