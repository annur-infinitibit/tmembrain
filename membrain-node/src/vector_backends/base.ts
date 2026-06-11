/**
 * Abstract base class for vector database backends.
 *
 * Implement this class to create custom vector database backends
 * that can be used with Membrain.
 */
export interface VectorSearchResult {
  memoryId: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface VectorBackendCapabilities {
  supportsMetadataFiltering: boolean;
  supportsHybridSearch: boolean;
  supportsBatchOperations: boolean;
  maxDimension: number;
  backendName: string;
}

export abstract class VectorBackend {
  abstract store(
    memoryId: string,
    embedding: number[],
    metadata: Record<string, unknown>,
  ): Promise<void>;

  abstract search(
    queryEmbedding: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorSearchResult[]>;

  abstract delete(memoryId: string): Promise<boolean>;

  abstract count(): Promise<number>;

  abstract healthCheck(): Promise<boolean>;

  abstract getCapabilities(): VectorBackendCapabilities;

  async close(): Promise<void> {
    // Override if cleanup is needed
  }
}
