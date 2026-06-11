/**
 * LanceDB embedded vector database backend.
 *
 * Requires: npm install @lancedb/lancedb
 */

import { VectorBackend, VectorBackendCapabilities, VectorSearchResult } from "./base";

export interface LanceDBConfig {
  uri?: string;
  tableName?: string;
  dimension?: number;
}

export class LanceDBBackend extends VectorBackend {
  private database: any;
  private table: any;
  private tableName: string;
  private dimension: number;
  private lancedb: any;
  private initialized: boolean = false;

  constructor(config: LanceDBConfig = {}) {
    super();
    this.tableName = config.tableName ?? "membrain";
    this.dimension = config.dimension ?? 1536;

    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      this.lancedb = require("@lancedb/lancedb");
    } catch {
      throw new Error(
        "@lancedb/lancedb is required for LanceDBBackend. " +
        "Install it with: npm install @lancedb/lancedb",
      );
    }

    const uri = config.uri ?? "./lancedb_data";
    this.database = this.lancedb.connect(uri);
  }

  async ensureTable(): Promise<void> {
    if (this.initialized) {
      return;
    }

    const database = await this.database;
    const tableNames = await database.tableNames();

    if (tableNames.includes(this.tableName)) {
      this.table = await database.openTable(this.tableName);
    } else {
      // Create table with an initial empty-ish record to define schema
      const initialData = [{
        memory_id: "__init__",
        vector: new Array(this.dimension).fill(0),
        metadata: "{}",
      }];
      this.table = await database.createTable(this.tableName, initialData);
      // Remove the placeholder record
      await this.table.delete("memory_id = '__init__'");
    }

    this.initialized = true;
  }

  async store(
    memoryId: string,
    embedding: number[],
    metadata: Record<string, unknown>,
  ): Promise<void> {
    await this.ensureTable();
    await this.table.add([{
      memory_id: memoryId,
      vector: embedding,
      metadata: JSON.stringify(metadata),
    }]);
  }

  async search(
    queryEmbedding: number[],
    limit: number,
    filters?: Record<string, unknown>,
  ): Promise<VectorSearchResult[]> {
    await this.ensureTable();

    let query = this.table.search(queryEmbedding).limit(limit);

    if (filters) {
      const conditions: string[] = [];
      for (const [key, value] of Object.entries(filters)) {
        if (key === "memory_id" && typeof value === "string") {
          conditions.push(`memory_id = '${value}'`);
        }
      }
      const filterExpression = conditions.join(" AND ");
      if (filterExpression) {
        query = query.where(filterExpression);
      }
    }

    const results = await query.toArray();

    return results.map((row: any) => {
      const distance = row._distance ?? 0;
      const score = 1.0 / (1.0 + distance);
      const parsedMetadata = JSON.parse(row.metadata ?? "{}");
      return {
        memoryId: row.memory_id,
        score,
        metadata: parsedMetadata,
      };
    });
  }

  async delete(memoryId: string): Promise<boolean> {
    try {
      await this.ensureTable();
      await this.table.delete(`memory_id = '${memoryId}'`);
      return true;
    } catch {
      return false;
    }
  }

  async count(): Promise<number> {
    await this.ensureTable();
    return await this.table.countRows();
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.ensureTable();
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
      maxDimension: 65536,
      backendName: "lancedb",
    };
  }

  async close(): Promise<void> {
    // LanceDB is embedded, no connection to close
  }
}
