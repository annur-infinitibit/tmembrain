/**
 * Case retrieval strategies for case-based reasoning.
 *
 * Exposes an abstract {@link CaseRetriever} base class and a
 * {@link NonParametricRetriever} that uses embedding similarity via
 * {@link MembrainClient.searchCases}.
 */

import type { MembrainClient } from "../client";
import type { CaseSearchResults } from "../types";

export abstract class CaseRetriever {
  abstract retrieve(
    query: string,
    limit?: number,
    positiveRewardThreshold?: number
  ): Promise<CaseSearchResults> | CaseSearchResults;
}

export class NonParametricRetriever extends CaseRetriever {
  private readonly client: MembrainClient;

  constructor(client: MembrainClient) {
    super();
    this.client = client;
  }

  retrieve(
    query: string,
    limit: number = 5,
    positiveRewardThreshold: number = 0.5
  ): CaseSearchResults {
    return this.client.searchCases(query, limit, positiveRewardThreshold);
  }
}
