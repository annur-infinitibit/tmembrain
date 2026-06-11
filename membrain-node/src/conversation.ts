/**
 * Automatic conversation management for Membrain.
 *
 * Wraps `MembrainClient` to track session state, retrieve relevant memories
 * before each turn, extract and store memorable information from each turn,
 * and learn from conversation outcomes via case-based reasoning.
 */

import { randomUUID } from "node:crypto";

import { MembrainClient } from "./client";
import type { CaseSearchResults, MemoryEntry, SearchResults } from "./types";

type Message = { role: string; content: string };

/** LLM callable: takes messages array, returns response string or Promise. */
export type LLMCallable = (messages: Message[]) => string | Promise<string>;

/** Strict-async LLM callable used by {@link AsyncConversation}. */
export type AsyncLLMCallable = (messages: Message[]) => Promise<string>;

export interface ConversationOptions {
  client?: MembrainClient;
  systemPrompt?: string;
  memoryLimit?: number;
  autoExtract?: boolean;
  historyLimit?: number;
  onExtractionError?: (err: unknown) => void;
}

export interface AsyncConversationOptions {
  client?: MembrainClient;
  systemPrompt?: string;
  memoryLimit?: number;
  autoExtract?: boolean;
  historyLimit?: number;
  onExtractionError?: (err: unknown) => void;
}

const EXTRACTION_PROMPT = `\
Analyze the following conversation turn and extract any memorable information.

Return a JSON array of objects. Each object must have:
- "type": one of "fact", "preference", "observation", "entity", "concept"
- Plus type-specific fields:

For "fact":
  {"type": "fact", "statement": "...", "confidence": 0.0-1.0}

For "preference":
  {"type": "preference", "holder": "user", "subject": "...", "preference": "...", "strength": "weak"|"moderate"|"strong"|"absolute"}

For "observation":
  {"type": "observation", "content": "..."}

For "entity":
  {"type": "entity", "name": "...", "entity_type": "person"|"organization"|"place"|"product"|"other"}

For "concept":
  {"type": "concept", "name": "...", "definition": "..."}

Rules:
- Only extract information that is worth remembering across conversations.
- Do not extract trivial greetings or filler.
- Return [] if nothing is worth remembering.
- Return valid JSON only, no markdown or explanation.`;

const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful assistant with access to long-term memory. " +
  "Use the relevant memories provided to personalize your responses " +
  "and maintain continuity across conversations.";

function formatMemoriesForPrompt(memories: MemoryEntry[]): string {
  if (memories.length === 0) return "";
  const lines = ["## Relevant Memories", ""];
  for (const memory of memories) {
    lines.push(`- [${memory.memory_type}] ${memory.content}`);
  }
  return lines.join("\n");
}

function parseExtractionResponse(response: string): Array<Record<string, any>> {
  let cleaned = response.trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned
      .split("\n")
      .filter((line) => !line.trim().startsWith("```"))
      .join("\n");
  }
  try {
    const parsed = JSON.parse(cleaned);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function retrieveCaseContext(client: MembrainClient, query: string): string {
  const cases: CaseSearchResults = client.searchCases(query, 3, 0.5);
  if (
    cases.positive_cases.length === 0 &&
    cases.negative_cases.length === 0
  ) {
    return "";
  }
  const sections = ["## Past Conversation Experience", ""];
  for (const c of cases.positive_cases.slice(0, 2)) {
    sections.push(
      `- Previously when asked about '${c.problem.slice(0, 60)}', ` +
        `a successful approach was: ${c.outcome.slice(0, 100)}`,
    );
  }
  for (const c of cases.negative_cases.slice(0, 1)) {
    sections.push(
      `- Avoid: when asked about '${c.problem.slice(0, 60)}', ` +
        `this approach failed: ${c.outcome.slice(0, 100)}`,
    );
  }
  return sections.join("\n");
}

function buildMessages(params: {
  client: MembrainClient;
  systemPrompt: string;
  history: Message[];
  historyLimit: number;
  userMessage: string;
  memories: MemoryEntry[];
}): Message[] {
  let systemContent = params.systemPrompt;
  const memorySection = formatMemoriesForPrompt(params.memories);
  if (memorySection) {
    systemContent = `${systemContent}\n\n${memorySection}`;
  }
  const caseSection = retrieveCaseContext(params.client, params.userMessage);
  if (caseSection) {
    systemContent = `${systemContent}\n\n${caseSection}`;
  }
  const messages: Message[] = [{ role: "system", content: systemContent }];
  messages.push(...params.history.slice(-(params.historyLimit * 2)));
  messages.push({ role: "user", content: params.userMessage });
  return messages;
}

function storeExtractedMemory(
  client: MembrainClient,
  item: Record<string, any>,
): void {
  const memoryType = item.type ?? "";
  if (memoryType === "fact") {
    client.storeFact(item.statement ?? "", item.confidence ?? 0.8);
  } else if (memoryType === "preference") {
    client.storePreference(
      item.holder ?? "user",
      item.subject ?? "",
      item.preference ?? "",
      item.strength ?? "moderate",
    );
  } else if (memoryType === "observation") {
    client.storeObservation(item.content ?? "");
  } else if (memoryType === "entity") {
    client.storeEntity(item.name ?? "", item.entity_type ?? "other");
  } else if (memoryType === "concept") {
    client.storeConcept(item.name ?? "", item.definition ?? "");
  }
}

function computeEndOfCase(history: Message[]): {
  firstUserMessage: string;
  turnCount: number;
  lastExchange: string;
} {
  const firstUserMessage =
    history.find((turn) => turn.role === "user")?.content ?? "";
  const turnCount = Math.floor(history.length / 2);
  const [prev, curr] = history.slice(-2);
  const lastExchange =
    prev && curr ? `User: ${prev.content}\nAssistant: ${curr.content}` : "";
  return { firstUserMessage, turnCount, lastExchange };
}

/**
 * Automatic conversation manager backed by Membrain.
 *
 * The LLM callable may be sync or async. See {@link AsyncConversation} for
 * the strict-async variant that mirrors Python's `AsyncConversation`.
 */
export class Conversation {
  private llm: LLMCallable;
  private client: MembrainClient;
  private ownsClient: boolean;
  private systemPrompt: string;
  private memoryLimit: number;
  private autoExtract: boolean;
  private historyLimit: number;
  private onExtractionError?: (err: unknown) => void;
  private _sessionId: string;
  private _history: Message[];

  constructor(llmCallable: LLMCallable, options: ConversationOptions = {}) {
    this.llm = llmCallable;
    this.ownsClient = !options.client;
    this.client = options.client ?? new MembrainClient();
    this.systemPrompt = options.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;
    this.memoryLimit = options.memoryLimit ?? 10;
    this.autoExtract = options.autoExtract ?? true;
    this.historyLimit = options.historyLimit ?? 50;
    this.onExtractionError = options.onExtractionError;
    this._sessionId = randomUUID();
    this._history = [];
  }

  get sessionId(): string {
    return this._sessionId;
  }

  get history(): Message[] {
    return [...this._history];
  }

  async reply(userMessage: string): Promise<string> {
    const results: SearchResults = this.client.search(
      userMessage,
      this.memoryLimit,
    );
    const messages = buildMessages({
      client: this.client,
      systemPrompt: this.systemPrompt,
      history: this._history,
      historyLimit: this.historyLimit,
      userMessage,
      memories: results.memories,
    });

    const assistantResponse = await Promise.resolve(this.llm(messages));

    this._history.push({ role: "user", content: userMessage });
    this._history.push({ role: "assistant", content: assistantResponse });

    this.client.storeEvent(
      "conversation_turn",
      `User: ${userMessage}\nAssistant: ${assistantResponse}`,
    );

    if (this.autoExtract) {
      await this.extractAndStore(userMessage, assistantResponse);
    }

    return assistantResponse;
  }

  end(outcome: string = "", reward: number = 1.0): void {
    if (this._history.length === 0) return;
    const { firstUserMessage, turnCount, lastExchange } = computeEndOfCase(
      this._history,
    );
    this.client.storeCase(
      firstUserMessage,
      `Conversation with ${turnCount} exchanges`,
      outcome || lastExchange,
      reward,
    );
  }

  close(): void {
    if (this.ownsClient) {
      this.client.close();
    }
  }

  private async extractAndStore(
    userMessage: string,
    assistantResponse: string,
  ): Promise<void> {
    const turnText = `User: ${userMessage}\nAssistant: ${assistantResponse}`;
    const extractionMessages: Message[] = [
      { role: "system", content: EXTRACTION_PROMPT },
      { role: "user", content: turnText },
    ];

    let extractionResponse: string;
    try {
      extractionResponse = await Promise.resolve(this.llm(extractionMessages));
    } catch (err) {
      if (this.onExtractionError) {
        this.onExtractionError(err);
      } else {
        console.warn("Membrain: memory extraction LLM call failed", err);
      }
      return;
    }

    for (const item of parseExtractionResponse(extractionResponse)) {
      try {
        storeExtractedMemory(this.client, item);
      } catch {
        // Non-fatal: skip individual memory storage failures.
      }
    }
  }
}

/**
 * Async counterpart to {@link Conversation}.
 *
 * Accepts a strictly async LLM callable and exposes async `close()`/`end()`
 * for symmetry with Python's `AsyncConversation`. Storage calls remain
 * synchronous under the hood — koffi FFI is in-process and fast.
 */
export class AsyncConversation {
  private llm: AsyncLLMCallable;
  private client: MembrainClient;
  private ownsClient: boolean;
  private systemPrompt: string;
  private memoryLimit: number;
  private autoExtract: boolean;
  private historyLimit: number;
  private onExtractionError?: (err: unknown) => void;
  private _sessionId: string;
  private _history: Message[];

  constructor(
    llmCallable: AsyncLLMCallable,
    options: AsyncConversationOptions = {},
  ) {
    this.llm = llmCallable;
    this.ownsClient = !options.client;
    this.client = options.client ?? new MembrainClient();
    this.systemPrompt = options.systemPrompt ?? DEFAULT_SYSTEM_PROMPT;
    this.memoryLimit = options.memoryLimit ?? 10;
    this.autoExtract = options.autoExtract ?? true;
    this.historyLimit = options.historyLimit ?? 50;
    this.onExtractionError = options.onExtractionError;
    this._sessionId = randomUUID();
    this._history = [];
  }

  get sessionId(): string {
    return this._sessionId;
  }

  get history(): Message[] {
    return [...this._history];
  }

  async reply(userMessage: string): Promise<string> {
    const results: SearchResults = this.client.search(
      userMessage,
      this.memoryLimit,
    );
    const messages = buildMessages({
      client: this.client,
      systemPrompt: this.systemPrompt,
      history: this._history,
      historyLimit: this.historyLimit,
      userMessage,
      memories: results.memories,
    });

    const assistantResponse = await this.llm(messages);

    this._history.push({ role: "user", content: userMessage });
    this._history.push({ role: "assistant", content: assistantResponse });

    this.client.storeEvent(
      "conversation_turn",
      `User: ${userMessage}\nAssistant: ${assistantResponse}`,
    );

    if (this.autoExtract) {
      await this.extractAndStore(userMessage, assistantResponse);
    }

    return assistantResponse;
  }

  async end(outcome: string = "", reward: number = 1.0): Promise<void> {
    if (this._history.length === 0) return;
    const { firstUserMessage, turnCount, lastExchange } = computeEndOfCase(
      this._history,
    );
    this.client.storeCase(
      firstUserMessage,
      `Conversation with ${turnCount} exchanges`,
      outcome || lastExchange,
      reward,
    );
  }

  async close(): Promise<void> {
    if (this.ownsClient) {
      this.client.close();
    }
  }

  private async extractAndStore(
    userMessage: string,
    assistantResponse: string,
  ): Promise<void> {
    const turnText = `User: ${userMessage}\nAssistant: ${assistantResponse}`;
    const extractionMessages: Message[] = [
      { role: "system", content: EXTRACTION_PROMPT },
      { role: "user", content: turnText },
    ];

    let extractionResponse: string;
    try {
      extractionResponse = await this.llm(extractionMessages);
    } catch (err) {
      if (this.onExtractionError) {
        this.onExtractionError(err);
      } else {
        console.warn("Membrain: memory extraction LLM call failed", err);
      }
      return;
    }

    for (const item of parseExtractionResponse(extractionResponse)) {
      try {
        storeExtractedMemory(this.client, item);
      } catch {
        // Non-fatal: skip individual memory storage failures.
      }
    }
  }
}
