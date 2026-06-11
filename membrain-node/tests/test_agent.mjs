/**
 * Tests for the Membrain agent framework (Node.js).
 *
 * Core planner/executor/agent logic is exercised with deterministic in-process
 * LLM callables. A single smoke test runs against the real OpenAI API when
 * OPENAI_API_KEY is set.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";

import {
  CasePromptBuilder,
  MembrainAgent,
  MembrainClient,
  MembrainExecutor,
  MembrainPlanner,
  NonParametricRetriever,
} from "../dist/index.js";

function loadApiKey() {
  if (process.env.OPENAI_API_KEY) return process.env.OPENAI_API_KEY;
  const envPath = join(import.meta.dirname, "..", "..", ".env");
  if (!existsSync(envPath)) return null;
  for (const line of readFileSync(envPath, "utf-8").split("\n")) {
    if (line.startsWith("OPENAI_API_KEY=")) {
      return line.split("=")[1].trim();
    }
  }
  return null;
}

function makeTmpPath() {
  return join(tmpdir(), `membrain_agent_${randomUUID()}`);
}

function planJson(steps) {
  const payload = {
    plan: steps.map((description, index) => ({ id: index + 1, description })),
  };
  return JSON.stringify(payload);
}

class EmptyRetriever extends NonParametricRetriever {
  constructor() {
    super(null);
  }
  retrieve() {
    return { positive_cases: [], negative_cases: [], duration_ms: 0 };
  }
}

describe("MembrainPlanner", () => {
  it("returns a structured plan", async () => {
    const calls = [];
    const llm = async (messages) => {
      calls.push(messages);
      return planJson(["Gather requirements", "Write code", "Deploy"]);
    };

    const planner = new MembrainPlanner(
      new EmptyRetriever(),
      new CasePromptBuilder(),
      llm,
    );
    const plan = await planner.plan("Ship a new microservice");
    assert.deepEqual(
      plan.steps.map((step) => step.description),
      ["Gather requirements", "Write code", "Deploy"],
    );
    assert.equal(plan.steps[0].id, 1);
    assert.equal(calls[0][0].role, "system");
    assert.match(calls[0][0].content, /task planner/i);
  });

  it("strips markdown fences", async () => {
    const llm = async () => "```json\n" + planJson(["Only step"]) + "\n```";
    const planner = new MembrainPlanner(
      new EmptyRetriever(),
      new CasePromptBuilder(),
      llm,
    );
    const plan = await planner.plan("Query");
    assert.equal(plan.steps.length, 1);
    assert.equal(plan.steps[0].description, "Only step");
  });

  it("falls back to raw text on invalid JSON", async () => {
    const llm = async () => "not-valid-json";
    const planner = new MembrainPlanner(
      new EmptyRetriever(),
      new CasePromptBuilder(),
      llm,
    );
    const plan = await planner.plan("Query");
    assert.equal(plan.steps.length, 1);
    assert.equal(plan.steps[0].description, "not-valid-json");
  });
});

describe("MembrainExecutor", () => {
  it("dispatches a registered tool", async () => {
    const llm = async () => "TOOL_CALL: echo('hello world')";
    const executor = new MembrainExecutor(llm);
    executor.registerTool("echo", (value) => `echoed: ${value}`);
    const result = await executor.execute({ id: 1, description: "Echo" });
    assert.equal(result.success, true);
    assert.equal(result.output, "echoed: hello world");
    assert.deepEqual(executor.availableTools, ["echo"]);
  });

  it("passes through plain responses", async () => {
    const llm = async () => "direct answer";
    const executor = new MembrainExecutor(llm);
    const result = await executor.execute({ id: 1, description: "Task" });
    assert.equal(result.output, "direct answer");
    assert.equal(result.success, true);
  });

  it("captures LLM failures", async () => {
    const llm = async () => {
      throw new Error("LLM upstream failure");
    };
    const executor = new MembrainExecutor(llm);
    const result = await executor.execute({ id: 1, description: "Task" });
    assert.equal(result.success, false);
    assert.match(result.error, /LLM upstream failure/);
  });

  it("reports unknown tool names", async () => {
    const llm = async () => "TOOL_CALL: missing()";
    const executor = new MembrainExecutor(llm);
    const result = await executor.execute({ id: 1, description: "Task" });
    assert.equal(result.success, true);
    assert.match(result.output, /Unknown tool: missing/);
  });
});

describe("MembrainAgent", () => {
  it("runs plan-execute end-to-end", async () => {
    const client = new MembrainClient({ storage: { path: makeTmpPath() } });
    try {
      let count = 0;
      const llm = async (messages) => {
        count += 1;
        const system = messages[0]?.content ?? "";
        if (system.toLowerCase().includes("task planner")) {
          return planJson(["Do the task"]);
        }
        return "completed";
      };
      const agent = new MembrainAgent(client, llm);
      const output = await agent.run("Diagnose the production outage");
      assert.equal(output, "completed");
      assert.ok(count >= 2);
    } finally {
      client.close();
    }
  });

  it("stops at max cycles when steps fail", async () => {
    const client = new MembrainClient({ storage: { path: makeTmpPath() } });
    try {
      const llm = async (messages) => {
        const system = messages[0]?.content ?? "";
        if (system.toLowerCase().includes("task planner")) {
          return planJson(["Will fail"]);
        }
        throw new Error("step blew up");
      };
      const agent = new MembrainAgent(client, llm, { maxCycles: 2 });
      const output = await agent.run("Impossible task");
      assert.equal(output, "");
    } finally {
      client.close();
    }
  });
});

describe("MembrainAgent live OpenAI", { skip: !loadApiKey() }, () => {
  it("produces a non-empty answer against gpt-4o-mini", async () => {
    const { default: OpenAI } = await import("openai").catch(() => ({
      default: null,
    }));
    if (!OpenAI) return;

    const client = new MembrainClient({ storage: { path: makeTmpPath() } });
    try {
      const openai = new OpenAI({ apiKey: loadApiKey() });
      const llm = async (messages) => {
        const response = await openai.chat.completions.create({
          model: "gpt-4o-mini",
          messages,
          temperature: 0,
        });
        return response.choices[0].message.content ?? "";
      };
      const agent = new MembrainAgent(client, llm, { maxCycles: 1 });
      const output = await agent.run(
        "Summarise the three steps to deploy a node web service",
      );
      assert.ok(output.length > 10);
    } finally {
      client.close();
    }
  });
});
