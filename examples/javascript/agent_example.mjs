/** Plan-execute-learn agent driven by an OpenAI LLM.
 *
 * Requires: OPENAI_API_KEY environment variable.
 */

import { MembrainAgent, MembrainClient } from "membrain";

if (!process.env.OPENAI_API_KEY) {
  console.log("Set OPENAI_API_KEY to run this example.");
  process.exit(0);
}

const OpenAI = (await import("openai")).default;
const openai = new OpenAI();

async function llm(messages) {
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages,
    temperature: 0,
  });
  return response.choices[0].message.content ?? "";
}

const client = new MembrainClient();
try {
  client.storeCase(
    "Deploy a Node web service",
    JSON.stringify({
      plan: [
        { id: 1, description: "Build the bundle" },
        { id: 2, description: "Run tests" },
        { id: 3, description: "Promote to production" },
      ],
    }),
    "Deployment completed without downtime",
    1.0,
  );

  const agent = new MembrainAgent(client, llm, { maxCycles: 1 });
  agent.executor.registerTool(
    "lookup_health",
    (service) => `${service}: healthy`,
  );

  const output = await agent.run(
    "Deploy the billing microservice to production",
  );
  console.log("--- Agent output ---");
  console.log(output);
} finally {
  client.close();
}
