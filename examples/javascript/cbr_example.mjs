/** Case-based reasoning: retrieve past cases, build a prompt, record training data.
 *
 * Requires: OPENAI_API_KEY environment variable.
 */

import {
  CasePromptBuilder,
  MembrainClient,
  NonParametricRetriever,
  TrainingDataCollector,
} from "membrain";

if (!process.env.OPENAI_API_KEY) {
  console.log("Set OPENAI_API_KEY to run this example.");
  process.exit(0);
}

const OpenAI = (await import("openai")).default;
const openai = new OpenAI();

const client = new MembrainClient();
try {
  client.storeCase(
    "Scale the API to handle 10k rps",
    "Add a connection pool and horizontal autoscaling",
    "Latency stayed under 200ms at peak load",
    1.0,
  );
  client.storeCase(
    "Scale the API to handle 10k rps",
    "Rewrite the hot path in Rust only",
    "Weeks of work, no measurable gain under load",
    0.0,
  );

  const retriever = new NonParametricRetriever(client);
  const promptBuilder = new CasePromptBuilder({
    maxPositiveExamples: 2,
    maxNegativeExamples: 1,
  });

  const query = "Plan capacity for the checkout service going into Black Friday";
  const cases = retriever.retrieve(query, 5);
  const context = promptBuilder.buildContext(cases);

  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    temperature: 0,
    messages: [
      {
        role: "system",
        content: "You are an SRE assistant. Reply in three steps.",
      },
      {
        role: "user",
        content: context
          ? `${context}\n\n## Current Query\n\n${query}`
          : query,
      },
    ],
  });
  const answer = response.choices[0].message.content ?? "";
  console.log("--- Assistant answer ---");
  console.log(answer);

  const collector = new TrainingDataCollector();
  collector.record(
    query,
    [...cases.positive_cases, ...cases.negative_cases],
    true,
  );
  const written = await collector.flush("./training_data.jsonl");
  console.log(`\nWrote ${written} training pairs to ./training_data.jsonl`);
} finally {
  client.close();
}
