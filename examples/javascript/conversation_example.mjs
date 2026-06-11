/** Automatic conversation management with memory extraction and retrieval.
 *
 * Requires: OPENAI_API_KEY environment variable.
 */

import { Conversation } from "membrain";

if (!process.env.OPENAI_API_KEY) {
  console.log("Set OPENAI_API_KEY to run this example.");
  process.exit(0);
}

const OpenAI = (await import("openai")).default;
const openai = new OpenAI();

async function llm(messages) {
  const response = await openai.chat.completions.create({ model: "gpt-4o-mini", messages });
  return response.choices[0].message.content;
}

const conv = new Conversation(llm);

try {
  console.log(`Session: ${conv.sessionId}\n`);

  const r1 = await conv.reply(
    "I prefer dark mode for all my tools, and I work at Acme Corp as a backend engineer.",
  );
  console.log(`Assistant: ${r1}\n`);

  const r2 = await conv.reply("Can you recommend a code editor setup for me?");
  console.log(`Assistant: ${r2}\n`);

  const r3 = await conv.reply("What do you remember about me?");
  console.log(`Assistant: ${r3}\n`);

  conv.end("User was satisfied with recommendations", 1.0);
  console.log(`Conversation ended. ${conv.history.length} messages tracked.`);
} finally {
  conv.close();
}
