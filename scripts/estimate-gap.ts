// Reproduces the estimate gap: right after pruning, pi-durable still sizes the next request by
// the last answer's measured usage, which counted the results before they were set aside.
// Run: npx tsx scripts/estimate-gap.ts
import { BACKGROUND_CONTEXT as ctx } from "@earendil-works/chord/context";
import {
  createModels,
  fauxAssistantMessage,
  fauxProvider,
  fauxText,
  fauxToolCall,
} from "@earendil-works/pi-ai";
import {
  createRegistry,
  defineExtension,
  defineTool,
  Harness,
  MemoryStorage,
} from "@earendil-works/pi-durable";
import { prune } from "../src/index.js";

const read = defineTool({
  name: "read_big",
  description: "read",
  parameters: { type: "object", properties: {} },
  replay: "safe",
  execute: async () => ({ output: [{ type: "text", text: "lorem ipsum ".repeat(1500) }] }),
});
const faux = fauxProvider({ provider: "faux", models: [{ id: "m", contextWindow: 14_000 }] });
const models = createModels();
models.setProvider(faux.provider);
const registry = createRegistry();
registry.install(defineExtension({ name: "t", tools: [read] }));
const harness = await Harness.open(new MemoryStorage(), { models, registry }, ctx);
const root = await harness.root(ctx, {
  agent: {
    model: { provider: "faux", modelId: "m" },
    compaction: { reserveTokens: 4_000, keepRecentTokens: 500, backgroundTokens: 0 },
  } as never,
});
let summaries = 0;
// The faux model reports usage the way a real provider does: the request's size in tokens.
type Reply = ReturnType<typeof fauxAssistantMessage>;
const measured =
  (reply: Reply) =>
  (c: { messages: unknown[] }): Reply => {
    const input = Math.ceil(JSON.stringify(c.messages).length / 4);
    const cost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 };
    return { ...reply, usage: { input, output: 10, cacheRead: 0, cacheWrite: 0, totalTokens: input + 10, cost } };
  };
async function ask(text: string, reads: number) {
  faux.setResponses([
    ...Array.from({ length: reads }, () =>
      measured(fauxAssistantMessage(fauxToolCall("read_big", {}), { stopReason: "toolUse" })),
    ),
    measured(fauxAssistantMessage(fauxText(`answered ${text}`))),
    // Any further request in this ask is a compaction summary.
    () => {
      summaries += 1;
      return fauxAssistantMessage(fauxText("SUMMARY"));
    },
    measured(fauxAssistantMessage(fauxText(`answered ${text}`))),
  ]);
  await (await root.submit({ type: "input", content: text }, ctx)).wait(ctx);
}
await ask("first", 2);
await ask("second", 0);
const before = (await root.context(ctx)).messages.findLast((m) => m.role === "assistant") as {
  usage?: { totalTokens: number };
};
const setAside = await prune(harness, root, {});
const chars = JSON.stringify((await root.context(ctx)).messages).length;
await ask(`third ${"please ".repeat(700)}`, 0);
console.log(
  JSON.stringify({
    lastMeasuredTokens: before.usage?.totalTokens,
    setAside,
    contextCharsAfterPrune: chars,
    compactionsOnNextAsk: summaries,
  }),
);
await harness.close(ctx);
