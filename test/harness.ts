import { BACKGROUND_CONTEXT as ctx } from "@earendil-works/chord/context";
import {
  createModels,
  fauxAssistantMessage,
  fauxProvider,
  fauxText,
  fauxToolCall,
  type Message,
} from "@earendil-works/pi-ai";
import {
  type Conversation,
  createRegistry,
  defineExtension,
  defineTool,
  Harness,
  MemoryStorage,
  type ToolRegistration,
} from "@earendil-works/pi-durable";

// A real pi-durable harness on in-memory storage with a scripted faux model. Every request the
// model receives is kept as its messages serialized one per string, so tests can compare bytes.

export const big = (tag: string) => `${tag}:${"x".repeat(3000)}`;

const params = { type: "object", properties: { q: { type: "string" } } } as const;

function tool(name: string, run: (args: { q?: string }) => unknown): ToolRegistration {
  return defineTool({
    name,
    description: name,
    parameters: params,
    replay: "safe",
    execute: async (args) => run(args as { q?: string }) as never,
  });
}

export const testTools = [
  tool("read_big", (a) => ({ output: [{ type: "text", text: big(`read ${a.q}`) }] })),
  tool("read_small", () => ({ output: [{ type: "text", text: "small" }] })),
  tool("use_skill", () => ({ output: [{ type: "text", text: big("skill") }] })),
  tool("picture", () => ({
    output: [
      { type: "text", text: big("picture") },
      { type: "image", data: "iVBORw0KGgo=", mimeType: "image/png" },
    ],
  })),
  tool("fail", () => {
    throw new Error(big("boom"));
  }),
];

export type Call = [name: string, args: Record<string, unknown>];

export async function openHarness(options: { extraTools?: ToolRegistration[]; agent?: object } = {}) {
  const faux = fauxProvider({ provider: "faux", models: [{ id: "m", contextWindow: 1_000_000 }] });
  const models = createModels();
  models.setProvider(faux.provider);
  const registry = createRegistry();
  registry.install(
    defineExtension({ name: "test", tools: [...testTools, ...(options.extraTools ?? [])] }),
  );
  const harness = await Harness.open(new MemoryStorage(), { models, registry }, ctx);
  const root: Conversation = await harness.root(ctx, {
    agent: { model: { provider: "faux", modelId: "m" }, maxTurns: 40, ...options.agent } as never,
  });
  const requests: string[][] = [];
  const record = (message: ReturnType<typeof fauxAssistantMessage>) => (c: { messages: Message[] }) => {
    requests.push(c.messages.map((m) => JSON.stringify(m)));
    return message;
  };

  /** One ask: the model calls `calls` in turn, then answers. Returns its first request's index. */
  async function ask(text: string, calls: Call[] = [], user: unknown = text) {
    faux.setResponses([
      ...calls.map(([name, args]) =>
        record(fauxAssistantMessage(fauxToolCall(name, args as never), { stopReason: "toolUse" })),
      ),
      record(fauxAssistantMessage(fauxText(`answered ${text}`))),
    ]);
    const at = requests.length;
    const submission = await root.submit({ type: "input", content: user as never }, ctx);
    const settled = await submission.wait(ctx);
    return { at, status: settled.status };
  }

  /** Every model message the conversation committed, as written. */
  async function raw(): Promise<Message[]> {
    const out: Message[] = [];
    let cursor: unknown;
    do {
      const page = await root.entries({ order: "ascending" }, 50, cursor as never, ctx);
      for (const entry of page.items) out.push(...((entry.model ?? []) as Message[]));
      cursor = page.next;
    } while (cursor);
    return out;
  }

  return { faux, harness, root, requests, record, ask, raw, ctx };
}

export { ctx, fauxAssistantMessage, fauxText, fauxToolCall };
