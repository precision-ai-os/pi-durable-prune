import type { Context } from "@earendil-works/chord/context";
import type { Message } from "@earendil-works/pi-ai";
import {
  type Cursor,
  defineTool,
  type ToolExecutionApi,
  type ToolRegistration,
} from "@earendil-works/pi-durable";
import { RECALL_TOOL } from "./note.js";
import { originalResult } from "./prune.js";

// The tool that reads a set-aside result again, from the conversation's transcript as written
// (edits do not apply there). A ref is only a locator: `visible` decides what the caller may read.

/** Narrows the transcript to what this call may read, such as one person's own turns. */
export type Visibility = (
  messages: Message[],
  call: { api: ToolExecutionApi; context: Context },
) => Message[] | Promise<Message[]>;

export interface RecallToolOptions {
  /** The tool's name. Default `recall_result`; a custom one should match the note's wording. */
  name?: string;
  /** The tool's description for the model. */
  description?: string;
  /** What the caller may read. Default: the whole transcript. */
  visible?: Visibility;
  /** The text returned when no readable result has the ref. */
  notFound?: (ref: string) => string;
  /** Passed through to the tool registration. Recalled results can be large. */
  outputLimits?: ToolRegistration["outputLimits"];
}

const DESCRIPTION = [
  "Read again, in full, an earlier tool result that was set aside to save room.",
  "Give the ref its note names. Use it when you need that result's exact words or figures; running the original tool again gives what is there now instead.",
].join(" ");

const NOT_FOUND = (ref: string) =>
  `No earlier result with ref "${ref}" can be read here. Run the original tool again.`;

const PAGE = 1_000;

/** A tool that reads a set-aside result back by its ref. Register it where `prune` runs. */
export function recallTool(options: RecallToolOptions = {}): ToolRegistration {
  const visible = options.visible ?? ((messages: Message[]) => messages);
  const notFound = options.notFound ?? NOT_FOUND;
  return defineTool({
    name: options.name ?? RECALL_TOOL,
    description: options.description ?? DESCRIPTION,
    parameters: {
      type: "object",
      properties: { ref: { type: "string", description: "The ref the note names." } },
      required: ["ref"],
      additionalProperties: false,
    },
    replay: "safe",
    callers: ["model"],
    ...(options.outputLimits ? { outputLimits: options.outputLimits } : {}),
    execute: async (args, api, context) => {
      const ref = String((args as { ref?: unknown }).ref ?? "").trim();
      const messages = await visible(await transcript(api, context), { api, context });
      const found = ref ? originalResult(messages, ref) : undefined;
      if (!found) return { output: [{ type: "text", text: notFound(ref) }] };
      return { output: found.content.filter((part) => part.type === "text") };
    },
  });
}

/** Every model message the conversation committed, oldest first, as written (no edits). */
export async function transcript(api: ToolExecutionApi, context: Context): Promise<Message[]> {
  const messages: Message[] = [];
  let cursor: Cursor | undefined;
  do {
    const page = await api.commit(
      (tx) =>
        tx.scanEntries({ conversationId: api.conversationId, order: "ascending" }, PAGE, cursor),
      context,
    );
    for (const entry of page.items) messages.push(...(entry.model ?? []));
    cursor = page.next;
  } while (cursor !== undefined);
  return messages;
}
