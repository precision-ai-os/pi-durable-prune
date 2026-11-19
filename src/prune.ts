import { BACKGROUND_CONTEXT, type Context } from "@earendil-works/chord/context";
import type { Message, ToolResultMessage } from "@earendil-works/pi-ai";
import {
  type ContextEdit,
  type ContextView,
  type Conversation,
  type EntryRecord,
  type Harness,
  ToolResultEntry,
  UserEntry,
} from "@earendil-works/pi-durable";
import { defaultNote, type SetAside } from "./note.js";

// Large tool results of earlier asks, set aside from what the model reads. Once a conversation
// rests at a finished answer, one entry's context edits replace each such result with a short
// note naming a ref. The results themselves stay in the transcript unchanged, so recall and audits
// read them there. A note never changes once written, so the request stays byte-identical up to
// the earliest result newly set aside and the provider's prompt cache keeps that prefix.

/** What `prune` sets aside. Every field has a default. */
export interface PrunePolicy {
  /** Results shorter than this many characters stay whole. Default 2,000. */
  minChars?: number;
  /** The latest asks (user inputs) whose results stay whole. Default 1. */
  keepAsks?: number;
  /** Tool names whose results always stay whole, such as instructions the model keeps following. */
  keep?: Iterable<string>;
  /** The text a result is replaced with. Default `defaultNote`. */
  note?: (result: SetAside) => string;
}

/** The kind of the entry that carries the edits. */
export const PRUNE_ENTRY_KIND = "context.prune";

const ARGS_CHARS = 160;

interface Resolved {
  minChars: number;
  keepAsks: number;
  keep: ReadonlySet<string>;
  note: (result: SetAside) => string;
}

function resolve(policy: PrunePolicy = {}): Resolved {
  return {
    minChars: policy.minChars ?? 2_000,
    keepAsks: policy.keepAsks ?? 1,
    keep: new Set(policy.keep ?? []),
    note: policy.note ?? defaultNote,
  };
}

/**
 * The edits that set aside what `policy` lets go among `entries` (a conversation's context view,
 * oldest first), skipping results an earlier edit already covers. Pure: writes nothing.
 */
export function pruneEdits(entries: readonly EntryRecord[], policy?: PrunePolicy): ContextEdit[] {
  const resolved = resolve(policy);
  const edited = new Set(entries.flatMap((entry) => (entry.edits ?? []).map((one) => one.target)));
  const calls = new Map<string, unknown>();
  const edits: ContextEdit[] = [];
  for (const entry of entries.slice(0, firstKept(entries, resolved.keepAsks))) {
    for (const message of entry.model ?? []) {
      if (message.role !== "assistant") continue;
      for (const part of message.content)
        if (part.type === "toolCall") calls.set(part.id, part.arguments);
    }
    if (edited.has(entry.id)) continue;
    const edit = setAside(entry, calls, resolved);
    if (edit) edits.push(edit);
  }
  return edits;
}

/** Index of the first entry that stays whole: the start of the latest `keepAsks` asks. */
function firstKept(entries: readonly EntryRecord[], keepAsks: number): number {
  if (keepAsks <= 0) return entries.length;
  const asks = entries.flatMap((entry, index) => (UserEntry.is(entry) ? [index] : []));
  return asks.length > keepAsks ? (asks[asks.length - keepAsks] as number) : 0;
}

/** The edit replacing one tool result entry with its note, when the policy lets it go. */
function setAside(
  entry: EntryRecord,
  calls: ReadonlyMap<string, unknown>,
  policy: Resolved,
): ContextEdit | undefined {
  if (!ToolResultEntry.is(entry) || entry.model?.length !== 1) return undefined;
  const result = entry.model[0] as Message;
  if (result.role !== "toolResult" || result.isError || policy.keep.has(result.toolName)) return;
  if (!result.content.every((part) => part.type === "text")) return undefined;
  const chars = result.content.reduce(
    (n, part) => n + (part.type === "text" ? part.text.length : 0),
    0,
  );
  if (chars < policy.minChars) return undefined;
  const text = policy.note({
    toolName: result.toolName,
    args: argsLine(calls.get(result.toolCallId)),
    chars,
    ref: result.toolCallId,
  });
  const { details: _details, ...kept } = result;
  return {
    target: entry.id,
    action: "replace",
    messages: [{ ...kept, content: [{ type: "text", text }] }],
  };
}

/**
 * Sets aside what `policy` lets go, but only on a conversation that is idle and resting at a
 * finished answer: no live task or queued submission of its own, and its newest message an
 * assistant answer that stopped on its own (not an error, an abort or a tool call). Anything else
 * returns 0 at once and is left for the next time, so the write never waits behind a run. Call it
 * before submitting a new input. The write is keyed by the conversation's newest entry, so a retry
 * after a crash finds the first write. Returns how many results were set aside.
 */
export async function prune(
  harness: Harness,
  conversation: Conversation,
  policy?: PrunePolicy,
  context: Context = BACKGROUND_CONTEXT,
): Promise<number> {
  if (await busy(harness, conversation, context)) return 0;
  const view = await conversation.context(context);
  if (!restsAtAnswer(view.entries)) return 0;
  const edits = pruneEdits(view.entries, policy);
  const tail = view.entries.at(-1);
  if (edits.length === 0 || !tail) return 0;
  await conversation.submit(
    {
      type: "write",
      requestId: `prune:${tail.id}`,
      entry: { kind: PRUNE_ENTRY_KIND, edits, data: { results: edits.length } },
    },
    context,
  );
  return edits.length;
}

/** Whether the conversation has a live task or a queued or placed submission. */
async function busy(harness: Harness, conversation: Conversation, context: Context) {
  const live = await harness.inspect(context);
  return (
    live.tasks.some((task) => task.record.conversationId === conversation.id) ||
    live.submissions.some((one) => one.conversationId === conversation.id)
  );
}

/** Whether the newest message is an answer the model finished: no tool calls, no error. */
function restsAtAnswer(entries: readonly EntryRecord[]): boolean {
  const said = entries.filter((entry) => entry.model?.some((m) => m.role !== "system"));
  const last = said.at(-1)?.model?.at(-1);
  if (last?.role !== "assistant") return false;
  if (last.stopReason !== "stop" && last.stopReason !== "length") return false;
  return !last.content.some((part) => part.type === "toolCall");
}

/**
 * A context view's messages with every set-aside result whole again. Use it to seed a forked or
 * child conversation, which has no recall of its parent's refs.
 */
export function withOriginals(view: ContextView): Message[] {
  const originals = new Map<string, ToolResultMessage>();
  for (const entry of view.entries) {
    if (!ToolResultEntry.is(entry)) continue;
    for (const message of entry.model ?? [])
      if (message.role === "toolResult") originals.set(message.toolCallId, message);
  }
  return view.messages.map((message) =>
    message.role === "toolResult" ? (originals.get(message.toolCallId) ?? message) : message,
  );
}

/** The tool result `ref` names among `messages`, if any. */
export function originalResult(
  messages: readonly Message[],
  ref: string,
): ToolResultMessage | undefined {
  return messages.find(
    (message): message is ToolResultMessage =>
      message.role === "toolResult" && message.toolCallId === ref,
  );
}

function argsLine(args: unknown): string {
  if (args === undefined || args === null) return "";
  const text = JSON.stringify(args);
  if (text === "{}") return "";
  return text.length > ARGS_CHARS ? `${text.slice(0, ARGS_CHARS)}…` : text;
}
