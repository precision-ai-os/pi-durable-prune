import type { Message } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import {
  defaultNote,
  loadOnCall,
  originalResult,
  PRUNE_ENTRY_KIND,
  type PrunePolicy,
  prune,
  recallTool,
  withOriginals,
} from "../src/index.js";
import { ctx, fauxAssistantMessage, fauxText, openHarness } from "./harness.js";
import { defineTool } from "@earendil-works/pi-durable";

const POLICY: PrunePolicy = { keep: ["use_skill"] };
const isNote = (m: string) => m.includes("Set aside to save room");
const toolResults = (request: string[]) =>
  request
    .filter((m) => m.includes('"role":"toolResult"'))
    .map((m) => `${JSON.parse(m).toolName}:${isNote(m) ? "note" : "whole"}`);

let close: (() => Promise<void>) | undefined;
afterEach(async () => {
  await close?.();
  close = undefined;
});

async function setup(options?: Parameters<typeof openHarness>[0]) {
  const h = await openHarness(options);
  close = () => h.harness.close(ctx);
  return h;
}

describe("prune", () => {
  it("sets aside only large text results of finished earlier asks, keeping the prompt cache prefix", async () => {
    const h = await setup();
    const pruned: number[] = [];
    const ask = async (text: string, calls: Parameters<typeof h.ask>[1]) => {
      pruned.push(await prune(h.harness, h.root, POLICY));
      return h.ask(text, calls);
    };
    await ask("ask1", [
      ["read_big", { q: "one" }],
      ["read_small", {}],
      ["use_skill", {}],
      ["picture", {}],
      ["fail", {}],
    ]);
    await ask("ask2", [["read_big", { q: "two" }]]);
    const a3 = await ask("ask3", [["read_big", { q: "three" }]]);
    const a4 = await ask("ask4", []);
    expect(a4.status).toBe("done");

    // Ask 2 runs on a fresh conversation's latest ask: nothing to set aside yet. Ask 3 sets aside
    // ask 1's large read; ask 4 sets aside ask 2's. Small, kept, picture and error results stay.
    expect(pruned).toEqual([0, 0, 1, 1]);
    const r3 = h.requests[a3.at] ?? [];
    const r4 = h.requests[a4.at] ?? [];
    expect(toolResults(r4)).toEqual([
      "read_big:note",
      "read_small:whole",
      "use_skill:whole",
      "picture:whole",
      "fail:whole",
      "read_big:note",
      "read_big:whole",
    ]);

    // Ask 4's first request matches ask 3's byte for byte up to the newly set-aside result, and
    // the note written before ask 3 is unchanged.
    let same = 0;
    while (same < r3.length && r3[same] === r4[same]) same += 1;
    const newNote = r4.findIndex((m) => isNote(m) && m.includes('\\"q\\":\\"two'));
    expect(newNote).toBeGreaterThan(0);
    expect(same).toBe(newNote);
    const firstNote = (r: string[]) => r.find((m) => isNote(m) && m.includes('\\"q\\":\\"one'));
    expect(firstNote(r3)).toBeDefined();
    expect(firstNote(r3)).toBe(firstNote(r4));

    // Idempotent: ask 3's read goes now that ask 4 finished, then nothing is left.
    expect(await prune(h.harness, h.root, POLICY)).toBe(1);
    expect(await prune(h.harness, h.root, POLICY)).toBe(0);
    const view = await h.root.context(ctx);
    expect(view.entries.filter((e) => e.kind === PRUNE_ENTRY_KIND)).toHaveLength(3);

    // The transcript keeps every original.
    const raw = await h.raw();
    expect(raw.filter((m) => JSON.stringify(m).includes("Set aside"))).toHaveLength(0);
  });

  it("follows the policy's size, kept asks and note", async () => {
    const h = await setup();
    await h.ask("a", [["read_big", { q: "a" }]]);
    await h.ask("b", [["read_big", { q: "b" }]]);
    await h.ask("c", [["read_big", { q: "c" }]]);
    expect(await prune(h.harness, h.root, { minChars: 10_000 })).toBe(0);
    expect(await prune(h.harness, h.root, { keepAsks: 2, note: (r) => `gone ${r.ref}` })).toBe(1);
    const messages = (await h.root.context(ctx)).messages;
    const notes = messages.filter((m) => JSON.stringify(m).includes('"gone '));
    expect(notes).toHaveLength(1);
  });

  it("skips a busy conversation at once and a run that ended in an error", async () => {
    const h = await setup();
    await h.ask("a", [["read_big", { q: "a" }]]);
    await h.ask("b", []);

    let release: (m: unknown) => void = () => {};
    h.faux.setResponses([() => new Promise((r) => (release = r)) as never]);
    const going = await h.root.submit({ type: "input", content: "slow" }, ctx);
    await new Promise((r) => setTimeout(r, 100));
    const started = Date.now();
    expect(await prune(h.harness, h.root, POLICY)).toBe(0);
    expect(Date.now() - started).toBeLessThan(1_000);
    release(fauxAssistantMessage(fauxText("slow done")));
    expect((await going.wait(ctx)).status).toBe("done");

    h.faux.setResponses([fauxAssistantMessage([], { stopReason: "error", errorMessage: "boom" })]);
    const failed = await h.root.submit({ type: "input", content: "failing" }, ctx);
    await failed.wait(ctx);
    expect(await prune(h.harness, h.root, POLICY)).toBe(0);
  });

  it("gives a fork the originals back and survives compaction", async () => {
    const h = await setup({ agent: { compaction: { keepRecentTokens: 50 } } });
    await h.ask("a", [["read_big", { q: "a" }]]);
    await h.ask("b", [["read_big", { q: "b" }]]);
    await h.ask("c", []);
    expect(await prune(h.harness, h.root, POLICY)).toBe(2);
    const view = await h.root.context(ctx);
    const notes = (messages: readonly Message[]) =>
      messages.filter((m) => JSON.stringify(m).includes("Set aside")).length;
    expect(notes(view.messages)).toBe(2);
    expect(notes(withOriginals(view))).toBe(0);
    expect(withOriginals(view)).toHaveLength(view.messages.length);

    // Compaction summarizes what the model reads (the notes), and later asks carry no notes.
    h.faux.setResponses([h.record(fauxAssistantMessage(fauxText("SUMMARY")))]);
    await h.root.compact(undefined, ctx);
    for (let i = 0; i < 50 && !(await h.root.context(ctx)).head; i++)
      await new Promise((r) => setTimeout(r, 50));
    expect((await h.root.context(ctx)).head).toBeDefined();
    expect((h.requests.at(-1) ?? []).join("")).toContain("Set aside");
    expect(await prune(h.harness, h.root, POLICY)).toBe(0);
    const after = await h.ask("d", []);
    expect(after.status).toBe("done");
    expect((h.requests[after.at] ?? []).filter(isNote)).toHaveLength(0);
    expect(originalResult(await h.raw(), refOf(await h.raw(), "read a"))).toBeDefined();
  });
});

function refOf(messages: Message[], text: string): string {
  const found = messages.find((m) => m.role === "toolResult" && JSON.stringify(m).includes(text));
  return (found as { toolCallId: string }).toolCallId;
}

describe("recallTool", () => {
  it("reads a set-aside result back by ref, within what the caller may see", async () => {
    let hidden = false;
    const seen: unknown[] = [];
    const recall = recallTool({
      visible: (messages, { api }) => {
        seen.push(api.conversationId);
        return hidden
          ? messages.filter((m) => !(m.role === "toolResult" && m.toolName === "read_big"))
          : messages;
      },
    });
    const h = await setup({ extraTools: [recall] });
    await h.ask("a", [["read_big", { q: "a" }]]);
    await h.ask("b", []);
    expect(await prune(h.harness, h.root, POLICY)).toBe(1);
    const ref = refOf(await h.raw(), "read a");
    const note = (await h.root.context(ctx)).messages.find((m) =>
      JSON.stringify(m).includes("Set aside"),
    );
    expect(JSON.stringify(note)).toContain(ref);
    expect(JSON.stringify(note)).toContain(
      JSON.stringify(defaultNote({ toolName: "read_big", args: '{"q":"a"}', chars: 3007, ref })).slice(1, -1),
    );

    const recalled = async () => {
      await h.ask("again", [["recall_result", { ref }]]);
      const results = (await h.raw()).filter(
        (m) => m.role === "toolResult" && m.toolName === "recall_result",
      );
      return JSON.stringify(results.at(-1));
    };
    expect(await recalled()).toContain("read a:xxxx");
    hidden = true;
    expect(await recalled()).not.toContain("read a:xxxx");
    expect(seen).toEqual([h.root.id, h.root.id]);
  });
});


describe("loadOnCall", () => {
  it("loads a deferred tool the model calls without searching, and the retry runs it", async () => {
    const vendors = defineTool({
      name: "list_vendors",
      description: "List vendors",
      parameters: { type: "object", properties: {} },
      deferred: true,
      replay: "safe",
      execute: async () => ({ output: [{ type: "text", text: "Acme, Summit" }] }),
    });
    const h = await setup({ extraTools: [vendors], hooks: [loadOnCall({ deferred: ["list_vendors"] })] });
    await h.ask("vendors?", [
      ["list_vendors", {}],
      ["list_vendors", {}],
    ]);
    const results = (await h.raw()).filter((m) => m.role === "toolResult");
    expect(results.map((m) => m.isError)).toEqual([true, false]);
    expect(JSON.stringify(results[1]?.content)).toContain("Acme, Summit");
    // A name that is not a deferred tool keeps pi-durable's own error.
    await h.ask("again", [["no_such_tool", {}]]);
    const last = (await h.raw()).filter((m) => m.role === "toolResult").at(-1);
    expect(last?.isError).toBe(true);
    expect(JSON.stringify(last?.content)).not.toContain("loaded now");
  });
});
