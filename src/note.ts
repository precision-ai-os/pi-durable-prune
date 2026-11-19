/** One result about to be set aside, as a note formatter sees it. */
export interface SetAside {
  /** The tool that produced the result. */
  toolName: string;
  /** The call's arguments as JSON, cut to 160 characters; empty when it had none. */
  args: string;
  /** The result's length in characters. */
  chars: number;
  /** The ref the recall tool reads it back by (the tool call id). */
  ref: string;
}

/** The default recall tool name, which the default note points to. */
export const RECALL_TOOL = "recall_result";

/**
 * The default note. It names the tool, its arguments, the size and the ref, and how to read the
 * result again. Keep a custom note deterministic: the same result must always give the same text.
 */
export function defaultNote(at: SetAside, recallTool: string = RECALL_TOOL): string {
  const asked = at.args ? ` for ${at.args}` : "";
  const size = at.chars.toLocaleString("en-US");
  return `[Set aside to save room: the ${at.toolName} result${asked}, ${size} characters. ${recallTool} with ref "${at.ref}" reads it again in full.]`;
}
