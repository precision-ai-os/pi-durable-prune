import { hook, ToolTask } from "@earendil-works/pi-durable";

/** Options of `loadOnCall`. */
export interface LoadOnCallOptions {
  /** The deferred tools a call may load: their names, or a test on a name. */
  deferred: Iterable<string> | ((name: string) => boolean);
  /** The reply to the call that loaded a tool. Default `defaultLoadedMessage`. */
  message?: (tool: string) => string;
}

/** The default reply when a call loads its tool. */
export function defaultLoadedMessage(tool: string): string {
  return `${tool} wasn't loaded, so that call didn't run. It's loaded now: call it again with the same arguments.`;
}

/**
 * A hook for conversations that defer tools behind `tool_search`. When the model calls a deferred
 * tool that is not loaded (one it never searched for, or one whose search result was set aside, so
 * it no longer sees what is loaded), pi-durable answers `tool_unavailable` and the model may retry
 * the same call. With this hook the tool is loaded on that call and the model is told to call it
 * again. Calls of names that are not deferred tools get pi-durable's error as before.
 */
export function loadOnCall(options: LoadOnCallOptions) {
  const names = typeof options.deferred === "function" ? undefined : new Set(options.deferred);
  const isDeferred =
    typeof options.deferred === "function"
      ? options.deferred
      : (name: string) => names?.has(name) ?? false;
  const message = options.message ?? defaultLoadedMessage;
  return hook(ToolTask, {
    afterTool: async (call, result) => {
      const unavailable = result.diagnostics?.some((one) => one.code === "tool_unavailable");
      if (!unavailable || !isDeferred(call.name)) return undefined;
      return {
        output: [{ type: "text", text: message(call.name) }],
        isError: true,
        control: { addTools: [call.name] },
      };
    },
  });
}
