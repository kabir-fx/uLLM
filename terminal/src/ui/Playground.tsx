import { useEffect, useRef, useState } from "react";
import { servedModel, streamChat } from "../playground";

export function Playground({ api }: { api: string }) {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(256);
  const [reply, setReply] = useState("");
  const [line, setLine] = useState("");
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const model = useRef<Promise<string> | null>(null);
  useEffect(() => () => abort.current?.abort(), []);

  async function send() {
    const controller = new AbortController();
    abort.current = controller;
    setBusy(true);
    setReply("");
    setLine("streaming…");
    try {
      model.current ??= servedModel(api);
      for await (const piece of streamChat(api, await model.current, prompt, maxTokens, controller.signal)) {
        if (piece.kind === "text") setReply((r) => r + piece.text);
        else setLine(`${piece.finishReason}: ${piece.promptTokens} prompt, ${piece.completionTokens} completion tokens`);
      }
    } catch (error) {
      model.current = null;
      setLine(controller.signal.aborted ? "stopped" : (error as Error).message);
    } finally {
      abort.current = null;
      setBusy(false);
    }
  }

  return (
    <section className="panel side-panel playground" aria-label="playground">
      <h2>
        playground <span className="dim">greedy, one request</span>
      </h2>
      <form
        className="controls"
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <textarea name="prompt" rows={3} value={prompt} placeholder="your prompt" onChange={(e) => setPrompt(e.target.value)} />
        <label>
          max_tokens
          <input type="number" name="maxTokens" min={1} max={4096} value={maxTokens} onChange={(e) => setMaxTokens(e.target.valueAsNumber)} />
        </label>
        <div className="buttons">
          <button type="submit" disabled={busy || prompt.trim() === ""}>
            send
          </button>
          <button type="button" disabled={!busy} onClick={() => abort.current?.abort()}>
            stop
          </button>
        </div>
      </form>
      <pre className="reply">{reply}</pre>
      <p className="line dim">{line}</p>
    </section>
  );
}
