import type { TelemetrySource } from "../source";
import { parseFrame } from "../trace";

/**
 * Reads `/telemetry/stream` (C2) with `fetch` rather than `EventSource`, so the same code
 * runs under Node in tests. It does not reconnect: a new connection starts with a new hello.
 */
export function liveSource(url: string): TelemetrySource {
  return {
    label: url,
    start(sink) {
      const abort = new AbortController();
      sink.status({ kind: "connecting" });
      void (async () => {
        try {
          const response = await fetch(url, { signal: abort.signal, headers: { accept: "text/event-stream" } });
          if (!response.ok || response.body === null) throw new Error(`${url}: HTTP ${response.status}`);
          sink.status({ kind: "open" });
          const parser = new SseParser();
          const text = response.body.pipeThrough(new TextDecoderStream());
          for await (const chunk of text) {
            for (const data of parser.push(chunk)) sink.frame(parseFrame(data));
          }
          sink.status({ kind: "ended" });
        } catch (error) {
          if (!abort.signal.aborted) sink.status({ kind: "error", message: (error as Error).message });
        } finally {
          abort.abort();
        }
      })();
      return () => abort.abort();
    },
  };
}

/** The `data` of each complete server-sent event, per the WHATWG event stream format. */
export class SseParser {
  private buffer = "";
  private data: string[] = [];

  push(chunk: string): string[] {
    this.buffer += chunk;
    // A CR at the end may be the first half of a CR LF pair; wait for the next chunk.
    const complete = this.buffer.endsWith("\r") ? this.buffer.slice(0, -1) : this.buffer;
    const lines = complete.split(/\r\n|\r|\n/);
    this.buffer = (lines.pop() ?? "") + this.buffer.slice(complete.length);
    const events: string[] = [];
    for (const line of lines) {
      if (line === "") {
        if (this.data.length > 0) events.push(this.data.join("\n"));
        this.data = [];
      } else if (line.startsWith("data:")) {
        this.data.push(line.slice(line.startsWith("data: ") ? 6 : 5));
      }
    }
    return events;
  }
}
