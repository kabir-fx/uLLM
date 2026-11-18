import type { Frame } from "./contracts/telemetry";

const GZIP_MAGIC = [0x1f, 0x8b] as const;

/** One frame from one SSE `data:` line or one trace line. Rejects other protocol versions (C2). */
export function parseFrame(text: string): Frame {
  const value: unknown = JSON.parse(text);
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("not a telemetry frame");
  }
  const { v } = value as { v?: unknown };
  if (v !== 1) throw new Error(`telemetry version ${String(v)}; this terminal reads version 1`);
  return value as Frame;
}

/**
 * A recorded trace: gzip JSONL, one frame per line (C2). A server may already have
 * decoded the gzip for the browser (Content-Encoding), so plain JSONL is read too.
 */
export async function parseTrace(bytes: Uint8Array<ArrayBuffer>): Promise<Frame[]> {
  const gzipped = bytes[0] === GZIP_MAGIC[0] && bytes[1] === GZIP_MAGIC[1];
  const text = gzipped ? await gunzip(bytes) : new TextDecoder().decode(bytes);
  const frames: Frame[] = [];
  text.split("\n").forEach((line, i) => {
    if (line.trim() === "") return;
    try {
      frames.push(parseFrame(line));
    } catch (error) {
      throw new Error(`trace line ${i + 1}: ${(error as Error).message}`, { cause: error });
    }
  });
  return frames;
}

export async function loadTrace(url: string): Promise<Frame[]> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return parseTrace(new Uint8Array(await response.arrayBuffer()));
}

async function gunzip(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).text();
}
