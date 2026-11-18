import { readFileSync } from "node:fs";
import type { Frame } from "../contracts/telemetry";
import { parseTrace } from "../trace";

export function fixtureBytes(name: string): Uint8Array<ArrayBuffer> {
  return new Uint8Array(readFileSync(new URL(`../../fixtures/telemetry/${name}`, import.meta.url)));
}

export function fixtureFrames(name: string): Promise<Frame[]> {
  return parseTrace(fixtureBytes(name));
}
