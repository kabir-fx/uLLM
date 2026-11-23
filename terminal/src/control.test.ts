import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import type { LoadSpec, LoadStatus } from "./contracts/control";
import { controlClient } from "./control";

let server: Server | undefined;
afterEach(() => new Promise<void>((done) => (server ? server.close(() => done()) : done())));

interface Seen {
  method: string;
  url: string;
  type: string | undefined;
  body: string;
}

function serve(reply: (seen: Seen, res: ServerResponse) => void): Promise<{ base: string; seen: Seen[] }> {
  const seen: Seen[] = [];
  server = createServer((req: IncomingMessage, res) => {
    let body = "";
    req.on("data", (c: Buffer) => (body += c.toString()));
    req.on("end", () => {
      const s = { method: req.method!, url: req.url!, type: req.headers["content-type"], body };
      seen.push(s);
      reply(s, res);
    });
  });
  return new Promise((resolve) =>
    server!.listen(0, "127.0.0.1", () => resolve({ base: `http://127.0.0.1:${(server!.address() as AddressInfo).port}`, seen })),
  );
}

const json = (res: ServerResponse, status: number, body: unknown) => {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
};

const SPEC: LoadSpec = { workload: "a", arrival: { kind: "open", rate_per_s: 4 }, max_tokens: 64, requests: 32 };
const RUNNING: LoadStatus = { state: "running", spec: SPEC, ok: 0, errors: 0, first_error: null };

describe("controlClient", () => {
  it("starts a run with the spec as a JSON body and returns the status", async () => {
    const { base, seen } = await serve((_, res) => json(res, 202, RUNNING));
    expect(await controlClient(base).start(SPEC)).toEqual(RUNNING);
    expect(seen).toEqual([{ method: "POST", url: "/load", type: "application/json", body: JSON.stringify(SPEC) }]);
  });

  it("reads the workloads and the status, and stops with DELETE", async () => {
    const { base, seen } = await serve((s, res) =>
      s.url === "/workloads" ? json(res, 200, [{ name: "a", requests: 256 }]) : json(res, 200, RUNNING),
    );
    const control = controlClient(`${base}/`);
    expect(await control.workloads()).toEqual([{ name: "a", requests: 256 }]);
    expect(await control.status()).toEqual(RUNNING);
    await control.stop();
    expect(seen.map((s) => `${s.method} ${s.url}`)).toEqual(["GET /workloads", "GET /load", "DELETE /load"]);
  });

  it("throws the server's detail on a refusal", async () => {
    const { base } = await serve((_, res) => json(res, 409, { detail: "a run is running; stop it first" }));
    await expect(controlClient(base).start(SPEC)).rejects.toThrow("409: a run is running; stop it first");
  });

  it("names the control process when it cannot reach it", async () => {
    const { base } = await serve(() => {});
    server!.close();
    server!.closeAllConnections();
    server = undefined;
    await expect(controlClient(base).status()).rejects.toThrow(`cannot reach the control process at ${base}`);
  });
});
