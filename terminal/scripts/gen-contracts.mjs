// Generate TypeScript types from the JSON Schemas that the Python models export:
// telemetry protocol v1 (C2, decision 0003) and load control (decision 0024).
// Never edit the output by hand.
//
//   node scripts/gen-contracts.mjs          write src/contracts/*.ts
//   node scripts/gen-contracts.mjs --check  exit 1 if one is stale
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { compile } from "json-schema-to-typescript";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const contracts = [
  {
    schema: "../../src/ullm/telemetry/schema/telemetry.v1.json",
    root: "Frame",
    out: "../src/contracts/telemetry.ts",
    source: "src/ullm/telemetry/schema/telemetry.v1.json",
    models: "src/ullm/telemetry/events.py",
  },
  {
    schema: "../../bench/control.schema.json",
    root: "Control",
    out: "../src/contracts/control.ts",
    source: "bench/control.schema.json",
    models: "bench/control.py",
  },
];

let stale = false;
for (const c of contracts) {
  const banner = `// Generated from ${c.source} by \`make contracts\`.
// Do not edit: change ${c.models} and regenerate.`;
  const schema = JSON.parse(await readFile(here(c.schema), "utf8"));
  const fresh = await compile(schema, c.root, {
    bannerComment: banner,
    additionalProperties: false,
    ignoreMinAndMaxItems: true,
    strictIndexSignatures: true,
    unknownAny: true,
    unreachableDefinitions: true,
    style: { printWidth: 100, singleQuote: false, semi: true, trailingComma: "all" },
  });
  const outPath = here(c.out);
  if (process.argv.includes("--check")) {
    const current = await readFile(outPath, "utf8").catch(() => "");
    if (current !== fresh) {
      console.error(`${outPath} is stale: run \`make contracts\``);
      stale = true;
    }
  } else {
    await writeFile(outPath, fresh);
    console.log(`wrote ${outPath}`);
  }
}
if (stale) process.exit(1);
