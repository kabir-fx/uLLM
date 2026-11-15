// Generate TypeScript types for telemetry protocol v1 from the JSON Schema that
// the Python models export (C2, decision 0003). Never edit the output by hand.
//
//   node scripts/gen-contracts.mjs          write src/contracts/telemetry.ts
//   node scripts/gen-contracts.mjs --check  exit 1 if it is stale
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { compile } from "json-schema-to-typescript";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const schemaPath = here("../../src/ullm/telemetry/schema/telemetry.v1.json");
const outPath = here("../src/contracts/telemetry.ts");

const banner = `/* eslint-disable */
// Generated from src/ullm/telemetry/schema/telemetry.v1.json by \`make contracts\`.
// Do not edit: change src/ullm/telemetry/events.py and regenerate.`;

const schema = JSON.parse(await readFile(schemaPath, "utf8"));
const fresh = await compile(schema, "Frame", {
  bannerComment: banner,
  additionalProperties: false,
  ignoreMinAndMaxItems: true,
  strictIndexSignatures: true,
  unknownAny: true,
  unreachableDefinitions: true,
  style: { printWidth: 100, singleQuote: false, semi: true, trailingComma: "all" },
});

if (process.argv.includes("--check")) {
  const current = await readFile(outPath, "utf8").catch(() => "");
  if (current !== fresh) {
    console.error(`${outPath} is stale: run \`make contracts\``);
    process.exit(1);
  }
} else {
  await writeFile(outPath, fresh);
  console.log(`wrote ${outPath}`);
}
