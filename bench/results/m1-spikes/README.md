# Thermal spikes

The two spikes behind decision 0016, committed as run (rule 7). Each `powermetrics` log is
gzipped; the home directory in paths reads `~`. `design-checks/` holds the scripts that
re-derived 0016's figures. The three `*_probe.py` memory probes have no output here: it was not
kept, so their figures wait on the sessions' `probes.jsonl`. These scripts ran from
`bench/.scratch/thermal-spike/`, so their paths point there.

The spike's own notes:

A 45-minute test of how to benchmark on this fanless Mac. Context and the first recommendation
are in `~/phoenix/perplexity/keel/Fanless Mac Benchmarking - Recommendation.md`.

- `run_spike.sh`: the whole run in one command. Writes a `run-<stamp>/` folder here.
- `spike.py`: the driver. uLLM and two identical mlx-lm servers take turns on the same 12
  prompts. Phases: cold to hot, rest then one round, hot again, Low Power Mode.
- `sensors.py`: logs the temperature sensors once a second, no root.
- `lpm_helper.sh`: root helper that switches Low Power Mode when the driver asks.
- `pm.py`: turns a powermetrics plist into one small record per second.

In a run folder: `raw.jsonl` (one line per request), `events.jsonl` (phases and chunk
summaries), `sensors.jsonl`, `powermetrics.plist`, one log per server, `meta.json`.

Still to do after a run: the analysis, then the final method write-up.
