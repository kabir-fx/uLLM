#!/usr/bin/env bash
# M1-T7 baselines (B4, decision 0014): six B5 runs one after another, then the comparison table.
#
#   OUT    where the run directories go          (default bench/results/m1)
#   ARGS   extra bench.run arguments              (default none)
#   GAP    seconds of idle between runs           (default 120)
#
# A smoke pass of the commands:
#   OUT=bench/.scratch/m1 ARGS="--smoke --requests 2 --repeats 1 --warmup 1 --cooldown 0" GAP=0 \
#     bench/baselines.sh
set -euo pipefail

OUT=${OUT:-bench/results/m1}
GAP=${GAP:-120}
read -r -a args <<<"${ARGS:-}"
uv="uv run --locked"

# mlx_lm.server gets the pinned snapshot by path: given a repo id it would load the main ref.
model=$($uv python scripts/fetch_models.py 2>/dev/null | sed -n 's/^qwen3-0.6b-bf16: //p')
[ -d "$model" ] || { echo "no local snapshot of the default model: run make models" >&2; exit 1; }
mlx_lm="$uv python scripts/mlx_lm_server.py --model $model --port 8000"
# A seeded request takes mlx_lm.server's single-request path; any other request is batched.
sequential=(--model "$model" --extra-body '{"seed": 0}')

run() { # target concurrency launch [bench.run arguments...]
  local target=$1 concurrency=$2 launch=$3
  shift 3
  # macOS bash 3.2 under set -u calls an empty "${args[@]}" unbound.
  $uv python -m bench.run ${args[@]+"${args[@]}"} --target "$target" --concurrency "$concurrency" \
    --launch "$launch" --out "$OUT/$target-c$concurrency" "$@"
}

run ullm 1 "$uv ullm serve"
sleep "$GAP"
run mlx-lm-sequential-no-cache 1 "$mlx_lm --prompt-cache-size 0" "${sequential[@]}"
sleep "$GAP"
run mlx-lm-sequential 1 "$mlx_lm" "${sequential[@]}"
sleep "$GAP"
run ullm 8 "$uv ullm serve"
sleep "$GAP"
run mlx-lm-batched 8 "$mlx_lm" --model "$model"
sleep "$GAP"
run mlx-lm-batched-no-cache 8 "$mlx_lm --prompt-cache-size 0" --model "$model"

$uv python -m bench.analysis "$OUT"/*-c1 "$OUT"/*-c8 --compare | grep '^|' >"$OUT/comparison.md"
cat "$OUT/comparison.md"
