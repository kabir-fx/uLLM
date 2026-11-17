#!/usr/bin/env bash
# Launch `ullm serve`, drive every endpoint with curl, assert on the answers, stop the server.
# Usage (from the repo root): .claude/skills/run-ullm/smoke.sh [--real] [--port N] [--keep]
#   --real   load the default MLX model instead of FakeRunner (needs `make models` once)
#   --keep   leave the server running afterwards and print its pid
# Prints each request and its response; exits non-zero on the first failed check.
set -uo pipefail

PORT=8765 REAL=0 KEEP=0
while [ $# -gt 0 ]; do
  case $1 in
    --real) REAL=1 ;;
    --keep) KEEP=1 ;;
    --port) PORT=$2; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

BASE=http://127.0.0.1:$PORT
LOG=${TMPDIR:-/tmp}/ullm-smoke-$PORT.log
fail() { echo "FAIL: $*" >&2; echo "--- server log ($LOG) ---" >&2; tail -30 "$LOG" >&2; exit 1; }

if curl -s -o /dev/null "$BASE/health"; then
  echo "something already listens on port $PORT; pick another with --port" >&2
  exit 2
fi

args=(serve --port "$PORT")
[ $REAL = 0 ] && args+=(--fake)
uv run --locked ullm "${args[@]}" >"$LOG" 2>&1 &
PID=$!
[ $KEEP = 0 ] && trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT

# The real server loads weights before it listens.
for _ in $(seq 1 600); do
  curl -sf -o /dev/null "$BASE/health" && break
  kill -0 $PID 2>/dev/null || fail "server exited during startup"
  sleep 0.2
done
curl -sf -o /dev/null "$BASE/health" || fail "no healthy /health after 120 s"

say() { printf '\n== %s\n' "$*"; }

say "GET /health"
curl -s "$BASE/health"; echo

say "GET /v1/models"
models=$(curl -s "$BASE/v1/models"); echo "$models"
MODEL=$(printf '%s' "$models" | sed -E 's/.*"id":"([^"]+)".*/\1/')
[ -n "$MODEL" ] || fail "no model id in /v1/models"

chat() { # $1: extra JSON fields
  printf '{"model":"%s","messages":[{"role":"user","content":"Name three primary colours."}]%s}' "$MODEL" "$1"
}

say "POST /v1/chat/completions (plain)"
out=$(curl -s -w '\n%{http_code}' "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d "$(chat ',"max_tokens":24,"temperature":0')")
echo "$out"
[ "$(tail -1 <<<"$out")" = 200 ] || fail "plain completion was not 200"
grep -q '"finish_reason"' <<<"$out" || fail "no finish_reason"

say "POST /v1/chat/completions (stream)"
out=$(curl -sN "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d "$(chat ',"max_tokens":8,"temperature":0,"stream":true,"stream_options":{"include_usage":true}')")
echo "$out"
grep -q '^data: \[DONE\]' <<<"$out" || fail "stream did not end with [DONE]"
grep -q '"usage":{' <<<"$out" || fail "stream carried no usage chunk"

say "unknown model -> 404"
out=$(curl -s -w '\n%{http_code}' "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"nope","messages":[{"role":"user","content":"hi"}]}')
echo "$out"
[ "$(tail -1 <<<"$out")" = 404 ] || fail "unknown model was not 404"

say "text/plain body -> 400"
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/chat/completions" \
  -H 'content-type: text/plain' -d "$(chat '')")
echo "$code"
[ "$code" = 400 ] || fail "text/plain was not refused"

say "foreign Host header -> 400"
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/models" -H 'Host: evil.example')
echo "$code"
[ "$code" = 400 ] || fail "foreign Host was not refused"

say "client disconnect mid-stream, then /health"
curl -sN --max-time 0.3 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d "$(chat ',"max_tokens":4000,"stream":true')" >/dev/null
sleep 0.5
curl -s "$BASE/health"; echo
curl -sf -o /dev/null "$BASE/health" || fail "engine unhealthy after a disconnect"

echo
if [ $KEEP = 1 ]; then
  echo "smoke: ok; server still running, pid $PID, log $LOG"
else
  echo "smoke: ok"
fi
