#!/bin/bash
# The whole thermal spike in one command, about 50 minutes. Quit every other app first, then:
#
#   ~/phoenix/perplexity/uLLM/bench/.scratch/thermal-spike/run_spike.sh
#
# It asks for your password once. Root is used for two things only: stopping a leftover
# powermetrics, and the helper that switches Low Power Mode on and off during the last phase.
set -u
HERE=~/phoenix/perplexity/uLLM/bench/.scratch/thermal-spike
OUT=$HERE/run-$(date +%Y%m%d-%H%M%S)
mkdir -p "$HERE/ctl"
rm -f "$HERE/ctl/stop" "$HERE/ctl/lpm_on" "$HERE/ctl/lpm_off"

sudo -v || exit 1
sudo pkill -x powermetrics
sudo "$HERE/lpm_helper.sh" >"$HERE/lpm_helper.log" 2>&1 &
sleep 2

cd ~/phoenix/perplexity/uLLM || exit 1
# 4 minutes of idle first, with the loggers running, so the machine settles after the apps close.
caffeinate -ims uv run --locked python bench/.scratch/thermal-spike/spike.py --out "$OUT" --settle 240 2>&1 |
  tee "$HERE/last-run.log"

touch "$HERE/ctl/stop"
sleep 3
echo
echo "Finished. Results: $OUT"
echo "Low Power Mode is now:$(pmset -g | grep lowpowermode)"
