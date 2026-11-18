#!/bin/sh
# Root helper for the thermal spike. Run it in its own Terminal window:
#
#   sudo ~/phoenix/perplexity/uLLM/bench/.scratch/thermal-spike/lpm_helper.sh
#
# It does one thing: switch Low Power Mode on or off when the spike driver asks, by dropping
# a file named lpm_on or lpm_off into the ctl folder. Ctrl-C stops it and leaves Low Power
# Mode off.
CTL="~/phoenix/perplexity/uLLM/bench/.scratch/thermal-spike/ctl"

[ "$(id -u)" = 0 ] || { echo "run this with sudo"; exit 1; }
[ -d "$CTL" ] || { echo "no ctl folder at $CTL"; exit 1; }

stop() {
  /usr/bin/pmset -a lowpowermode 0
  rm -f "$CTL/helper_alive"
  echo "$(date +%T) Low Power Mode off, helper stopped"
  exit 0
}
trap stop INT TERM

echo "Low Power Mode helper running. Leave this window open. Ctrl-C stops it."
while :; do
  date +%s >"$CTL/helper_alive"
  if [ -f "$CTL/stop" ]; then
    rm -f "$CTL/stop"
    stop
  fi
  if [ -f "$CTL/lpm_on" ]; then
    /usr/bin/pmset -a lowpowermode 1
    rm -f "$CTL/lpm_on"
    echo "$(date +%T) Low Power Mode ON"
  fi
  if [ -f "$CTL/lpm_off" ]; then
    /usr/bin/pmset -a lowpowermode 0
    rm -f "$CTL/lpm_off"
    echo "$(date +%T) Low Power Mode OFF"
  fi
  sleep 1
done
