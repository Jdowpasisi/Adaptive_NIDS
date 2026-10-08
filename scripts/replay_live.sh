#!/usr/bin/env bash
# C15 live mode: tcpreplay a PCAP into a veth pair and capture it with NFStream on the other end (needs sudo).
#
#   sudo bash scripts/replay_live.sh [pcap] [multiplier]       # default: data/replay/demo_slice.pcap at x1
#
# Starts its own fresh API (port 8030). Run as root because capturing needs CAP_NET_RAW. Timing features (durations,
# inter-arrival times) only match file mode at multiplier 1: tcpreplay --multiplier compresses packet gaps, so a
# sped-up replay yields DIFFERENT flows. The parity check (scripts/replay_parity.py) therefore uses x1 on a slice.
set -euo pipefail
cd "$(dirname "$0")/.."
PCAP=${1:-data/replay/demo_slice.pcap}
MULT=${2:-1}
PY=${PY:-/home/${SUDO_USER:-$USER}/.venvs/driftguard/bin/python}
RUN=live-$(date +%Y%m%dT%H%M%S)
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
cleanup() { ip link del veth0 2>/dev/null || true; }
trap cleanup EXIT
ip link add veth0 type veth peer name veth1
for i in veth0 veth1; do
  sysctl -qw "net.ipv6.conf.$i.disable_ipv6=1"        # no router solicitations etc. on the capture interface
  ip link set "$i" mtu 9000 up
done
sleep 1
PYTHONPATH=src MLFLOW_DISABLE_AGENT_HINT=1 "$PY" scripts/replay.py --mode live --source veth1 --run-id "$RUN" \
    --port 8030 --idle-stop 150 > "logs/$RUN.log" 2>&1 &
RUNNER=$!
sleep 25                                                  # API + monitor start-up
echo "replaying $PCAP at x$MULT into veth0 ..."
tcpreplay -q -i veth0 --multiplier="$MULT" "$PCAP"
echo "tcpreplay done; waiting for NFStream idle expiry (up to 150 s) ..."
wait "$RUNNER"
chown -R "${SUDO_USER:-$USER}" "data/live/runs/$RUN" "logs/$RUN.log" 2>/dev/null || true
echo "run: data/live/runs/$RUN   log: logs/$RUN.log"
echo "parity: python scripts/replay_parity.py --live data/live/runs/$RUN"
