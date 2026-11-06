#!/usr/bin/env bash
# Run a command; if it fails, wait and retry exactly once before giving up.
# Used by update-data.yml's self-hosted job around each network-heavy step
# (build/predict/score), since that runner is a Mac on home Wi-Fi -- a
# dropped connection or brief sleep mid-step is a real, seen failure mode
# there (confirmed live: GitHub's own "lost communication with the server"
# runner annotation), unlike the hosted fallback job. This only helps when
# the runner process itself survives the blip and just the step's network
# calls fail -- if the whole runner disconnects from GitHub, the job
# terminates before a retry can even run; that case is still bounded by
# update-fallback picking up the work instead.
#
#   scripts/retry_once.sh <command> [args...]
set -uo pipefail

RETRY_DELAY="${RETRY_ONCE_DELAY:-30}"

"$@"
status=$?
if [ "$status" -eq 0 ]; then
  exit 0
fi

echo "::warning::'$*' failed (exit $status) -- retrying once after ${RETRY_DELAY}s"
sleep "$RETRY_DELAY"
"$@"
