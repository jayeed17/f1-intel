#!/usr/bin/env bash
# Manually complete any partial prebuilt sessions and push the result.
# Run anytime, from anywhere, on this Mac:
#
#   ./scripts/complete_partials.sh
#
# Does the same four things the self-hosted Action job does (pull, build
# --only-missing, commit, push) -- useful right after the Mac comes back
# online, or any time you just want a fresher build without waiting for
# the next cron. Uses its own .venv (created on first run) with the
# pinned requirements-dev.txt, never whatever "python3" happens to resolve
# to interactively (e.g. an anaconda env) -- same discipline as
# update-data.yml's self-hosted job, and for the same reason: a joblib
# model pickled under one scikit-learn version can fail to load under
# another (see CLAUDE.md's "Gotchas").
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> git pull"
git pull --ff-only

if [ ! -x .venv/bin/python ]; then
  echo "==> No .venv yet -- creating one"
  rm -rf .venv
  python3 -m venv .venv
fi

RESOLVED="$(.venv/bin/python -c 'import sys; print(sys.executable)')"
case "$RESOLVED" in
  *anaconda*|*miniconda*|*/conda/*|*condabin*)
    echo "Refusing to run: .venv resolved to a conda Python ($RESOLVED)." >&2
    echo "Delete .venv and make sure 'python3' on PATH is not a conda interpreter, then re-run." >&2
    exit 1
    ;;
esac

echo "==> Installing pinned requirements"
.venv/bin/pip install --upgrade pip --quiet
.venv/bin/pip install -r requirements-dev.txt --quiet

echo "==> Building missing/partial sessions"
set +e
.venv/bin/python -m scripts.build_prebuilt --year 2026 --only-missing
BUILD_STATUS=$?
set -e
if [ "$BUILD_STATUS" -ne 0 ]; then
  echo "!! build_prebuilt.py exited non-zero ($BUILD_STATUS) -- it does this when a session" >&2
  echo "   whose scheduled time is >6h in the past is still entirely missing (see its output" >&2
  echo "   above). Committing whatever it DID manage to build anyway -- best-effort, not an" >&2
  echo "   all-or-nothing run." >&2
fi

echo "==> Committing and pushing if anything changed"
git add data/prebuilt predictions
if git diff --cached --quiet; then
  echo "No data changes."
else
  git commit -m "Complete partial prebuilt sessions ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
  git push
fi
