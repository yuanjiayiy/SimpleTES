#!/usr/bin/env bash
# Mirror SimpleTES run results from WEKA to this machine over WEKA's S3 interface.
#
# Runs `aws --profile weka-aus --endpoint-url "$WEKA_ENDPOINT_URL" s3 sync` from
# s3://oe-adapt-default/carriey/simpletes/ into a local directory. Only new or changed
# files are transferred, and --delete keeps the copy an exact mirror: the engine keeps
# just the latest db_state_* per run, so a superseded local checkpoint is removed too.
#
# Skipped by default: each run's instance-level shared_constructions/ (one small JSON
# per chain-best snapshot, often thousands of files). The copies inside db_state_*
# are always synced. Pass --with-snapshots to include them.
#
# Usage:
#   scripts/sync_weka_results.sh [options] [SUBPATH]
#
#   SUBPATH           Path under the bucket prefix to sync, e.g. ac3_paper_replication/phase1
#                     (default: everything under the prefix).
#
# Options:
#   --dest DIR        Local mirror directory (default: checkpoints/weka). --delete only
#                     runs in a directory this script created (it holds a .weka-sync marker).
#   --src URI         S3 prefix (default: s3://oe-adapt-default/carriey/simpletes/).
#   --profile NAME    AWS profile (default: weka-aus).
#   --with-snapshots  Also sync instance-level shared_constructions/.
#   --no-delete       Keep local files that no longer exist on WEKA.
#   --html            Build HTML reports (scripts/checkpoint_to_html.py) for every synced
#                     checkpoint that is new or changed, into checkpoints/weka_html/<run>/.
#   --dryrun          Show what would be transferred or deleted, change nothing.
#   -h, --help        Show this help.
#
# Requires: aws CLI v2 with the profile configured, WEKA_ENDPOINT_URL set, and uv for --html.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="s3://oe-adapt-default/carriey/simpletes/"
DEST="$REPO/checkpoints/weka"
HTML_ROOT="$REPO/checkpoints/weka_html"
PROFILE="weka-aus"
WITH_SNAPSHOTS=0
DELETE=1
HTML=0
DRYRUN=0
SUBPATH=""
MARKER=".weka-sync"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift ;;
    --src) SRC="${2%/}/"; shift ;;
    --profile) PROFILE="$2"; shift ;;
    --with-snapshots) WITH_SNAPSHOTS=1 ;;
    --no-delete) DELETE=0 ;;
    --html) HTML=1 ;;
    --dryrun|--dry-run) DRYRUN=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "error: unknown option $1" >&2; usage >&2; exit 2 ;;
    *) [ -z "$SUBPATH" ] || { echo "error: only one SUBPATH is allowed" >&2; exit 2; }
       SUBPATH="${1#/}"; SUBPATH="${SUBPATH%/}" ;;
  esac
  shift
done
case "$SUBPATH" in *..*) echo "error: SUBPATH must not contain '..'" >&2; exit 2 ;; esac
command -v aws >/dev/null || { echo "error: aws CLI not found" >&2; exit 1; }
[ -n "${WEKA_ENDPOINT_URL:-}" ] || { echo "error: WEKA_ENDPOINT_URL is not set" >&2; exit 1; }

# Refuse to --delete inside a directory this script did not create.
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
if [ ! -e "$DEST/$MARKER" ]; then
  if [ -n "$(find "$DEST" -type f | head -1)" ] && [ "$DELETE" = 1 ]; then
    echo "error: $DEST is not empty and has no $MARKER marker; pass --no-delete or use an empty --dest" >&2
    exit 1
  fi
  [ "$DRYRUN" = 1 ] || echo "Mirror of $SRC created by scripts/sync_weka_results.sh" > "$DEST/$MARKER"
fi

REMOTE="$SRC${SUBPATH:+$SUBPATH/}"
LOCAL="$DEST${SUBPATH:+/$SUBPATH}"
[ "$DRYRUN" = 1 ] || mkdir -p "$LOCAL"

ARGS=(--profile "$PROFILE" --endpoint-url "$WEKA_ENDPOINT_URL" s3 sync "$REMOTE" "$LOCAL" --exclude "$MARKER")
if [ "$WITH_SNAPSHOTS" = 0 ]; then
  # Later filters win: drop instance-level snapshots, keep the ones inside checkpoints.
  ARGS+=(--exclude "*shared_constructions/*" --include "*db_state_*/shared_constructions/*")
fi
[ "$DELETE" = 1 ] && ARGS+=(--delete)
[ "$DRYRUN" = 1 ] && ARGS+=(--dryrun)

echo "Syncing $REMOTE -> $LOCAL"
aws "${ARGS[@]}"
[ "$DRYRUN" = 1 ] && exit 0
echo "Done. Local mirror: $LOCAL ($(du -sh "$LOCAL" | cut -f1))"

if [ "$HTML" = 1 ]; then
  # Rebuild a run's reports when its checkpoint is newer than the last build.
  find "$LOCAL" -type d -name 'db_state_*' -prune | sort | while IFS= read -r ckpt; do
    rel="${ckpt#"$DEST"/}"
    out="$HTML_ROOT/$(dirname "$rel")"
    if [ -f "$out/index.html" ] && [ -z "$(find "$ckpt" -maxdepth 1 -name 'nodes.json*' -newer "$out/index.html")" ]; then
      echo "HTML up to date for $rel"
      continue
    fi
    echo "Building HTML for $rel -> $out"
    (cd "$REPO" && uv run --frozen python scripts/checkpoint_to_html.py "$ckpt" --out "$out" \
      --source "${SRC}${rel}")
  done
fi
