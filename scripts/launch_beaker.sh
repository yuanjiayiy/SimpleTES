#!/usr/bin/env bash
# Launch a SimpleTES run on Beaker for any task under datasets/<family>/<subtask>/.
#
# Stages the current working tree (engine + the task's family directory), uploads it as
# a Beaker dataset, and submits an experiment whose job:
#   1. builds the engine env (uv sync --frozen) and the family's eval venv at
#      datasets/<family>/venv, from setup.sh, pyproject.toml or requirements.txt
#      (whichever exists first), where the engine auto-detects it;
#   2. runs main.py with checkpoints on WEKA under <weka-root>/<name>/;
#   3. on preemption, Beaker re-runs the job and it resumes from the latest db_state_*.
#
# Nothing here is task-specific. From the task directory it takes init_program.*,
# evaluator.py, the instruction (<subtask>.txt, else the first *.txt) and the evaluator's
# TIMEOUT_SECONDS (used as --eval-timeout). Search settings default to the SimpleTES
# paper's (arXiv 2604.19341): C=32, K=16, L=100, RPUCG with 5 inspirations, reflection,
# failure patterns and warm-start constructions. Anything after `--` is passed to
# main.py last, so it overrides these defaults.
#
# Usage:
#   scripts/launch_beaker.sh [options] TASK_DIR [-- MAIN_PY_ARGS...]
#
# Examples:
#   scripts/launch_beaker.sh --name ac3-paper datasets/autocorrelation/autocorrelation_third
#   scripts/launch_beaker.sh --name ac3-small datasets/autocorrelation/autocorrelation_third \
#       -- --num-chains 4 --k-candidates 4 --max-generations 1600 --selector balance
#   # The paper's single best-solution restart, from a finished run's checkpoint on WEKA:
#   scripts/launch_beaker.sh --name ac3-paper-restart datasets/autocorrelation/autocorrelation_third \
#       --restart-from ac3-paper/2026-10-05/instance-xxxx/db_state_yyyy
#
# Task options:
#   --name NAME              Run name (required): output dir <weka-root>/NAME, experiment name.
#   --init-program PATH      Starting program (default: TASK_DIR/init_program.*).
#   --init-construction PATH Seed every chain's GLOBAL_BEST_CONSTRUCTION from this JSON.
#   --restart-from PATH      Best-solution restart: start from that checkpoint's best_program.py
#                            and its best node's construction. PATH is a db_state_* dir on WEKA,
#                            absolute or relative to --weka-root.
#   --instruction PATH       Instruction file (default: auto-detected in TASK_DIR).
#   --model MODEL            LiteLLM model (default: openrouter/openai/gpt-oss-120b).
#   --secret ENV=SECRET      Expose Beaker workspace secret SECRET as ENV in the job; repeatable.
#                            Default for openrouter/ models: OPENROUTER_API_KEY=OPENROUTER_API_KEY_SOCIALRL.
#   --no-construction        Do not pass --include-construction.
#   --no-llm-io              Do not save LLM prompts/responses in checkpoints (--save-llm-io).
#
# Beaker options:
#   --gen-concurrency N      LLM calls in flight (default: 64).
#   --eval-concurrency N     Evaluations in flight (default: 128).
#   --cpus N                 CPU request (default: eval concurrency + 8).
#   --memory SIZE            Memory request (default: 192GiB).
#   --gpus N                 GPU request, for GPU tasks (default: 0).
#   --cluster C              Cluster; repeatable (default: ai2/holmes and ai2/titan).
#   --priority P             Priority (default: urgent).
#   --min-runtime DUR        Run at least this long before Beaker may preempt it, e.g. 1h
#                            (default: 0s, i.e. preemptible at any time; auto-resume stays on).
#   --workspace WS           Workspace (default: ai2/carriey).
#   --image IMAGE            Beaker image (default: 01KY5WDGQKDE194RY8PED6MQGE). Tasks that need
#                            extra toolchains (e.g. cargo for qubit_routing) need an image with them.
#   --weka-root PATH         Output root (default: /weka/oe-adapt-default/carriey/simpletes).
#   --dry-run                Stage files and print the spec and main.py arguments; submit nothing.
#   -h, --help               Show this help.
#
# Requires: beaker CLI (logged in), python3, rsync. Works with macOS bash 3.2.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME=""
TASK_DIR=""
INIT_PROGRAM=""
INIT_CONSTRUCTION=""
RESTART_FROM=""
INSTRUCTION=""
MODEL="openrouter/openai/gpt-oss-120b"
SECRETS=""
CONSTRUCTION=1
LLM_IO=1
GEN_CONCURRENCY=64
EVAL_CONCURRENCY=128
CPUS=""
MEMORY="192GiB"
GPUS=0
CLUSTERS=""
PRIORITY="urgent"
MIN_RUNTIME="0s"
WORKSPACE="ai2/carriey"
IMAGE="01KY5WDGQKDE194RY8PED6MQGE"
WEKA_ROOT="/weka/oe-adapt-default/carriey/simpletes"
DRY_RUN=0
EXTRA=()

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift ;;
    --init-program) INIT_PROGRAM="$2"; shift ;;
    --init-construction) INIT_CONSTRUCTION="$2"; shift ;;
    --restart-from) RESTART_FROM="${2%/}"; shift ;;
    --instruction) INSTRUCTION="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --secret) SECRETS="$SECRETS $2"; shift ;;
    --no-construction) CONSTRUCTION=0 ;;
    --no-llm-io) LLM_IO=0 ;;
    --gen-concurrency) GEN_CONCURRENCY="$2"; shift ;;
    --eval-concurrency) EVAL_CONCURRENCY="$2"; shift ;;
    --cpus) CPUS="$2"; shift ;;
    --memory) MEMORY="$2"; shift ;;
    --gpus) GPUS="$2"; shift ;;
    --cluster) CLUSTERS="$CLUSTERS $2"; shift ;;
    --priority) PRIORITY="$2"; shift ;;
    --min-runtime) MIN_RUNTIME="$2"; shift ;;
    --workspace) WORKSPACE="$2"; shift ;;
    --image) IMAGE="$2"; shift ;;
    --weka-root) WEKA_ROOT="${2%/}"; shift ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    -*) die "unknown option $1 (pass main.py flags after --)" ;;
    *) [ -z "$TASK_DIR" ] || die "only one TASK_DIR is allowed"; TASK_DIR="${1%/}" ;;
  esac
  shift
done

[ -n "$NAME" ] || die "--name is required"
case "$NAME" in *[!a-zA-Z0-9._-]*|"") die "--name may only contain letters, digits, '.', '_' and '-'" ;; esac
[ -n "$TASK_DIR" ] || die "TASK_DIR is required (e.g. datasets/autocorrelation/autocorrelation_third)"
[ -z "$INIT_PROGRAM" ] || [ -z "$RESTART_FROM" ] || die "--init-program and --restart-from are mutually exclusive"
[ -z "$INIT_CONSTRUCTION" ] || [ -z "$RESTART_FROM" ] || die "--init-construction and --restart-from are mutually exclusive"
for cmd in beaker python3 rsync; do command -v "$cmd" >/dev/null || die "required command not found: $cmd"; done
[ -n "$CLUSTERS" ] || CLUSTERS="ai2/holmes ai2/titan"
[ -n "$CPUS" ] || CPUS=$((EVAL_CONCURRENCY + 8))
case "$MODEL" in openrouter/*) [ -n "$SECRETS" ] || SECRETS="OPENROUTER_API_KEY=OPENROUTER_API_KEY_SOCIALRL" ;; esac

# Paths inside the repo, as given relative to it.
rel_to_repo() {
  local abs
  abs="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
  case "$abs" in "$REPO"/*) echo "${abs#"$REPO"/}" ;; *) echo "" ;; esac
}

[ -f "$TASK_DIR/evaluator.py" ] || TASK_DIR="$REPO/$TASK_DIR"
[ -f "$TASK_DIR/evaluator.py" ] || die "no evaluator.py in $TASK_DIR"
TASK_REL="$(rel_to_repo "$TASK_DIR/evaluator.py")"; TASK_REL="${TASK_REL%/evaluator.py}"
case "$TASK_REL" in datasets/*/*) ;; *) die "TASK_DIR must be datasets/<family>/<subtask> inside $REPO" ;; esac
FAMILY_REL="$(dirname "$TASK_REL")"
SUBTASK="$(basename "$TASK_REL")"

if [ -z "$INSTRUCTION" ]; then
  if [ -f "$REPO/$TASK_REL/$SUBTASK.txt" ]; then INSTRUCTION="$TASK_REL/$SUBTASK.txt"
  else
    first_txt="$(cd "$REPO" && ls "$TASK_REL"/*.txt 2>/dev/null | head -1 || true)"
    [ -n "$first_txt" ] || die "no instruction .txt in $TASK_REL; pass --instruction"
    INSTRUCTION="$first_txt"
  fi
fi

# Extra inputs (init program / construction) outside the task dir are staged under inputs/.
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/simpletes_launch.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
stage_input() {  # $1 = local path -> echoes the path the job should use
  local rel
  [ -f "$1" ] || die "file not found: $1"
  rel="$(rel_to_repo "$1")"
  if [ -n "$rel" ]; then mkdir -p "$STAGE/$(dirname "$rel")"; cp "$1" "$STAGE/$rel"; echo "$rel"
  else mkdir -p "$STAGE/inputs"; cp "$1" "$STAGE/inputs/$(basename "$1")"; echo "inputs/$(basename "$1")"
  fi
}
INSTRUCTION="$(stage_input "$( [ -f "$INSTRUCTION" ] && echo "$INSTRUCTION" || echo "$REPO/$INSTRUCTION")")"
if [ -z "$RESTART_FROM" ]; then
  if [ -z "$INIT_PROGRAM" ]; then
    INIT_PROGRAM="$(cd "$REPO" && ls "$TASK_REL"/init_program.* 2>/dev/null | head -1 || true)"
    [ -n "$INIT_PROGRAM" ] || die "no init_program.* in $TASK_REL; pass --init-program"
    INIT_PROGRAM="$REPO/$INIT_PROGRAM"
  fi
  INIT_PROGRAM="$(stage_input "$INIT_PROGRAM")"
  [ -z "$INIT_CONSTRUCTION" ] || INIT_CONSTRUCTION="$(stage_input "$INIT_CONSTRUCTION")"
else
  case "$RESTART_FROM" in /*) ;; *) RESTART_FROM="$WEKA_ROOT/$RESTART_FROM" ;; esac
fi

# Evaluation timeout: the evaluator's own TIMEOUT_SECONDS (its last integer literal).
EVAL_TIMEOUT="$(grep -m1 -E '^[[:space:]]*TIMEOUT_SECONDS[[:space:]]*=' "$REPO/$TASK_REL/evaluator.py" \
  | grep -oE '[0-9]+' | tail -1 || true)"

# Stage the engine and the task family (minus local envs and caches).
(cd "$REPO" && rsync -aR --exclude venv --exclude .venv --exclude __pycache__ --exclude '*.pyc' \
  --exclude .git --exclude checkpoints --exclude target \
  main.py sitecustomize.py pyproject.toml uv.lock README.md LICENSE simpletes "$FAMILY_REL" "$STAGE/")

# main.py arguments, one per line; the job reads them back into an array.
{
  printf '%s\n' --evaluator "$TASK_REL/evaluator.py" --instruction "$INSTRUCTION" --model "$MODEL"
  [ -z "$INIT_PROGRAM" ] || printf '%s\n' --init-program "$INIT_PROGRAM"
  printf '%s\n' --selector rpucg --num-chains 32 --k-candidates 16 --max-generations 51200 --num-inspirations 5
  [ -z "$EVAL_TIMEOUT" ] || printf '%s\n' --eval-timeout "$EVAL_TIMEOUT"
  [ "$CONSTRUCTION" = 0 ] || printf '%s\n' --include-construction
  [ "$LLM_IO" = 0 ] || printf '%s\n' --save-llm-io
  printf '%s\n' --gzip --log-interval 1024 --gen-concurrency "$GEN_CONCURRENCY" --eval-concurrency "$EVAL_CONCURRENCY"
  [ ${#EXTRA[@]} -eq 0 ] || printf '%s\n' "${EXTRA[@]}"
} > "$STAGE/main_args.txt"

cat > "$STAGE/launch.env" <<EOF
OUT=$(printf '%q' "$WEKA_ROOT/$NAME")
FAMILY=$(printf '%q' "$FAMILY_REL")
INIT_CONSTRUCTION=$(printf '%q' "$INIT_CONSTRUCTION")
RESTART_FROM=$(printf '%q' "$RESTART_FROM")
EOF

cat > "$STAGE/run_job.sh" <<'JOB'
#!/usr/bin/env bash
# Generated by scripts/launch_beaker.sh; runs inside the Beaker job.
set -euo pipefail
source /code/launch.env
mapfile -t ARGS < /code/main_args.txt
mkdir -p "$OUT"
WORK=/tmp/SimpleTES
rm -rf "$WORK" && cp -r /code "$WORK" && cd "$WORK"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --frozen

# The family's eval venv at datasets/<family>/venv, where the engine auto-detects it.
if [ -f "$FAMILY/setup.sh" ]; then
  (cd "$FAMILY" && bash setup.sh)
elif [ -f "$FAMILY/pyproject.toml" ]; then
  (cd "$FAMILY" && UV_PROJECT_ENVIRONMENT=venv uv sync)
elif [ -f "$FAMILY/requirements.txt" ]; then
  (cd "$FAMILY" && uv venv venv && uv pip install --python venv/bin/python -r requirements.txt)
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ -n "$RESTART_FROM" ]; then
  # Best-solution restart: same settings, fresh history, seeded from the checkpoint's best.
  ARGS+=(--init-program "$RESTART_FROM/best_program.py")
  BEST_ID=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_node_id'])" "$RESTART_FROM/metadata.json")
  SEED="$RESTART_FROM/shared_constructions/$BEST_ID.json"
  if [ -f "$SEED" ]; then export SIMPLETES_INITIAL_SHARED_CONSTRUCTION_PATH="$SEED"
  else echo "No construction snapshot for best node $BEST_ID; restarting from the program only"; fi
elif [ -n "$INIT_CONSTRUCTION" ]; then
  export SIMPLETES_INITIAL_SHARED_CONSTRUCTION_PATH="$WORK/$INIT_CONSTRUCTION"
fi

LATEST=$(ls -dt "$OUT"/*/instance-*/db_state_* 2>/dev/null | head -1 || true)
if [ -n "$LATEST" ]; then
  echo "Resuming from $LATEST"
  ARGS+=(--resume "$LATEST")
fi

echo "main.py ${ARGS[*]}"
uv run --frozen python main.py "${ARGS[@]}" --output-path "$OUT" 2>&1 | tee -a "$OUT/run_${BEAKER_JOB_ID:-local}.log"
JOB
chmod +x "$STAGE/run_job.sh"

ACCOUNT="$(beaker account whoami --format json 2>/dev/null \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d["name"])')"
STAMP="$(date +%Y%m%d-%H%M%S)"
EXPERIMENT="simpletes-$NAME-$STAMP"
DATASET="simpletes-$NAME-code-$STAMP"
SPEC="$STAGE/spec.json"
NAME="$NAME" EXPERIMENT="$EXPERIMENT" WORKSPACE="$WORKSPACE" ACCOUNT="$ACCOUNT" DATASET="$DATASET" IMAGE="$IMAGE" \
WEKA_ROOT="$WEKA_ROOT" SECRETS="$SECRETS" CPUS="$CPUS" MEMORY="$MEMORY" GPUS="$GPUS" \
CLUSTERS="$CLUSTERS" PRIORITY="$PRIORITY" MIN_RUNTIME="$MIN_RUNTIME" TASK_REL="$TASK_REL" MODEL="$MODEL" python3 - "$SPEC" <<'PY'
import json, os, sys

e = os.environ
bucket = e["WEKA_ROOT"].split("/")[2]  # /weka/<bucket>/...
env_vars = []
for item in e["SECRETS"].split():
    name, _, secret = item.partition("=")
    if not secret:
        sys.exit(f"error: --secret expects ENV=SECRET, got {item!r}")
    env_vars.append({"name": name, "secret": secret})
resources = {"cpuCount": float(e["CPUS"]), "memory": e["MEMORY"]}
if int(e["GPUS"]):
    resources["gpuCount"] = int(e["GPUS"])
spec = {
    "version": "v2",
    "description": f"SimpleTES run '{e['NAME']}' on {e['TASK_REL']} with {e['MODEL']} "
                   f"(scripts/launch_beaker.sh); checkpoints under {e['WEKA_ROOT']}/{e['NAME']}",
    "tasks": [{
        "name": f"simpletes-{e['NAME']}",
        "image": {"beaker": e["IMAGE"]},
        "command": ["bash", "-lc", "bash /code/run_job.sh"],
        "envVars": env_vars,
        "datasets": [
            {"mountPath": "/code", "source": {"beaker": f"{e['ACCOUNT']}/{e['DATASET']}"}},
            {"mountPath": f"/weka/{bucket}", "source": {"weka": bucket}},
        ],
        "result": {"path": "/results"},
        "resources": resources,
        "context": {"priority": e["PRIORITY"], "minRuntime": e["MIN_RUNTIME"], "autoResume": True},
        "constraints": {"cluster": e["CLUSTERS"].split()},
        "hostNetworking": True,
    }],
}
with open(sys.argv[1], "w") as f:
    json.dump(spec, f, indent=2)
PY

echo "Task:        $TASK_REL"
echo "Output:      $WEKA_ROOT/$NAME"
echo "main.py:     $(tr '\n' ' ' < "$STAGE/main_args.txt")"
[ -z "$RESTART_FROM" ] || echo "Restart:     $RESTART_FROM"
echo "Staged code: $(du -sh "$STAGE" | cut -f1)"
if [ "$DRY_RUN" = 1 ]; then
  cat "$SPEC"
  exit 0
fi

beaker dataset create "$STAGE" --name "$DATASET" --workspace "$WORKSPACE" >/dev/null
echo "Uploaded code dataset $DATASET"
beaker experiment create "$SPEC" --name "$EXPERIMENT" --workspace "$WORKSPACE" 2>&1 \
  | grep -v -e 'newer version' -e 'To update' -e 'curl -fsSL' -e '^$'
