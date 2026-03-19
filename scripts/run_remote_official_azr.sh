#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_remote_official_azr.sh [MODE] [extra args...]

MODE:
  7b (default), coder7b, 14b, coder14b, llama

Examples:
  bash scripts/run_remote_official_azr.sh 7b
  bash scripts/run_remote_official_azr.sh coder7b EXTRA_ARG1=VALUE

This wrapper executes official Ray/vLLM training flows from the external
official checkout `scripts/selfplay/*.sh` for VPS-scale experimentation.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.."; pwd)"
AZR_DIR="${AZR_OFFICIAL_REPO_PATH:-$PROJECT_ROOT/Absolute-Zero-Reasoner}"

MODE="${1:-7b}"
if [[ "$MODE" == -* ]]; then
  echo "First positional argument must be mode or a script name argument (7b by default)."
  usage
  exit 2
fi

case "$MODE" in
  7b)
    SCRIPT_NAME="scripts/selfplay/7b.sh"
    ;;
  coder7b)
    SCRIPT_NAME="scripts/selfplay/coder7b.sh"
    ;;
  14b)
    SCRIPT_NAME="scripts/selfplay/14b.sh"
    ;;
  coder14b)
    SCRIPT_NAME="scripts/selfplay/coder14b.sh"
    ;;
  llama)
    SCRIPT_NAME="scripts/selfplay/llama.sh"
    ;;
  *)
    echo "Unsupported mode: $MODE"
    usage
    exit 2
    ;;
esac

if [[ "$#" -gt 0 ]]; then
  shift
fi

RUN_SCRIPT="$AZR_DIR/$SCRIPT_NAME"
if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "Expected script not found: $RUN_SCRIPT"
  echo "Set AZR_OFFICIAL_REPO_PATH to your external AZR checkout if needed:"
  echo "  export AZR_OFFICIAL_REPO_PATH=/path/to/AshutoshBuilds-Absolute-Zero-Reasoner"
  exit 2
fi

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export RAY_LOGGING_LEVEL="${RAY_LOGGING_LEVEL:-DEBUG}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

cd "$AZR_DIR"
echo "Launching official AZR Ray/vLLM flow:"
echo "  mode: $MODE"
echo "  script: $SCRIPT_NAME"
bash "$RUN_SCRIPT" "$@"
