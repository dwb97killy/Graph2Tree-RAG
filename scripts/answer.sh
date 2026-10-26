#!/usr/bin/env bash
# Answer the questions of one dataset with GTRAG (after scripts/build_kg.sh).
# Usage: bash scripts/answer.sh configs/hotpotqa.yaml --set llm.model_path=/path/to/model
set -euo pipefail
CONFIG="${1:-configs/hotpotqa.yaml}"; shift || true

python -m gtrag.answer --config "$CONFIG" "$@"
