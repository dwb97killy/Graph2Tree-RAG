#!/usr/bin/env bash
# Build the knowledge graph and its vector indexes (steps 1 and 2).
# Usage: bash scripts/build_kg.sh configs/hotpotqa.yaml [--set key=value ...]
set -euo pipefail
CONFIG="${1:-configs/hotpotqa.yaml}"; shift || true

python -m gtrag.kg_construction --config "$CONFIG" "$@"   # chunks, chunk index, entities, relations
python -m gtrag.kg_embedding   --config "$CONFIG" "$@"    # entity index
