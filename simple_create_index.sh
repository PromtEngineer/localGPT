#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo 'Usage: ./simple_create_index.sh "Index name" document [document ...]' >&2
  exit 2
fi

index_name=$1
shift
exec python3 create_index_script.py --name "$index_name" "$@"
