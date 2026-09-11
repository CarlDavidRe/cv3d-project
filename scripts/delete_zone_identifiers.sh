#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
deleted=0

while IFS= read -r -d '' file; do
    rm -- "$file"
    printf 'Deleted: %s\n' "${file#"$repo_root"/}"
    ((deleted += 1))
done < <(
    find "$repo_root" -type f -name '*Zone.Identifier' \
        ! -path "$repo_root/.git/*" -print0
)

printf 'Removed %d Zone.Identifier file(s).\n' "$deleted"
