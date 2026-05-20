#!/usr/bin/env bash
# Checks that the start-path tutorials reference the same transformers version
# as the README. The README "Start in 5 minutes" block / dependency snippet is
# the documented source of truth for first-run snippets.
set -euo pipefail

cd "$(dirname "$0")/.."

readme_version="$(grep -oE 'skainet-transformers-bom:[0-9]+\.[0-9]+\.[0-9]+' README.md \
  | head -n1 | cut -d: -f2)"

if [[ -z "${readme_version}" ]]; then
  echo "FAIL: could not find a skainet-transformers-bom version in README.md"
  exit 1
fi

echo "README source-of-truth version: ${readme_version}"

status=0
check() {
  local file="$1"
  if grep -q "skainet-transformers-bom:${readme_version}" "${file}"; then
    echo "OK   ${file}"
  else
    echo "FAIL ${file} does not reference skainet-transformers-bom:${readme_version}"
    status=1
  fi
}

check docs/modules/ROOT/pages/tutorials/getting-started-java.adoc
check docs/modules/ROOT/pages/tutorials/llama3-tool-calling.adoc

exit "${status}"
