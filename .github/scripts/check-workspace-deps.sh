#!/bin/bash
set -euo pipefail

# Check that symbolic_regression depends on the correct version of dynamic_expressions

DE_VERSION=$(grep -m1 '^version' dynamic_expressions/Cargo.toml | cut -d'"' -f2)
SR_DEP_VERSION=$(grep '^dynamic_expressions' symbolic_regression/Cargo.toml | cut -d'"' -f2)

echo "dynamic_expressions version: $DE_VERSION"
echo "symbolic_regression depends on dynamic_expressions: $SR_DEP_VERSION"

if [ "$DE_VERSION" != "$SR_DEP_VERSION" ]; then
  echo "❌ ERROR: Version mismatch!"
  echo "   symbolic_regression must depend on dynamic_expressions = \"$DE_VERSION\""
  echo "   but it currently depends on \"$SR_DEP_VERSION\""
  echo ""
  echo "   Update symbolic_regression/Cargo.toml line:"
  echo "   dynamic_expressions = \"$DE_VERSION\""
  exit 1
fi

echo "✅ Workspace dependency versions are in sync"
