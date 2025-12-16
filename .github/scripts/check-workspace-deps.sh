#!/bin/bash
set -euo pipefail

# Invariants:
# - Release tooling requires literal `version = "x.y.z"` strings in member Cargo.toml.
# - `dynamic_expressions` + `symbolic_regression` versions stay in lockstep.
# - Any internal `path` dependency must also specify an explicit `version` that matches the depended-on crate.
# - release-please manifest package keys are workspace-relative paths.

fail() {
  echo "❌ ERROR: $1"
  exit 1
}

require_no_workspace_inheritance() {
  if grep -R -nE '^[[:space:]]*version[[:space:]]*\.[[:space:]]*workspace[[:space:]]*=' Cargo.toml dynamic_expressions/Cargo.toml symbolic_regression/Cargo.toml web/wasm/Cargo.toml >/dev/null; then
    fail "found version.workspace usage; release tooling requires literal version strings"
  fi
  if grep -R -nE '\{[^}]*workspace[[:space:]]*=[[:space:]]*true[^}]*\}' dynamic_expressions/Cargo.toml symbolic_regression/Cargo.toml web/wasm/Cargo.toml >/dev/null; then
    fail "found workspace=true dependency usage; release tooling requires explicit version strings"
  fi
}

toml_get_package_version() {
  local manifest_path="$1"
  python3 - "$manifest_path" <<'PY'
import sys
import tomllib

path = sys.argv[1]
with open(path, "rb") as f:
    data = tomllib.load(f)
version = data.get("package", {}).get("version")
if not isinstance(version, str) or not version:
    raise SystemExit(f"missing [package].version in {path}")
print(version)
PY
}

toml_get_dep_version_and_path() {
  local manifest_path="$1"
  local dep_name="$2"
  python3 - "$manifest_path" "$dep_name" <<'PY'
import sys
import tomllib

path = sys.argv[1]
dep = sys.argv[2]
with open(path, "rb") as f:
    data = tomllib.load(f)
deps = data.get("dependencies", {})
if dep not in deps:
    raise SystemExit(f"{path}: dependency {dep} not found in [dependencies]")
spec = deps[dep]
if isinstance(spec, str):
    raise SystemExit(f"{path}: dependency {dep} must use inline table with version+path")
if not isinstance(spec, dict):
    raise SystemExit(f"{path}: dependency {dep} has unsupported format")
version = spec.get("version")
dep_path = spec.get("path")
if not isinstance(version, str) or not version:
    raise SystemExit(f"{path}: dependency {dep} missing version")
if not isinstance(dep_path, str) or not dep_path:
    raise SystemExit(f"{path}: dependency {dep} missing path")
print(version + "\t" + dep_path)
PY
}

require_no_workspace_inheritance

DE_VERSION="$(toml_get_package_version dynamic_expressions/Cargo.toml)"
SR_VERSION="$(toml_get_package_version symbolic_regression/Cargo.toml)"
SRW_VERSION="$(toml_get_package_version web/wasm/Cargo.toml)"

echo "dynamic_expressions version: $DE_VERSION"
echo "symbolic_regression version: $SR_VERSION"
echo "symbolic_regression_wasm version: $SRW_VERSION"

if [ "$DE_VERSION" != "$SR_VERSION" ]; then
  fail "dynamic_expressions and symbolic_regression must share the same version"
fi

read -r SR_DE_DEP_VERSION SR_DE_DEP_PATH < <(toml_get_dep_version_and_path symbolic_regression/Cargo.toml dynamic_expressions)
if [ "$SR_DE_DEP_VERSION" != "$DE_VERSION" ]; then
  fail "symbolic_regression -> dynamic_expressions version ($SR_DE_DEP_VERSION) must match dynamic_expressions version ($DE_VERSION)"
fi
if [ "$SR_DE_DEP_PATH" != "../dynamic_expressions" ]; then
  fail "symbolic_regression -> dynamic_expressions path must be ../dynamic_expressions (got $SR_DE_DEP_PATH)"
fi

read -r SRW_SR_DEP_VERSION SRW_SR_DEP_PATH < <(toml_get_dep_version_and_path web/wasm/Cargo.toml symbolic_regression)
if [ "$SRW_SR_DEP_VERSION" != "$SR_VERSION" ]; then
  fail "symbolic_regression_wasm -> symbolic_regression version ($SRW_SR_DEP_VERSION) must match symbolic_regression version ($SR_VERSION)"
fi
if [ "$SRW_SR_DEP_PATH" != "../../symbolic_regression" ]; then
  fail "symbolic_regression_wasm -> symbolic_regression path must be ../../symbolic_regression (got $SRW_SR_DEP_PATH)"
fi

read -r SRW_DE_DEP_VERSION SRW_DE_DEP_PATH < <(toml_get_dep_version_and_path web/wasm/Cargo.toml dynamic_expressions)
if [ "$SRW_DE_DEP_VERSION" != "$DE_VERSION" ]; then
  fail "symbolic_regression_wasm -> dynamic_expressions version ($SRW_DE_DEP_VERSION) must match dynamic_expressions version ($DE_VERSION)"
fi
if [ "$SRW_DE_DEP_PATH" != "../../dynamic_expressions" ]; then
  fail "symbolic_regression_wasm -> dynamic_expressions path must be ../../dynamic_expressions (got $SRW_DE_DEP_PATH)"
fi

python3 - <<'PY'
import json

manifest = json.load(open(".release-please-manifest.json", "r", encoding="utf-8"))
expected_keys = {"dynamic_expressions", "symbolic_regression", "web/wasm"}
missing = expected_keys - set(manifest)
extra = set(manifest) - expected_keys
if missing:
    raise SystemExit(f"manifest missing keys: {sorted(missing)}")
if extra:
    raise SystemExit(f"manifest has unexpected keys: {sorted(extra)}")
PY

echo "✅ Workspace dependency versions are consistent"
