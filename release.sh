#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: ./release.sh [OPTIONS]

Build and publish the source distribution plus repaired Linux wheels.

Options:
  --repository NAME   Twine repository name from ~/.pypirc (default: pypi)
  --test-pypi        Shortcut for --repository testpypi
  --skip-upload      Build and check artifacts, but do not upload them
  --no-clean         Keep existing dist/ and wheelhouse/ contents
  --allow-dirty      Allow publishing from a dirty git working tree
  -h, --help         Show this help

Environment:
  PYTHON             Python executable used for build, cibuildwheel, and twine
  DIST_DIR           Source distribution output directory (default: dist)
  WHEELHOUSE         Wheel output directory (default: wheelhouse)
  CIBW_*             Passed through to cibuildwheel
  CIBW_CONTAINER_ENGINE
                     Container engine for Linux wheels (default: podman)

Common setup:
  python -m pip install -e '.[release]'
  export TWINE_USERNAME=__token__
  export TWINE_PASSWORD=pypi-...
EOF
}

die() {
  printf 'release.sh: %s\n' "$*" >&2
  exit 1
}

repository="pypi"
upload=1
clean=1
allow_dirty=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repository)
      [[ $# -ge 2 ]] || die "--repository requires a value"
      repository="$2"
      shift 2
      ;;
    --test-pypi)
      repository="testpypi"
      shift
      ;;
    --skip-upload)
      upload=0
      shift
      ;;
    --no-clean)
      clean=0
      shift
      ;;
    --allow-dirty)
      allow_dirty=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

python_bin="${PYTHON:-python}"
dist_dir="${DIST_DIR:-dist}"
wheelhouse="${WHEELHOUSE:-wheelhouse}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$project_root"

if [[ "$allow_dirty" -eq 0 ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ -n "$(git status --porcelain)" ]]; then
    git status --short
    die "working tree is dirty; commit/stash changes or pass --allow-dirty"
  fi
fi

"$python_bin" -c "import build" >/dev/null 2>&1 || die "missing Python module 'build'; run: python -m pip install -e '.[release]'"
"$python_bin" -c "import cibuildwheel" >/dev/null 2>&1 || die "missing Python module 'cibuildwheel'; run: python -m pip install -e '.[release]'"
"$python_bin" -c "import twine" >/dev/null 2>&1 || die "missing Python module 'twine'; run: python -m pip install -e '.[release]'"

export CIBW_CONTAINER_ENGINE="${CIBW_CONTAINER_ENGINE:-podman}"
container_engine="${CIBW_CONTAINER_ENGINE%%;*}"
container_engine="${container_engine//[[:space:]]/}"
if ! command -v "$container_engine" >/dev/null 2>&1; then
  die "${container_engine} is required for Linux manylinux wheels; install it or set CIBW_CONTAINER_ENGINE"
fi

if [[ "$clean" -eq 1 ]]; then
  rm -rf "$dist_dir" "$wheelhouse"
fi
mkdir -p "$dist_dir" "$wheelhouse"

echo "Building source distribution..."
"$python_bin" -m build --sdist --outdir "$dist_dir"

echo "Building repaired Linux wheels with cibuildwheel using ${CIBW_CONTAINER_ENGINE}..."
"$python_bin" -m cibuildwheel --platform linux --output-dir "$wheelhouse"

shopt -s nullglob
artifacts=( "$dist_dir"/*.tar.gz "$wheelhouse"/*.whl )
shopt -u nullglob
[[ ${#artifacts[@]} -gt 0 ]] || die "no artifacts were produced"

echo "Checking artifacts..."
"$python_bin" -m twine check "${artifacts[@]}"

if [[ "$upload" -eq 0 ]]; then
  printf 'Built artifacts:\n'
  printf '  %s\n' "${artifacts[@]}"
  exit 0
fi

echo "Uploading artifacts to ${repository}..."
"$python_bin" -m twine upload --repository "$repository" "${artifacts[@]}"
