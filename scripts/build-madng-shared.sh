#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: build-madng-shared.sh [--clean] /path/to/MAD-NG/src

Build a loadable libmad.so exposing the MAD-NG descriptor/TPSA C API used by
madng-tpsa.  On Linux this intentionally links only the real/complex TPSA core
objects, plus the standalone MAD log helper, because the full libmad.a archive
may include non-PIC LuaJIT objects that cannot be embedded in a shared object.

Options:
  --clean  remove MAD-NG build artifacts before rebuilding libmad.a
EOF
}

clean=0
if [[ $# -gt 0 && "$1" == "--clean" ]]; then
  clean=1
  shift
fi

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

SRC_DIR=$(cd "$1" && pwd)
MAKEFILE=$SRC_DIR/Makefile.linux
OUT=$SRC_DIR/libmad.so
LOG_OBJ=$SRC_DIR/build/mad_log_standalone.o
CC=${CC:-gcc}

if [[ ! -f "$MAKEFILE" ]]; then
  echo "error: expected $MAKEFILE" >&2
  exit 1
fi

cd "$SRC_DIR"

if (( clean )); then
  make -f "$MAKEFILE" clean
fi

make -f "$MAKEFILE" libmad.a

"$CC" -std=c99 -fPIC -I"$SRC_DIR" -c "$SRC_DIR/libgtpsa/mad_log.c" -o "$LOG_OBJ"

"$CC" -shared -o "$OUT" \
  build/mad_bit.o \
  build/mad_cst.o \
  build/mad_ctpsa.o \
  build/mad_ctpsa_comp.o \
  build/mad_ctpsa_conv.o \
  build/mad_ctpsa_fun.o \
  build/mad_ctpsa_io.o \
  build/mad_ctpsa_minv.o \
  build/mad_ctpsa_mops.o \
  build/mad_ctpsa_ops.o \
  build/mad_desc.o \
  build/mad_erfw.o \
  build/mad_mat.o \
  build/mad_mem.o \
  build/mad_mono.o \
  build/mad_num.o \
  build/mad_poly.o \
  build/mad_rad.o \
  build/mad_str.o \
  build/mad_tpsa.o \
  build/mad_tpsa_comp.o \
  build/mad_tpsa_fun.o \
  build/mad_tpsa_io.o \
  build/mad_tpsa_minv.o \
  build/mad_tpsa_mops.o \
  build/mad_tpsa_ops.o \
  build/mad_vec.o \
  "$LOG_OBJ" \
  ../bin/linux/liblapack.a \
  ../bin/linux/librefblas.a \
  -lm -ldl -lgomp -lstdc++ -lgfortran -lquadmath

required_symbols=(
  mad_desc_newv
  mad_desc_getnv
  mad_tpsa_newd
  mad_tpsa_setvar
  mad_tpsa_add
)

if command -v nm >/dev/null 2>&1; then
  defined_symbols=$(nm -D --defined-only "$OUT")
  for symbol in "${required_symbols[@]}"; do
    if ! grep -q "[[:space:]]$symbol$" <<<"$defined_symbols"; then
      echo "error: $OUT does not export $symbol" >&2
      exit 1
    fi
  done
fi

cat <<MSG
Built: $OUT
Use it with:
  export MADNG_TPSA_LIBRARY=$OUT
MSG
