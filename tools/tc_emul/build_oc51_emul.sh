#!/bin/sh
# Build + run the OC51 T2 emulator (CPU, no nvcc; uses REAL reduce_sum<1>).
# -O0 is REQUIRED: the per-warp acc harness array is ~256 KB on the stack and
# MinGW -O2 miscompiles it (heap corruption artifact; O0 is clean and fast).
R=$(cd "$(dirname "$0")/../.." && pwd)
W="$R/build-oc51-emul"
mkdir -p "$W"
for f in gpu31.cuh tc.cuh fherma.h; do
  python -c "
import re,sys
s=open(sys.argv[1],encoding='utf-8').read()
s=re.sub(r'asm\s+volatile\s*\((?:[^()]|\([^()]*\))*\)\s*;','/*asm*/;',s)
s=re.sub(r'asm\s*\((?:[^()]|\([^()]*\))*\)\s*;','/*asm*/;',s)
s=re.sub(r'<<<.*?>>>','',s,flags=re.S)
open(sys.argv[2],'w',encoding='utf-8').write(s)" "$R/$f" "$W/$f"
done
cp "$R/tools/tc_emul/oc51_emul.cpp" "$W/"
"$W/oc51_emul.exe" 2>/dev/null || true   # stale exe guard (Windows: not found is ok)
GXX=$(command -v g++ || command -v g++.exe)
"$GXX" -std=c++17 -O0 -w -D__CUDACC__ -D__CUDA_ARCH__=1200 \
  -include "$R/tools/nvsyntax/extra.keep" \
  -I "$R/tools/nvsyntax/inc" -I "$W" \
  "$(cygpath -w "$W/oc51_emul.cpp" 2>/dev/null || echo "$W/oc51_emul.cpp")" \
  -o "$(cygpath -w "$W/oc51_emul.exe" 2>/dev/null || echo "$W/oc51_emul.exe")" || exit 1
"$W/oc51_emul.exe"