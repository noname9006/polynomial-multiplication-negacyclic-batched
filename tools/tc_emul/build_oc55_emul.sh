#!/bin/sh
# Build + run the OC55 W8 emulator (CPU, no nvcc; uses REAL reduce_sum<1>).
# -O0 required (large stack acc arrays; MinGW -O2 miscompiles them).
R=$(cd "$(dirname "$0")/../.." && pwd)
W="$R/build-oc55-emul"
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
cp "$R/tools/tc_emul/oc55_emul.cpp" "$W/"
GXX=$(command -v g++ || command -v g++.exe)
"$GXX" -std=c++17 -O0 -w -D__CUDACC__ -D__CUDA_ARCH__=1200 \
  -include "$R/tools/nvsyntax/extra.keep" \
  -I "$R/tools/nvsyntax/inc" -I "$W" \
  "$(cygpath -w "$W/oc55_emul.cpp" 2>/dev/null || echo "$W/oc55_emul.cpp")" \
  -o "$(cygpath -w "$W/oc55_emul.exe" 2>/dev/null || echo "$W/oc55_emul.exe")" || exit 1
"$W/oc55_emul.exe"