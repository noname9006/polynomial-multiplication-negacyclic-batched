#!/bin/sh
# Build + run the OC41 R8F 64-row emulator (CPU, no nvcc needed).
# Uses the REAL gpu31::reduce_sum<1> and tc::bake_crt from the CUDA sources,
# neutralized the same way tools/nvsyntax/check.sh does it.
# Usage: sh tools/tc_emul/build_oc41_emul.sh   (from the repo root)
R=$(cd "$(dirname "$0")/../.." && pwd)
W="$R/build-oc41-emul"
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
cp "$R/tools/tc_emul/oc41_r64f_emul.cpp" "$W/"
g++ -std=c++17 -O2 -w -D__CUDACC__ -D__CUDA_ARCH__=1200 \
  -include "$R/tools/nvsyntax/extra.keep" \
  -I "$R/tools/nvsyntax/inc" -I "$W" \
  "$W/oc41_r64f_emul.cpp" -o "$W/oc41_emul" || exit 1
"$W/oc41_emul"