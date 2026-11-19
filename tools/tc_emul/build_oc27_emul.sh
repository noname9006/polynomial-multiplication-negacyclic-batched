#!/bin/sh
# Build + run the OC27 r64 emulator (CPU, no nvcc needed).
# The device sections of tc.cuh/gpu31.cuh are neutralized the same way
# tools/nvsyntax/check.sh does it (strip <<<...>>>, fake CUDA via
# tools/nvsyntax/extra.keep + inc/cuda_runtime.h); inline PTX asm bodies
# are stripped too so g++ can compile the kernels as dead host code.
# The REAL gpu31::reduce_sum<1> is compiled and used by the emulator.
# Usage: sh tools/tc_emul/build_oc27_emul.sh   (from the repo root)
R=$(cd "$(dirname "$0")/../.." && pwd)
W="$R/build-emul"
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
cp "$R/tools/tc_emul/oc27_r64_emul.cpp" "$W/"
g++ -std=c++17 -O2 -w -D__CUDACC__ -D__CUDA_ARCH__=1200 \
  -include "$R/tools/nvsyntax/extra.keep" \
  -I "$R/tools/nvsyntax/inc" -I "$W" \
  "$W/oc27_r64_emul.cpp" -o "$W/oc27_emul" || exit 1
"$W/oc27_emul"