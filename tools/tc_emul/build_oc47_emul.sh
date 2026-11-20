#!/bin/sh
# Build + run the OC47 emulator (CPU, no nvcc). Uses the real
# gpu31::reduce_sum<1> from the CUDA sources, neutralized like nvsyntax.
R=$(cd "$(dirname "$0")/../.." && pwd)
W="$R/build-oc47-emul"
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
cp "$R/tools/tc_emul/oc47_emul.cpp" "$W/"
g++ -std=c++17 -O2 -w -D__CUDACC__ -D__CUDA_ARCH__=1200 \
  -include "$R/tools/nvsyntax/extra.keep" \
  -I "$R/tools/nvsyntax/inc" -I "$W" \
  "$W/oc47_emul.cpp" -o "$W/oc47_emul" || exit 1
"$W/oc47_emul"
