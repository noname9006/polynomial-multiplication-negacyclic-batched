#!/bin/sh
# g++ syntax check of the CUDA sources (no nvcc here). Strips <<<...>>> launch
# configs, fakes the CUDA runtime and intrinsics. Usage: sh tools/nvsyntax/check.sh
H=$(cd "$(dirname "$0")" && pwd); R=$(cd "$H/../.." && pwd); W=$R/build-cpu/nvsyntax
rm -rf "$W"; mkdir -p "$W"
for f in "$R"/*.cu "$R"/*.cuh "$R"/*.hpp "$R"/*.h; do
  python -c "
import re,sys
s=open(sys.argv[1],encoding='utf-8').read()
open(sys.argv[2],'w',encoding='utf-8').write(re.sub(r'<<<.*?>>>','',s,flags=re.S))" "$f" "$W/$(basename "$f")"
done
g++ -std=c++17 -fsyntax-only -w -D__CUDACC__ -D__CUDA_ARCH__=1200 -include "$H/extra.keep" -I "$H/inc" -I "$W" -x c++ "$W/solve.cu" 2>&1 | grep -E "error" | head -30
echo "nvsyntax check done"
