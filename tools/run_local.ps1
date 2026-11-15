# Builds the portable core with g++ and runs the whole local loop:
#   bundle.py make  ->  ./build-cpu/solution <dir>  ->  bundle.py verify
#
# The same sources build with nvcc through the generated CMakeLists, which is
# what the platform does. This script never touches the CUDA path.
param(
    [string]$Point = '{"B":2,"N":1024,"W":128}',
    [int]$Cases = 3,
    [string]$Work = "work/local",
    [string]$Python = "python",
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot        # the implementation directory (repo root)
$root = $proj
$exe = Join-Path $proj "build-cpu\solution.exe"

if (-not $NoBuild) {
    New-Item -ItemType Directory -Force -Path (Split-Path $exe) | Out-Null
    Write-Host "building $exe" -ForegroundColor Cyan
    g++ -O3 -std=c++17 -march=native -static-libgcc -static-libstdc++ `
        -o $exe (Join-Path $proj "main.cpp") -x c++ (Join-Path $proj "solve.cu")
    if ($LASTEXITCODE -ne 0) { throw "build failed" }
}

$dir = Join-Path $root $Work
& $Python (Join-Path $PSScriptRoot "bundle.py") make $dir --point $Point --cases $Cases
if ($LASTEXITCODE -ne 0) { throw "make failed" }

& $exe $dir
if ($LASTEXITCODE -ne 0) { throw "solution exited $LASTEXITCODE" }

& $Python (Join-Path $PSScriptRoot "bundle.py") verify $dir
exit $LASTEXITCODE
