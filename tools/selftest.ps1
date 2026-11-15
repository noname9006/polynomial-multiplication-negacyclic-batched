# Builds once, then runs the full local loop over a matrix of points.
# Everything here is the portable core; no GPU involved.
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot        # the implementation directory (repo root)
$root = $proj
$exe = Join-Path $proj "build-cpu\solution.exe"
$bundle = Join-Path $PSScriptRoot "bundle.py"

New-Item -ItemType Directory -Force -Path (Split-Path $exe) | Out-Null
Write-Host "building $exe" -ForegroundColor Cyan
g++ -O3 -std=c++17 -march=native -static-libgcc -static-libstdc++ `
    -o $exe (Join-Path $proj "main.cpp") -x c++ (Join-Path $proj "solve.cu")
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$points = @(
    @{ p = '{"B":1,"N":1,"W":2}';       c = 1; name = "degenerate N=1" },
    @{ p = '{"B":2,"N":2,"W":3}';       c = 1; name = "degenerate N=2" },
    @{ p = '{"B":1,"N":64,"W":32}';    c = 3; name = "tiny" },
    @{ p = '{"B":3,"N":128,"W":64}';   c = 2; name = "batch of 3" },
    @{ p = '{"B":2,"N":1024,"W":128}'; c = 2; name = "128-bit q" },
    @{ p = '{"B":1,"N":512,"W":254}';  c = 2; name = "W not a multiple of 32" },
    @{ p = '{"B":1,"N":256,"W":868}';  c = 2; name = "challenge point #2 width" },
    @{ p = '{"B":1,"N":65536,"W":1747}'; c = 1; name = "the ranked point, one pair" }
)

$failed = 0
foreach ($pt in $points) {
    Write-Host "`n=== $($pt.name)  $($pt.p)" -ForegroundColor Yellow
    $dir = Join-Path $root ("work/selftest-" + ($pt.name -replace '[^a-zA-Z0-9]', '-'))
    $pf = Join-Path $root "work/point.json"
    New-Item -ItemType Directory -Force -Path (Split-Path $pf) | Out-Null
    Set-Content -Path $pf -Value $pt.p -Encoding ascii
    & $Python $bundle make $dir --point-file $pf --cases $pt.c | Out-Null
    & $exe $dir
    if ($LASTEXITCODE -ne 0) { $failed++; Write-Host "solution failed" -ForegroundColor Red; continue }
    & $Python $bundle verify $dir
    if ($LASTEXITCODE -ne 0) { $failed++ }
    $res = Get-Content (Join-Path $dir "out/results.json") | ConvertFrom-Json
    $secs = ($res.cases | ForEach-Object { $_.seconds }) -join ", "
    Write-Host ("  run: {0} s   init {1:N3} s" -f $secs, $res.init_s)
}

Write-Host ""
if ($failed) { Write-Host "$failed point(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "all points exact" -ForegroundColor Green
