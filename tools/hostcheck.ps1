# Type-checks solve.cu — its host half and, through the shim, everything it calls
# on the device — with an ordinary C++ compiler.
#
#   ./tools/hostcheck.ps1
#
# This is not a substitute for nvcc: it cannot check the launch configuration,
# the register budget or the generated code. It checks the thing that a platform
# round trip would otherwise be the first to discover: that every name, type and
# call in solve.cu is right. tools/gpucheck.cpp then checks the arithmetic.
param(
    [string]$Compiler = "g++"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$log = Join-Path $root "build-cpu/hostcheck.log"
$err = Join-Path $root "build-cpu/hostcheck.err"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

# Start-Process rather than a plain call: PowerShell 5.1 turns a native program's
# stderr into a terminating error when the preference is Stop, which would hide
# the compiler's own message behind a stack trace.
$arguments = @(
    "-std=c++17", "-fsyntax-only", "-Wall", "-Wextra",
    "-D__CUDACC__", "-DPGPU_HOST_CHECK",
    "-include", (Join-Path $root "tools/shim/cuda_shim.hpp"),
    "-I", (Join-Path $root "tools/fakecuda"),
    "-x", "c++", (Join-Path $root "solve.cu")
)
$proc = Start-Process -FilePath $Compiler -ArgumentList $arguments -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $err
if ($proc.ExitCode -ne 0) {
    Get-Content $err | ForEach-Object { Write-Host $_ }
    Get-Content $log | ForEach-Object { Write-Host $_ }
    throw "host check failed"
}
Get-Content $err | Where-Object { $_ -match 'warning' } | ForEach-Object { Write-Host $_ }
Write-Host "solve.cu type-checks" -ForegroundColor Green
