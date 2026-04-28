Param(
  [string]$InnoSetupISCC = "",
  [string]$IssPath = (Join-Path $PSScriptRoot "ECGMonitor.iss")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $IssPath)) {
  throw "Inno Setup script not found: $IssPath"
}

function Resolve-ISCC {
  param([string]$Explicit)

  if ($Explicit -and (Test-Path $Explicit)) { return (Resolve-Path $Explicit).Path }

  $candidates = @(
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 5\ISCC.exe",
    "$env:ProgramFiles(x86)\Inno Setup 5\ISCC.exe"
  )

  foreach ($c in $candidates) {
    if (Test-Path $c) { return (Resolve-Path $c).Path }
  }

  $cmd = Get-Command iscc.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }

  return $null
}

$iscc = Resolve-ISCC -Explicit $InnoSetupISCC
if (-not $iscc) {
  throw "Could not find ISCC.exe. Install Inno Setup, or pass -InnoSetupISCC 'C:\Path\To\ISCC.exe'."
}

Write-Host "Using ISCC: $iscc"
Write-Host "Compiling : $IssPath"

& $iscc $IssPath | Write-Host

Write-Host "Done. Installer output is in: $(Resolve-Path (Join-Path $PSScriptRoot '..\dist_installer')).Path"

