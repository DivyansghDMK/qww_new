param(
  [string]$Name = "ECGMonitor",
  [string]$Version = "",
  [switch]$ConsoleBuild,
  [switch]$OneFile
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Version)) {
  $Version = Get-Date -Format "yyyy.MM.dd.HHmm"
}

Write-Host "=== ECG Release Build ===" -ForegroundColor Cyan
Write-Host "Name    : $Name"
Write-Host "Version : $Version"

Write-Host "`n[1/2] Building EXE..." -ForegroundColor Yellow
$buildArgs = @("build_exe.py", "--name", $Name)
if ($ConsoleBuild) { $buildArgs += "--console" }
if ($OneFile) { $buildArgs += "--onefile" }
python @buildArgs
if ($LASTEXITCODE -ne 0) { throw "EXE build failed" }

Write-Host "`n[2/2] Building setup.exe..." -ForegroundColor Yellow
python build_setup.py --name $Name --version $Version
if ($LASTEXITCODE -ne 0) {
  Write-Host "Setup build could not be completed (likely ISCC missing)." -ForegroundColor Red
  Write-Host "Install Inno Setup 6, then rerun: python build_setup.py --name $Name --version $Version"
  exit $LASTEXITCODE
}

Write-Host "`nRelease build complete." -ForegroundColor Green
Write-Host "Installer output: dist\installers"
