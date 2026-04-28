param(
  [string]$Name = "ECGMonitor",
  [string]$Version = "",
  [string]$Channel = "stable",
  [string]$Repository = "",
  [switch]$ConsoleBuild,
  [switch]$OneFile
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Version)) {
  $Version = Get-Date -Format "yyyy.MM.dd.HHmm"
}

if ([string]::IsNullOrWhiteSpace($Repository)) {
  try {
    $remote = git remote get-url origin 2>$null
    if ($remote -match 'github\.com[:/](?<owner>[^/]+)/(?<repo>[^/.]+)') {
      $Repository = "$($Matches.owner)/$($Matches.repo)"
    }
  } catch {
    $Repository = ""
  }
}

$versionFile = Join-Path $PSScriptRoot "src\version.py"
$versionContent = @"
"""Build-time version metadata for ECG Monitor."""

APP_VERSION = "$Version"
UPDATE_CHANNEL = "$Channel"
GITHUB_REPOSITORY = "$Repository"
"@
Set-Content -Path $versionFile -Value $versionContent -Encoding utf8

Write-Host "=== ECG Release Build ===" -ForegroundColor Cyan
Write-Host "Name    : $Name"
Write-Host "Version : $Version"
Write-Host "Channel : $Channel"
Write-Host "Repo    : $Repository"
Write-Host "Tag     : $Channel-$Version"

Write-Host "`n[1/2] Building EXE..." -ForegroundColor Yellow
$buildArgs = @("build_exe.py", "--name", $Name)
if ($ConsoleBuild) { $buildArgs += "--console" }
if ($OneFile) { $buildArgs += "--onefile" }
python @buildArgs
if ($LASTEXITCODE -ne 0) { throw "EXE build failed" }

Write-Host "`n[2/2] Building setup.exe..." -ForegroundColor Yellow
python build_setup.py --name $Name --version $Version --channel $Channel --repository $Repository
if ($LASTEXITCODE -ne 0) {
  Write-Host "Setup build could not be completed (likely ISCC missing)." -ForegroundColor Red
  Write-Host "Install Inno Setup 6, then rerun: python build_setup.py --name $Name --version $Version"
  exit $LASTEXITCODE
}

Write-Host "`nRelease build complete." -ForegroundColor Green
Write-Host "Installer output: dist\installers"
