# omnia-sdk installer for Windows (PowerShell).
#
#   irm https://raw.githubusercontent.com/mishel-0/omnia-sdk/main/install.ps1 | iex
#
# Creates a private environment in %USERPROFILE%\.omnia, installs omnia-sdk with every
# format reader (no admin rights), and adds the `omnia` command to your user PATH.
# Re-run it to update. Set $env:OMNIA_WITH_TORCH = "1" first to also install PyTorch.

$ErrorActionPreference = "Stop"
$OmniaHome = if ($env:OMNIA_HOME) { $env:OMNIA_HOME } else { Join-Path $env:USERPROFILE ".omnia" }
$Src = if ($env:OMNIA_SRC) { $env:OMNIA_SRC } else { "https://github.com/mishel-0/omnia-sdk/archive/refs/heads/main.zip" }
$Extras = if ($env:OMNIA_WITH_TORCH -eq "1") { "all,train,benchmark" } else { "all" }

function Say($m) { Write-Host $m -ForegroundColor Cyan }

# 1. Find Python 3.11 or newer (the "py" launcher first, then python on PATH).
$Py = $null
foreach ($cand in @(@("py", "-3.13"), @("py", "-3.12"), @("py", "-3.11"), @("python"))) {
    try {
        $exe = $cand[0]; $rest = @($cand | Select-Object -Skip 1)
        & $exe @rest -c "import sys; sys.exit(sys.version_info < (3, 11))" 2>$null
        if ($LASTEXITCODE -eq 0) { $Py = $cand; break }
    } catch { }
}
if (-not $Py) {
    Write-Host "Python 3.11+ not found. Install it from https://www.python.org/downloads/windows/" -ForegroundColor Red
    Write-Host "(tick 'Add python.exe to PATH' in the installer), then run this command again." -ForegroundColor Red
    exit 1
}
$PyExe = $Py[0]; $PyArgs = @($Py | Select-Object -Skip 1)
Say ("Using " + (& $PyExe @PyArgs --version))

# 2. Private environment.
$VPy = Join-Path $OmniaHome "venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) {
    Say "Creating environment in $OmniaHome"
    & $PyExe @PyArgs -m venv (Join-Path $OmniaHome "venv")
}
& $VPy -m pip install --quiet --upgrade pip

# 3. Install.
Say "Installing omnia-sdk [$Extras] (this can take a minute)"
& $VPy -m pip install --quiet --upgrade "omnia-sdk[$Extras] @ $Src"
if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed (see above)." -ForegroundColor Red; exit 1 }

# 4. Add the Scripts folder to the user PATH.
$Scripts = Join-Path $OmniaHome "venv\Scripts"
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not ($UserPath -split ";" | Where-Object { $_ -eq $Scripts })) {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$Scripts", "User")
    Say "Added $Scripts to your PATH. Open a new terminal to use 'omnia'."
}
$env:Path = "$env:Path;$Scripts"

# 5. Show what works.
Write-Host ""
& (Join-Path $Scripts "omnia.exe") doctor
Write-Host ""
Say "Installed. Convert a file with:  omnia convert <file or folder>"
