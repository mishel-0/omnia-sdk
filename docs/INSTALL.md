# Installing omnia-sdk

Step-by-step for macOS, Linux and Windows. The one-line installer is the easiest route; a manual
`pip` route follows for people who manage their own environments.

What you get: the `omnia` command with every format reader (whole-slide images, DICOM, DICOM
whole-slide, NIfTI). PyTorch is optional and only needed to train from `.omnia` or to run benchmarks.

- [macOS](#macos)
- [Linux](#linux)
- [Windows](#windows)
- [Install with PyTorch](#install-with-pytorch)
- [Manual install with pip](#manual-install-with-pip)
- [Update](#update) · [Uninstall](#uninstall) · [Troubleshooting](#troubleshooting)

---

## macOS

Works on Apple Silicon and Intel.

**1. Open Terminal** (Applications → Utilities → Terminal).

**2. Check for Python 3.11 or newer.**

```bash
python3 --version
```

If it prints `Python 3.11` or higher, go to step 3. Otherwise install Python with one of these:

```bash
# Option A: Homebrew (https://brew.sh)
brew install python@3.12
```

Option B: download the macOS installer from <https://www.python.org/downloads/macos/>, run it, then open a new Terminal window.

**3. Run the installer.**

```bash
curl -fsSL https://raw.githubusercontent.com/mishel-0/omnia-sdk/main/install.sh | sh
```

**4. Reload your shell** (or open a new Terminal window).

```bash
source ~/.zshrc
```

**5. Check it works.**

```bash
omnia doctor
```

**6. Convert your first file.**

```bash
omnia convert ~/Downloads/slide.svs
```

---

## Linux

**1. Install Python 3.11+, venv and curl.**

Ubuntu 24.04 / Debian 12 and newer:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip curl
```

Ubuntu 22.04 (ships Python 3.10, so add a newer one):

```bash
sudo apt update
sudo apt install -y software-properties-common curl
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.12 python3.12-venv
```

Fedora / RHEL:

```bash
sudo dnf install -y python3.12 curl
```

Arch:

```bash
sudo pacman -S --needed python curl
```

**2. Check the version.**

```bash
python3 --version        # or: python3.12 --version
```

**3. Run the installer.**

```bash
curl -fsSL https://raw.githubusercontent.com/mishel-0/omnia-sdk/main/install.sh | sh
```

**4. Reload your shell** (or open a new terminal).

```bash
source ~/.bashrc
```

**5. Check it works.**

```bash
omnia doctor
```

**6. Convert your first file.**

```bash
omnia convert ~/data/slide.svs
```

---

## Windows

Windows 10 or 11. Use **PowerShell** (Start → type "PowerShell"); you do not need to run it as administrator.

**1. Check for Python 3.11 or newer.**

```powershell
py --version
```

If that fails or prints a version below 3.11, install Python with one of these:

```powershell
# Option A: winget (built into Windows 10/11)
winget install -e --id Python.Python.3.12
```

Option B: download the installer from <https://www.python.org/downloads/windows/>. On the first screen,
tick **"Add python.exe to PATH"**, then click **Install Now**.

Close PowerShell and open a new window after installing Python.

**2. Run the installer.**

```powershell
irm https://raw.githubusercontent.com/mishel-0/omnia-sdk/main/install.ps1 | iex
```

**3. Close PowerShell and open a new window** so the updated PATH is picked up.

**4. Check it works.**

```powershell
omnia doctor
```

**5. Convert your first file.**

```powershell
omnia convert "$env:USERPROFILE\Downloads\slide.svs"
```

---

## Install with PyTorch

Needed for `omnia_sdk.OmniaDataset`, `omnia benchmark` and `omnia recommend`. Adds roughly 1–2 GB.

macOS / Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/mishel-0/omnia-sdk/main/install.sh | OMNIA_WITH_TORCH=1 sh
```

Windows (PowerShell):

```powershell
$env:OMNIA_WITH_TORCH = "1"
irm https://raw.githubusercontent.com/mishel-0/omnia-sdk/main/install.ps1 | iex
```

For an NVIDIA GPU, install the CUDA build of PyTorch afterwards by following the selector at
<https://pytorch.org/get-started/locally/>, using the Python inside `~/.omnia/venv`
(Windows: `%USERPROFILE%\.omnia\venv\Scripts\python.exe`).

---

## Manual install with pip

Use this if you want omnia-sdk inside your own virtual environment or conda env.

macOS / Linux:

```bash
python3 -m venv ~/omnia-env
source ~/omnia-env/bin/activate
pip install --upgrade pip
pip install "omnia-sdk[all] @ https://github.com/mishel-0/omnia-sdk/archive/refs/heads/main.zip"
omnia doctor
```

Windows (PowerShell):

```powershell
py -3.12 -m venv $env:USERPROFILE\omnia-env
& $env:USERPROFILE\omnia-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install "omnia-sdk[all] @ https://github.com/mishel-0/omnia-sdk/archive/refs/heads/main.zip"
omnia doctor
```

If `Activate.ps1` is blocked, allow local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Extras you can combine (comma-separated inside the brackets):

| Extra | Adds |
|---|---|
| `svs` | OpenSlide (SVS, NDPI, MRXS, SCN, BIF, TIFF) |
| `dicom` | pydicom with JPEG / JPEG 2000 decoders |
| `dicom-wsi` | DICOM whole-slide (wsidicom) |
| `nifti` | nibabel |
| `all` | every reader above |
| `train` | PyTorch, for `OmniaDataset` |
| `benchmark` | PyTorch and torchvision, for `omnia benchmark` / `omnia recommend` |

Example: `pip install "omnia-sdk[all,train] @ https://github.com/mishel-0/omnia-sdk/archive/refs/heads/main.zip"`

---

## Update

Run the same installer command again. It upgrades in place.

With pip: rerun the `pip install ...` line with `--upgrade`.

---

## Uninstall

macOS / Linux:

```bash
rm -rf ~/.omnia ~/.local/bin/omnia
```

Then delete the two lines marked `# omnia-sdk: PATH` from `~/.zshrc`, `~/.bashrc` or `~/.bash_profile`.

Windows (PowerShell):

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.omnia"
$p = [Environment]::GetEnvironmentVariable("Path", "User") -split ";" | Where-Object { $_ -notlike "*\.omnia\venv\Scripts" }
[Environment]::SetEnvironmentVariable("Path", ($p -join ";"), "User")
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `omnia: command not found` (macOS/Linux) | Open a new terminal, or run `export PATH="$HOME/.local/bin:$PATH"`. |
| `'omnia' is not recognized` (Windows) | Close and reopen PowerShell. Still failing: run `%USERPROFILE%\.omnia\venv\Scripts\omnia.exe doctor` directly. |
| `Python 3.11+ not found` | Install Python as in step 1 for your system, open a new terminal, run the installer again. |
| `could not create a virtual environment` (Linux) | `sudo apt install python3-venv` (or `python3.12-venv`). |
| A format shows `--` in `omnia doctor` | `omnia doctor --fix` |
| DICOM error `cannot decode pixel data` | `omnia doctor --fix` installs the JPEG / JPEG 2000 decoders. |
| `pip install failed` behind a company proxy | Set `HTTPS_PROXY=http://proxy:port` before running the installer. |
| Anything else | Open an issue with the output of `omnia doctor`: <https://github.com/mishel-0/omnia-sdk/issues> |
