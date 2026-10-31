#!/bin/sh
# omnia-sdk installer for macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/mishel-0/omnia-sdk/main/install.sh | sh
#
# Creates a private environment in ~/.omnia, installs omnia-sdk with every format
# reader (no admin rights, nothing touches your system Python), and puts the
# `omnia` command on your PATH. Re-run it to update.
#
# Options (environment variables):
#   OMNIA_WITH_TORCH=1   also install PyTorch (training dataset, benchmarks)
#   OMNIA_HOME=<dir>     install somewhere other than ~/.omnia
set -eu

OMNIA_HOME="${OMNIA_HOME:-$HOME/.omnia}"
SRC="${OMNIA_SRC:-https://github.com/mishel-0/omnia-sdk/archive/refs/heads/main.zip}"
EXTRAS="all"
[ "${OMNIA_WITH_TORCH:-0}" = "1" ] && EXTRAS="all,train,benchmark"

say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# 1. Find Python 3.11 or newer.
PY=""
for c in python3.13 python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  case "$(uname -s)" in
    Darwin) die "Python 3.11+ not found. Install it from https://www.python.org/downloads/ (or: brew install python@3.12), then re-run." ;;
    *)      die "Python 3.11+ not found. Install it with your package manager (e.g. sudo apt install python3.12 python3.12-venv), then re-run." ;;
  esac
fi
say "Using $($PY --version) ($(command -v "$PY"))"

# 2. Private environment.
if [ ! -x "$OMNIA_HOME/venv/bin/python" ]; then
  say "Creating environment in $OMNIA_HOME"
  "$PY" -m venv "$OMNIA_HOME/venv" || die "could not create a virtual environment (on Debian/Ubuntu: sudo apt install python3-venv)"
fi
VPY="$OMNIA_HOME/venv/bin/python"
"$VPY" -m pip install --quiet --upgrade pip

# 3. Install.
say "Installing omnia-sdk [$EXTRAS] (this can take a minute)"
"$VPY" -m pip install --quiet --upgrade "omnia-sdk[$EXTRAS] @ $SRC" || die "pip install failed (see above)"

# 4. Put `omnia` on PATH.
BIN="$HOME/.local/bin"
mkdir -p "$BIN"
ln -sf "$OMNIA_HOME/venv/bin/omnia" "$BIN/omnia"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *)
    case "$(basename "${SHELL:-sh}")" in
      zsh)  RC="$HOME/.zshrc" ;;
      bash) if [ "$(uname -s)" = Darwin ]; then RC="$HOME/.bash_profile"; else RC="$HOME/.bashrc"; fi ;;
      *)    RC="$HOME/.profile" ;;
    esac
    grep -qs 'omnia-sdk: PATH' "$RC" || printf '\n# omnia-sdk: PATH\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$RC"
    say "Added ~/.local/bin to your PATH. Open a new terminal (or run: export PATH=\"\$HOME/.local/bin:\$PATH\")."
    ;;
esac

# 5. Show what works.
echo
"$OMNIA_HOME/venv/bin/omnia" doctor || true
echo
say "Installed. Convert a file with:  omnia convert <file or folder>"
