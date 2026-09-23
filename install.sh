#!/bin/sh
#
# OllamaDiffuser one-line installer.
#
#   curl -fsSL https://raw.githubusercontent.com/LocalKinAI/ollamadiffuser/main/install.sh | sh
#
# What it does — and does NOT need your system Python:
#   1. installs `uv` (a single static binary) if you don't have it
#   2. uses uv to fetch a standalone Python 3.12 (does not touch your system Python)
#   3. creates an isolated environment at ~/.ollamadiffuser/venv
#   4. installs the compile-free core (prebuilt wheels only — no CMake/CUDA)
#   5. on Apple Silicon, also enables the native MLX backend (still compile-free)
#   6. drops an `ollamadiffuser` launcher on your PATH
#
# Optional backends stay opt-in afterward:
#   ollamadiffuser enable gguf   # low-VRAM quantized models (compiles once)
#   ollamadiffuser enable mcp    # Model Context Protocol server
#
# Uninstall:  rm -rf ~/.ollamadiffuser  (and remove the launcher printed below)
#
set -eu

ODIR="${OLLAMADIFFUSER_HOME:-$HOME/.ollamadiffuser}"
VENV="$ODIR/venv"
BIN="$ODIR/bin"
PYVER="3.12"

say()  { printf '%s\n' "$*"; }
step() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warning:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m error:\033[0m %s\n' "$*" >&2; exit 1; }

say "🎨 OllamaDiffuser installer"
say "==========================="

# --- 1. Ensure uv is available -------------------------------------------
ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    # uv may already be installed but not yet on PATH in this shell.
    for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        if [ -x "$d/uv" ]; then
            PATH="$d:$PATH"
            return 0
        fi
    done
    step "Installing uv (single static binary; no Python needed)..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "Need curl or wget to install uv."
    fi
    # uv's installer drops it in ~/.local/bin (recent) or ~/.cargo/bin.
    for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        [ -x "$d/uv" ] && PATH="$d:$PATH"
    done
    command -v uv >/dev/null 2>&1 || die "uv install did not land on PATH."
}
ensure_uv
step "uv: $(command -v uv)"

# --- 2. Standalone Python (does not touch system Python) -----------------
step "Fetching a standalone Python $PYVER (isolated)..."
uv python install "$PYVER"

# --- 3. Isolated environment (--seed gives pip so 'enable' works) --------
mkdir -p "$ODIR"
if [ -d "$VENV" ]; then
    step "Reusing existing environment at $VENV (upgrading)..."
else
    step "Creating isolated environment at $VENV..."
    uv venv --seed --python "$PYVER" "$VENV"
fi
VENV_PY="$VENV/bin/python"

# --- 4. Compile-free core -------------------------------------------------
step "Installing OllamaDiffuser (compile-free core)..."
uv pip install --python "$VENV_PY" --upgrade ollamadiffuser

# --- 5. Apple Silicon: native MLX backend (also compile-free) ------------
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    step "🍎 Apple Silicon — enabling MLX backend (typically 2-3× faster)..."
    uv pip install --python "$VENV_PY" "mflux>=0.20.0" \
        || warn "MLX install skipped; run 'ollamadiffuser enable mlx' later."
fi

# --- 6. Launcher on PATH --------------------------------------------------
mkdir -p "$BIN"
cat > "$BIN/ollamadiffuser" <<EOF
#!/bin/sh
exec "$VENV/bin/ollamadiffuser" "\$@"
EOF
chmod +x "$BIN/ollamadiffuser"

# Prefer symlinking into an existing on-PATH dir; else add to shell rc.
linked=""
if printf '%s' ":$PATH:" | grep -q ":$HOME/.local/bin:" && [ -d "$HOME/.local/bin" ]; then
    ln -sf "$BIN/ollamadiffuser" "$HOME/.local/bin/ollamadiffuser"
    linked="$HOME/.local/bin/ollamadiffuser"
fi

if [ -z "$linked" ]; then
    # Idempotently add our bin dir to the user's shell rc.
    case "${SHELL:-}" in
        */zsh) RC="$HOME/.zshrc" ;;
        */bash) RC="$HOME/.bashrc" ;;
        *) RC="$HOME/.profile" ;;
    esac
    MARK="# added by OllamaDiffuser installer"
    if [ ! -f "$RC" ] || ! grep -qF "$MARK" "$RC" 2>/dev/null; then
        {
            printf '\n%s\n' "$MARK"
            # shellcheck disable=SC2016  # $PATH must stay literal for the rc file
            printf 'export PATH="%s:$PATH"\n' "$BIN"
        } >> "$RC"
        say ""
        warn "Added $BIN to PATH in $RC — restart your shell or run: source $RC"
    fi
fi

# --- 7. Verify ------------------------------------------------------------
say ""
if "$VENV/bin/ollamadiffuser" --help >/dev/null 2>&1; then
    step "✅ Installed."
    say ""
    say "Quick start:"
    say "  ollamadiffuser recommend               # which models fit your hardware"
    say "  ollamadiffuser pull flux.1-schnell     # download a model"
    say "  ollamadiffuser run  flux.1-schnell     # serve it"
    say ""
    say "Optional backends (opt-in):"
    say "  ollamadiffuser enable gguf             # low-VRAM quantized models (compiles once)"
    say "  ollamadiffuser enable mcp              # Model Context Protocol server"
    say ""
    say "Uninstall:  rm -rf $ODIR${linked:+  &&  rm -f $linked}"
else
    die "Install finished but 'ollamadiffuser --help' failed. See output above."
fi
