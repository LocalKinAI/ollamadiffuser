#!/bin/bash
#
# OllamaDiffuser installer.
#
# Design goal: a compile-free default. The core install pulls only
# prebuilt wheels (no CMake, no CUDA toolchain). The one backend that
# needs to compile — GGUF — is opt-in via `ollamadiffuser enable gguf`.
#
set -e

echo "🎨 OllamaDiffuser Installer"
echo "==========================="

command_exists() { command -v "$1" >/dev/null 2>&1; }

# --- Python 3.10+ ---
if ! command_exists python3; then
    echo "❌ Python 3 not found. Install Python 3.10+ first."
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
MIN_VERSION="3.10"
if [ "$(printf '%s\n' "$MIN_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$MIN_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION detected. OllamaDiffuser needs Python 3.10+."
    exit 1
fi
echo "✅ Python $PYTHON_VERSION"

if ! command_exists pip3; then
    echo "❌ pip3 not found. Install pip first."
    exit 1
fi

# --- Core install (compile-free: prebuilt wheels only) ---
echo ""
echo "📦 Installing OllamaDiffuser (compile-free core)..."
pip3 install --upgrade pip >/dev/null
pip3 install ollamadiffuser

# --- Apple Silicon: enable the fast MLX path (also compile-free) ---
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    echo ""
    echo "🍎 Apple Silicon detected — enabling MLX backend (typically 2-3× faster)..."
    pip3 install "mflux>=0.17.0" \
        || echo "⚠️  MLX install skipped; run 'ollamadiffuser enable mlx' later."
fi

# --- Verify ---
echo ""
if command_exists ollamadiffuser; then
    echo "✅ Installed."
    echo ""
    echo "Quick start:"
    echo "  ollamadiffuser recommend               # which models fit your hardware"
    echo "  ollamadiffuser pull flux.1-schnell     # download a model"
    echo "  ollamadiffuser run  flux.1-schnell     # serve it"
    echo ""
    echo "Optional backends (opt-in, no bracket-quoting needed):"
    echo "  ollamadiffuser enable gguf             # low-VRAM quantized models (compiles)"
    echo "  ollamadiffuser enable mcp              # Model Context Protocol server"
    echo "  ollamadiffuser enable mlx              # Apple Silicon native (if skipped above)"
else
    echo "❌ 'ollamadiffuser' not on PATH after install."
    echo "   pip may have installed to a user directory that isn't on your PATH."
    echo "   Try: python3 -m ollamadiffuser --help"
    exit 1
fi
