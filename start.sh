#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "Installing project requirements..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo "Starting MLX Auto-Router..."
echo "If no local models exist, the setup wizard will offer model downloads."
exec "$VENV_DIR/bin/python" mlx_dashboard.py