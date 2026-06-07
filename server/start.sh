#!/usr/bin/env bash
# start.sh — run the Sauron server locally without Docker
# Requires: uv (https://docs.astral.sh/uv/getting-started/installation/)
#           PostgreSQL 15+ running locally (see SETUP POSTGRESQL below)
#
# Usage:
#   ./start.sh              — production mode (port 8000)
#   ./start.sh --dev        — development mode with auto-reload
#   ./start.sh --port 9000  — custom port

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=8000
RELOAD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)   RELOAD="--reload"; shift ;;
        --port)  PORT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Sync dependencies ──────────────────────────────────────────────────────
echo "[start.sh] Syncing dependencies with uv..."
uv sync

# ── Check .env ─────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    echo "[start.sh] WARNING: .env not found. Creating a minimal template..."
    cat > .env <<'EOF'
DATABASE_URL=postgresql://iotel:iotel@localhost:5432/iotel
IOT_ENDPOINT=
IOT_CERT_PATH=certs/certificate.pem.crt
IOT_KEY_PATH=certs/private.pem.key
IOT_CA_PATH=certs/AmazonRootCA1.pem
IOT_CLIENT_ID=iotel-server
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
EOF
    echo "[start.sh] Edit .env and set IOT_ENDPOINT, then re-run."
fi

# ── Start server ───────────────────────────────────────────────────────────
echo "[start.sh] Starting server on port $PORT..."
uv run uvicorn app.main:app --host 0.0.0.0 --port "$PORT" $RELOAD

# ─────────────────────────────────────────────────────────────────────────────
# SETUP POSTGRESQL (one-time, run manually)
# ─────────────────────────────────────────────────────────────────────────────
#
# macOS (Homebrew):
#   brew install postgresql@15
#   brew services start postgresql@15
#   createdb iotel
#   psql iotel -c "CREATE USER iotel WITH PASSWORD 'iotel';"
#   psql iotel -c "GRANT ALL PRIVILEGES ON DATABASE iotel TO iotel;"
#
# Ubuntu / Raspberry Pi OS:
#   sudo apt-get install -y postgresql
#   sudo systemctl start postgresql
#   sudo -u postgres psql -c "CREATE USER iotel WITH PASSWORD 'iotel';"
#   sudo -u postgres psql -c "CREATE DATABASE iotel OWNER iotel;"
#
# TimescaleDB (optional — gives better time-series query performance):
#   Follow https://docs.timescale.com/self-hosted/latest/install/
#   The server degrades gracefully to plain PostgreSQL if TimescaleDB
#   is absent — you will see a log warning but nothing will break.
#
# Then set DATABASE_URL in .env:
#   DATABASE_URL=postgresql://iotel:iotel@localhost:5432/iotel
