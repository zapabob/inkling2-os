#!/usr/bin/env bash
# One-Command Installer for Meta Muse.ai Cloud Linux VM (Outbound-Only Tailscale -> PC)
# Run on Muse.ai VM:
#   PC_TARGET="downl@<your-pc-tailscale-ip>" bash install_on_muse_vm.sh
set -euo pipefail

PC_TARGET="${PC_TARGET:?Set PC_TARGET=user@pc-tailscale-ip (e.g. downl@100.x.y.z)}"
PC_REPO_DIR="${PC_REPO_DIR:-C:/Users/downl/inkling2-os}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/inkling2-os}"
STATE_DIR="${STATE_DIR:-$HOME/inkling2-state}"

echo "[Inkling2->Muse.ai] Installing into $INSTALL_DIR (state=$STATE_DIR)..."
mkdir -p "$INSTALL_DIR/bin" "$INSTALL_DIR/python" "$STATE_DIR"

# 1. Pull VM sync script and hakua-memory bridge from PC over Tailscale SSH
SSH_OPTS=(-o BatchMode=yes -o PasswordAuthentication=no -o ConnectTimeout=10)
scp "${SSH_OPTS[@]}" "$PC_TARGET:$PC_REPO_DIR/scripts/muse_vm_sync.sh" "$INSTALL_DIR/bin/muse_vm_sync.sh"
scp "${SSH_OPTS[@]}" "$PC_TARGET:$PC_REPO_DIR/python/hakua_bridge.py" "$INSTALL_DIR/python/hakua_bridge.py"
chmod +x "$INSTALL_DIR/bin/muse_vm_sync.sh"

# 2. Install uv + hakua-memory==0.3.5 inside Muse.ai VM
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv "$INSTALL_DIR/.venv"
uv pip install --python "$INSTALL_DIR/.venv/bin/python" hakua-memory
"$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/python/hakua_bridge.py" --probe

# 3. Initial pull of 45MB compressed memory & dynamic LoRA weights from PC
PC_TARGET="$PC_TARGET" VM_STATE_DIR="$STATE_DIR" "$INSTALL_DIR/bin/muse_vm_sync.sh" pull

# 4. Register helper environment script
cat > "$INSTALL_DIR/env.sh" <<EOF
export INKLING2_PC_TARGET="$PC_TARGET"
export INKLING2_STATE_DIR="$STATE_DIR"
export PATH="$INSTALL_DIR/bin:\$PATH"
alias muse-pull='PC_TARGET="$PC_TARGET" VM_STATE_DIR="$STATE_DIR" "$INSTALL_DIR/bin/muse_vm_sync.sh" pull'
alias muse-push='PC_TARGET="$PC_TARGET" VM_STATE_DIR="$STATE_DIR" "$INSTALL_DIR/bin/muse_vm_sync.sh" push'
alias muse-tunnel='PC_TARGET="$PC_TARGET" "$INSTALL_DIR/bin/muse_vm_sync.sh" tunnel'
EOF

echo "[Inkling2->Muse.ai] Installation complete!"
echo "  source $INSTALL_DIR/env.sh"
echo "  muse-pull   # Pull 45MB memory + Dynamic LoRA from PC"
echo "  muse-push   # Push updated memory + LoRA back to PC inbox"
echo "  muse-tunnel # Open VM->PC encrypted tunnel (127.0.0.1:18080 -> PC:8080)"
