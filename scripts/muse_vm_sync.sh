#!/usr/bin/env bash
# VM-Initiated Sync & Tunnel Script for Meta Muse.ai Cloud VM -> Inkling2-OS / PC (over Tailscale SSH)
# Usage on Muse.ai VM:
#   PC_TARGET="downl@<your-pc-tailscale-ip>" ./muse_vm_sync.sh pull
#   PC_TARGET="downl@<your-pc-tailscale-ip>" ./muse_vm_sync.sh push
#   PC_TARGET="downl@<your-pc-tailscale-ip>" ./muse_vm_sync.sh tunnel
set -euo pipefail

PC_TARGET="${PC_TARGET:?Set PC_TARGET=user@pc-tailscale-ip}"
PC_STATE_DIR="${PC_STATE_DIR:-C:/Users/downl/inkling2-os/state}"
VM_STATE_DIR="${VM_STATE_DIR:-$HOME/inkling2-state}"
LOCAL_PORT="${LOCAL_PORT:-18080}"
PC_PORT="${PC_PORT:-8080}"
SSH_OPTS=(-o BatchMode=yes -o PasswordAuthentication=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=10)

mkdir -p "$VM_STATE_DIR"

case "${1:-help}" in
  pull)
    echo "[Muse->PC] Pulling memory45.ikm and dynamic_lora.iklora from $PC_TARGET..."
    scp "${SSH_OPTS[@]}" "$PC_TARGET:$PC_STATE_DIR/memory45.ikm" "$VM_STATE_DIR/memory45.ikm" || true
    scp "${SSH_OPTS[@]}" "$PC_TARGET:$PC_STATE_DIR/dynamic_lora.iklora" "$VM_STATE_DIR/dynamic_lora.iklora" || true
    ls -lh "$VM_STATE_DIR"
    ;;
  push)
    echo "[Muse->PC] Pushing state into $PC_TARGET:$PC_STATE_DIR/muse-inbox/..."
    for f in memory45.ikm dynamic_lora.iklora; do
      if [[ -f "$VM_STATE_DIR/$f" ]]; then
        scp "${SSH_OPTS[@]}" "$VM_STATE_DIR/$f" "$PC_TARGET:$PC_STATE_DIR/muse-inbox/$f"
      fi
    done
    echo "[Muse->PC] Push complete. Run 'inkling2-shell -c \"muse import\"' on PC to validate & activate."
    ;;
  tunnel)
    echo "[Muse->PC] Opening VM-initiated tunnel: VM 127.0.0.1:$LOCAL_PORT -> PC 127.0.0.1:$PC_PORT"
    exec ssh "${SSH_OPTS[@]}" -N -o ExitOnForwardFailure=yes -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$PC_PORT" "$PC_TARGET"
    ;;
  *)
    echo "Usage: PC_TARGET=user@pc-tailscale-ip $0 <pull|push|tunnel>"
    ;;
esac
