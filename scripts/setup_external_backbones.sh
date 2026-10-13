#!/usr/bin/env bash
# Clone external IRSTD backbones (SCTransNet, MSHNet) under HALO/external/ and install deps.
# Run once on the training machine:  bash scripts/setup_external_backbones.sh
set -e

HALO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$HALO_DIR/external"
mkdir -p "$EXT"

# Upstream revisions used for the manuscript's backbone comparison.
SCTRANSNET_COMMIT=e0276283794e9a54db334c2105f6e70107312202
MSHNET_COMMIT=46cdfd46802629da51f70124662af7335be74b56

[ -d "$EXT/SCTransNet" ] || git clone https://github.com/xdFai/SCTransNet.git "$EXT/SCTransNet"
[ -d "$EXT/MSHNet" ]     || git clone https://github.com/ying-fu/MSHNet.git     "$EXT/MSHNet"
git -C "$EXT/SCTransNet" checkout -q "$SCTRANSNET_COMMIT"
git -C "$EXT/MSHNet"     checkout -q "$MSHNET_COMMIT"

# SCTransNet deps (MSHNet needs only torch). thop is imported at module top in SCTransNet.py.
pip install ml_collections einops thop

echo "Done. External backbones cloned under: $EXT"
echo "Quick check:"
echo "  python -c \"from model import build_model; import torch; m=build_model('MSHNet',1,False); print(m(torch.rand(1,1,256,256)).shape)\""
