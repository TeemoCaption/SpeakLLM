#!/bin/bash
set -euo pipefail

OUTPUT_DIR=onnx_exports
export PYTHONPATH=src:

python - <<'PY'
from pathlib import Path
import torch

from asr.connector.model import Connector, ConnectorConfig

output_dir = Path("")
output_dir.mkdir(parents=True, exist_ok=True)
model = Connector(ConnectorConfig())
dummy = torch.randn(1, 200, 768)
onnx_path = output_dir / "connector.onnx"
torch.onnx.export(model, dummy, onnx_path, input_names=["features"], output_names=["projected"], dynamic_axes={"features": {1: "time"}})
print(f"Connector exported to {onnx_path}")
PY
