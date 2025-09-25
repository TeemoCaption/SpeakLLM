#!/bin/bash
set -euo pipefail

CONFIG=configs/train/phase0_connector.yaml
export PYTHONPATH=src:

python - <<'PY'
from pathlib import Path
from common.utils import load_yaml
from training.phase0_connector_trainer import Phase0Config, Phase0Trainer

config_path = Path("")
cfg_dict = load_yaml(config_path)
trainer = Phase0Trainer(Phase0Config())
print("Phase 0 trainer initialized (placeholder run)")
PY
