#!/bin/bash
set -euo pipefail

CONFIG=configs/train/phase1_sft.yaml
export PYTHONPATH=src:

python - <<'PY'
from pathlib import Path
from common.utils import load_yaml
from training.phase1_sft_trainer import Phase1Config, Phase1Trainer

config_path = Path("")
cfg_dict = load_yaml(config_path)
trainer = Phase1Trainer(Phase1Config())
print("Phase 1 trainer initialized (placeholder run)")
PY
