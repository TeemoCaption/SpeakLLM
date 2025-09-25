"""LoRA helpers for Qwen fine-tuning and export."""
from __future__ import annotations

from pathlib import Path

from peft import PeftModel


def save_lora(model: PeftModel, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)


def merge_lora(model: PeftModel, output_path: str) -> None:
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)
