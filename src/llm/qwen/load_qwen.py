"""Utilities for loading Qwen 7B with optional LoRA adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class QwenLoadConfig:
    checkpoint: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    max_length: int = 16384
    lora_r: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


class QwenLoader:
    def __init__(self, config: QwenLoadConfig | None = None) -> None:
        self.config = config or QwenLoadConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint, trust_remote_code=True)
        model_config = AutoConfig.from_pretrained(self.config.checkpoint, trust_remote_code=True)
        model_config.max_position_embeddings = max(model_config.max_position_embeddings, self.config.max_length)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.checkpoint,
            config=model_config,
            torch_dtype=self.config.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.config.lora_r > 0:
            lora = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora)
        self.model.eval()

    def load_lora_weights(self, lora_path: str) -> None:
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("Model was not initialized with LoRA")
        self.model.load_adapter(lora_path, adapter_name="default")
        self.model.set_adapter("default")

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids=prompt_ids.to(self.model.device), **kwargs)
