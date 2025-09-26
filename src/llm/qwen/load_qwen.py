"""Utilities for loading Qwen 7B with optional LoRA adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

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
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    adapters: tuple[str, ...] = ()
    adapter_paths: Dict[str, str] = field(default_factory=dict)


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
        adapter_items: Tuple[Tuple[str, str], ...] = tuple(self.config.adapter_paths.items())
        if adapter_items:
            primary_name, primary_path = adapter_items[0]
            self.model = PeftModel.from_pretrained(self.model, primary_path, adapter_name=primary_name)
            for name, path in adapter_items[1:]:
                self.model.load_adapter(path, adapter_name=name)
            if hasattr(self.model, "enable_adapter_layers"):
                self.model.enable_adapter_layers()
            for name, _ in adapter_items:
                self.model.set_adapter(name)
            self.model.set_adapter(primary_name)
            self.config.adapters = tuple(name for name, _ in adapter_items)
        elif self.config.lora_r > 0:
            lora = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            primary_adapter = self.config.adapters[0] if self.config.adapters else "default"
            self.model = get_peft_model(self.model, lora, adapter_name=primary_adapter)
            for extra_adapter in self.config.adapters[1:]:
                self.model.add_adapter(extra_adapter, lora)
            self.model.set_adapter(primary_adapter)
        if isinstance(self.model, PeftModel) and hasattr(self.model, "enable_adapter_layers"):
            self.model.enable_adapter_layers()
            if self.config.adapters:
                for name in self.config.adapters:
                    self.model.set_adapter(name)
                self.model.set_adapter(self.config.adapters[0])
        self.model.eval()

    def load_lora_weights(self, lora_path: str, adapters: Optional[tuple[str, ...]] = None) -> None:
        if not isinstance(self.model, PeftModel):
            raise RuntimeError("Model was not initialized with LoRA")
        names = adapters or (self.config.adapters if self.config.adapters else ("default",))
        for adapter_name in names:
            self.model.load_adapter(lora_path, adapter_name=adapter_name)
        self.model.set_adapter(names[0])

    @torch.inference_mode()
    def generate(self, prompt_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.generate(input_ids=prompt_ids.to(self.model.device), **kwargs)
