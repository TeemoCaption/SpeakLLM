import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WhisperFeatureExtractor,
    WhisperModel,
)

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class OptimizerConfig:
    lr_adapter: float
    lr_audio_head: Optional[float] = None
    lr_text_head: Optional[float] = None
    lr_qwen_unfrozen: Optional[float] = None
    lr_qwen_lora: Optional[float] = None
    lr_whisper_unfrozen: Optional[float] = None
    lr_interrupt_head: Optional[float] = None
    warmup_steps: int = 0
    scheduler: str = "cosine"
    min_lr_ratio: float = 0.05


@dataclass
class TrainConfig:
    seed: int
    precision: str
    train_steps: int
    save_interval: int
    log_interval: int
    validation_interval: int
    resume_from: Optional[str]
    optimizer: OptimizerConfig
    batching: Dict
    loss: Dict
    mixed_precision: Optional[str] = None
    use_deepspeed: bool = False
    accelerate_config: Optional[str] = None


def load_yaml(path: Path) -> Dict:
    if yaml is None:
        raise ImportError("PyYAML 尚未安裝，請先執行 pip install pyyaml")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_accelerator(cfg: TrainConfig, output_dir: Path) -> Accelerator:
    project_config = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs"))
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision or cfg.precision,
        project_config=project_config,
        log_with=["tensorboard", "wandb"],
    )
    return accelerator


def _gather_special_tokens(model_cfg: Dict) -> List[str]:
    tokens: List[str] = []
    for value in model_cfg.get("special_tokens", {}).values():
        if isinstance(value, list):
            tokens.extend(value)
        elif isinstance(value, str):
            tokens.append(value)
    tokens.append("<AUDIO_CHUNK>")
    return sorted(set(tokens))


def prepare_tokenizer(model_cfg: Dict) -> AutoTokenizer:
    tokenizer_path = model_cfg["qwen"]["tokenizer_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    special_tokens = _gather_special_tokens(model_cfg)
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_manifests(stage_cfg: Dict, key: str) -> List[Dict]:
    entries = stage_cfg.get(key, [])
    if not entries:
        raise ValueError(f"缺少 {key} 設定，請在階段 YAML 中提供 manifest 列表")
    resolved: List[Dict] = []
    for item in entries:
        manifest_path = Path(item["manifest"]).expanduser()
        resolved.append(
            {
                "manifest": manifest_path,
                "weight": float(item.get("weight", 1.0)),
                "type": item.get("type", "generic"),
            }
        )
    return resolved


class SpeechSequenceDataset(Dataset):
    def __init__(
        self,
        manifests: List[Dict],
        tokenizer: AutoTokenizer,
        num_codebooks: int,
        stage_cfg: Dict,
        split: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.stage_cfg = stage_cfg
        self.split = split
        batching_cfg = stage_cfg.get("batching", {})
        self.max_context = batching_cfg.get("max_context", 2048)
        fields = stage_cfg.get("data_fields", {})
        self.text_field = fields.get("text", "text")
        self.prompt_field = fields.get("prompt", "prompt")
        self.response_field = fields.get("response", "response")
        self.codes_field = fields.get("audio_codes", "audio_codes")
        self.codes_path_field = fields.get("codes_path", "codes_path")
        self.mel_path_field = fields.get("mel_path", "mel_path")
        self.chunk_spans_field = fields.get("chunk_spans", "chunk_spans")
        self.interrupt_label_field = fields.get("interrupt_label", "interrupt_label")
        self.records: List[Dict] = []
        for item in manifests:
            manifest_path = item["manifest"]
            if not manifest_path.exists():
                raise FileNotFoundError(f"找不到 manifest 檔案 {manifest_path}")
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    data = json.loads(line)
                    data["_weight"] = item["weight"]
                    data["_type"] = item["type"]
                    self.records.append(data)
        if not self.records:
            raise ValueError(f"{split} 資料為空，請確認 manifest 內容")
        spec_cfg = stage_cfg.get("spec_augment", {})
        self.enable_spec = bool(spec_cfg.get("enabled", False)) and split == "train"
        self.time_mask_param = spec_cfg.get("time_mask_param", 20)
        self.time_mask_count = spec_cfg.get("time_mask_count", 2)
        self.freq_mask_param = spec_cfg.get("freq_mask_param", 15)
        self.freq_mask_count = spec_cfg.get("freq_mask_count", 2)
        noise_cfg = stage_cfg.get("noise_augmentation", {})
        self.noise_probability = float(noise_cfg.get("probability", 0.0)) if split == "train" else 0.0
        self.noise_snr = noise_cfg.get("snr_db", [15, 25])

    def __len__(self) -> int:
        return len(self.records)

    def _load_mel(self, record: Dict) -> torch.Tensor:
        if self.mel_path_field in record:
            path = Path(record[self.mel_path_field]).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"缺少 log-mel 特徵檔案 {path}")
            if path.suffix == ".pt":
                mel = torch.load(path)
            else:
                mel = torch.from_numpy(np.load(path))
        elif "mel" in record:
            mel = torch.tensor(record["mel"], dtype=torch.float32)
        else:
            raise KeyError("資料缺少 mel 特徵欄位")
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        return mel.float()

    def _maybe_spec_augment(self, mel: torch.Tensor) -> torch.Tensor:
        if not self.enable_spec:
            return mel
        augmented = mel.clone()
        time_steps = augmented.size(0)
        freq_bins = augmented.size(1)
        for _ in range(self.time_mask_count):
            width = random.randint(0, self.time_mask_param)
            if width > 0:
                start = random.randint(0, max(time_steps - width, 0))
                augmented[start : start + width, :] = 0
        for _ in range(self.freq_mask_count):
            width = random.randint(0, self.freq_mask_param)
            if width > 0:
                start = random.randint(0, max(freq_bins - width, 0))
                augmented[:, start : start + width] = 0
        if self.noise_probability > 0 and random.random() < self.noise_probability:
            noise = torch.randn_like(augmented)
            if isinstance(self.noise_snr, list) and self.noise_snr:
                target_snr = random.uniform(float(self.noise_snr[0]), float(self.noise_snr[-1]))
            else:
                target_snr = float(self.noise_snr)
            signal_power = augmented.pow(2).mean().clamp(min=1e-6)
            noise_power = signal_power / (10 ** (target_snr / 10))
            augmented = augmented + noise * torch.sqrt(noise_power)
        return augmented

    def _load_codes(self, record: Dict) -> torch.Tensor:
        if self.codes_path_field in record:
            path = Path(record[self.codes_path_field]).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"缺少碼本索引檔案 {path}")
            data = np.load(path)
            codes = data["codes"] if isinstance(data, np.lib.npyio.NpzFile) else data
        elif self.codes_field in record:
            codes = np.asarray(record[self.codes_field], dtype=np.int64)
        else:
            return torch.zeros((0, self.num_codebooks), dtype=torch.long)
        tensor = torch.tensor(codes, dtype=torch.long)
        if tensor.dim() == 2 and tensor.size(0) == self.num_codebooks:
            tensor = tensor.transpose(0, 1)
        if tensor.dim() != 2 or tensor.size(1) != self.num_codebooks:
            raise ValueError("碼本索引形狀不符合預期")
        return tensor

    def _load_chunk_spans(self, record: Dict, num_chunks: int) -> List[Tuple[int, int]]:
        if self.chunk_spans_field in record:
            spans = record[self.chunk_spans_field]
            return [(int(span[0]), int(span[1])) for span in spans]
        frame_span = self.stage_cfg.get("voila_chunk_frames", 320)
        spans: List[Tuple[int, int]] = []
        for idx in range(num_chunks):
            start = idx * frame_span
            spans.append((start, start + frame_span))
        return spans

    def _build_text_tokens(self, record: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = record.get(self.prompt_field)
        response = record.get(self.response_field) or record.get(self.text_field)
        token_ids: List[int] = []
        label_ids: List[int] = []
        if self.tokenizer.bos_token_id is not None:
            token_ids.append(self.tokenizer.bos_token_id)
            label_ids.append(-100)
        if isinstance(prompt, str) and prompt:
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            token_ids.extend(prompt_ids)
            label_ids.extend([-100] * len(prompt_ids))
        if isinstance(response, str) and response:
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)
            token_ids.extend(response_ids)
            label_ids.extend(response_ids)
        if not token_ids:
            raise ValueError("文字序列為空")
        if len(token_ids) > self.max_context:
            token_ids = token_ids[-self.max_context :]
            label_ids = label_ids[-self.max_context :]
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = torch.tensor(label_ids, dtype=torch.long)
        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]
        mel = self._maybe_spec_augment(self._load_mel(record))
        audio_codes = self._load_codes(record)
        spans = self._load_chunk_spans(record, audio_codes.size(0))
        input_ids, labels = self._build_text_tokens(record)
        attention_mask = torch.ones_like(input_ids)
        interrupt_raw = int(record.get(self.interrupt_label_field, -100))
        return {
            "input_features": mel,
            "audio_codes": audio_codes,
            "chunk_spans": spans,
            "input_ids": input_ids,
            "labels_text": labels,
            "attention_mask": attention_mask,
            "interrupt_label": torch.tensor(interrupt_raw, dtype=torch.long),
        }


class SpeechBatchCollator:
    def __init__(self, tokenizer: AutoTokenizer, num_codebooks: int, max_context: int) -> None:
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.max_context = max_context

    def __call__(self, samples: List[Dict]) -> Dict:
        batch_size = len(samples)
        pad_id = self.tokenizer.pad_token_id
        max_seq = min(max(sample["input_ids"].size(0) for sample in samples), self.max_context)
        input_ids = torch.full((batch_size, max_seq), pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_seq), -100, dtype=torch.long)
        attention = torch.zeros((batch_size, max_seq), dtype=torch.long)
        max_frames = max(sample["input_features"].size(0) for sample in samples)
        mel_dim = samples[0]["input_features"].size(1)
        input_features = torch.zeros((batch_size, mel_dim, max_frames), dtype=torch.float32)
        max_chunks = max(sample["audio_codes"].size(0) for sample in samples)
        chunk_bounds = torch.full((batch_size, max_chunks, 2), -1, dtype=torch.long) if max_chunks > 0 else torch.zeros((batch_size, 0, 2), dtype=torch.long)
        chunk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.bool) if max_chunks > 0 else torch.zeros((batch_size, 0), dtype=torch.bool)
        audio_targets = torch.full((batch_size, max_chunks, self.num_codebooks), -100, dtype=torch.long) if max_chunks > 0 else torch.zeros((batch_size, 0, self.num_codebooks), dtype=torch.long)
        interrupt_labels = torch.stack([sample["interrupt_label"] for sample in samples])
        for idx, sample in enumerate(samples):
            seq = sample["input_ids"][:max_seq]
            input_ids[idx, : seq.size(0)] = seq
            lab = sample["labels_text"][:max_seq]
            labels[idx, : lab.size(0)] = lab
            attention[idx, : seq.size(0)] = 1
            mel = sample["input_features"]
            frames = mel.size(0)
            input_features[idx, :, :frames] = mel.transpose(0, 1)
            codes = sample["audio_codes"]
            spans = sample["chunk_spans"]
            if max_chunks > 0 and codes.size(0) > 0:
                audio_targets[idx, : codes.size(0)] = codes
                chunk_mask[idx, : len(spans)] = True
                for span_idx, span in enumerate(spans):
                    chunk_bounds[idx, span_idx] = torch.tensor(span, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels_text": labels,
            "attention_mask": attention,
            "input_features": input_features,
            "chunk_bounds": chunk_bounds,
            "chunk_mask": chunk_mask,
            "audio_targets": audio_targets,
            "interrupt_labels": interrupt_labels,
        }


def create_dataloaders(data_cfg: Dict, stage_cfg: Dict, model_cfg: Dict) -> Tuple[Iterable, Iterable, AutoTokenizer]:
    tokenizer = prepare_tokenizer(model_cfg)
    num_codebooks = model_cfg.get("voila", {}).get("codebooks", 4)
    train_entries = _resolve_manifests(stage_cfg, "train_manifests")
    eval_entries = _resolve_manifests(stage_cfg, "eval_manifests")
    train_dataset = SpeechSequenceDataset(train_entries, tokenizer, num_codebooks, stage_cfg, split="train")
    eval_dataset = SpeechSequenceDataset(eval_entries, tokenizer, num_codebooks, stage_cfg, split="eval")
    max_context = stage_cfg.get("batching", {}).get("max_context", 2048)
    collator = SpeechBatchCollator(tokenizer, num_codebooks, max_context)
    max_tokens = stage_cfg.get("batching", {}).get("max_tokens_per_device", 2048)
    batch_size = max(1, max_tokens // max_context)
    num_workers = stage_cfg.get("batching", {}).get("dataloader_num_workers", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collator)
    return train_loader, eval_loader, tokenizer


class WhisperAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        heads = max(1, hidden_dim // 512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.RMSNorm(hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.proj(hidden_states)
        encoded = self.transformer(projected)
        return self.norm(encoded)


class AudioCodebookHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_sizes: List[int], dropout: float = 0.1) -> None:
        super().__init__()
        modules = []
        for vocab in vocab_sizes:
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, vocab),
                )
            )
        self.heads = nn.ModuleList(modules)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        logits = [head(states) for head in self.heads]
        return torch.stack(logits, dim=2)


class InterruptibilityHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class SpeechLLMSystem(nn.Module):
    def __init__(self, model_cfg: Dict, stage_cfg: Dict, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.stage_cfg = stage_cfg
        self.tokenizer = tokenizer
        whisper_ckpt = model_cfg["whisper"]["checkpoint"]
        self.whisper_model = WhisperModel.from_pretrained(whisper_ckpt)
        self.whisper_encoder = self.whisper_model.get_encoder()
        adapter_cfg = model_cfg["adapter"]
        self.adapter = WhisperAdapter(
            input_dim=adapter_cfg["input_dim"],
            hidden_dim=adapter_cfg["output_dim"],
            layers=adapter_cfg.get("layers", 2),
            dropout=adapter_cfg.get("dropout", 0.1),
        )
        qwen_ckpt = model_cfg["qwen"]["checkpoint"]
        self.qwen_model = AutoModelForCausalLM.from_pretrained(qwen_ckpt, trust_remote_code=True)
        self.qwen_model.config.use_cache = False
        if self.qwen_model.get_input_embeddings().num_embeddings != len(tokenizer):
            self.qwen_model.resize_token_embeddings(len(tokenizer))
        qwen_cfg = model_cfg["qwen"]
        freeze_cfg = stage_cfg.get("freeze", {})
        if freeze_cfg.get("qwen", {}).get("use_lora", qwen_cfg.get("use_lora", False)):
            lora_options = qwen_cfg.get("lora", {})
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_options.get("r", 16),
                lora_alpha=lora_options.get("alpha", 16),
                lora_dropout=lora_options.get("dropout", 0.05),
                target_modules=lora_options.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            )
            self.qwen_model = get_peft_model(self.qwen_model, lora_config)
        self.audio_chunk_token_id = tokenizer.convert_tokens_to_ids("<AUDIO_CHUNK>")
        audio_head_cfg = model_cfg.get("heads", {}).get("audio", {})
        vocab_sizes = audio_head_cfg.get("vocab_sizes", [8192] * model_cfg.get("voila", {}).get("codebooks", 4))
        self.audio_head = AudioCodebookHead(adapter_cfg["output_dim"], vocab_sizes, dropout=audio_head_cfg.get("dropout", 0.1))
        interrupt_cfg = stage_cfg.get("additional_heads", {}).get("interruptibility")
        self.interrupt_head: Optional[InterruptibilityHead]
        if interrupt_cfg:
            self.interrupt_head = InterruptibilityHead(adapter_cfg["output_dim"])
        else:
            self.interrupt_head = None
        self.num_codebooks = len(vocab_sizes)
        self._apply_whisper_freeze(freeze_cfg.get("whisper", {}))
        self._apply_qwen_freeze(freeze_cfg.get("qwen", {}))

    def _apply_whisper_freeze(self, cfg: Dict) -> None:
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        trainable_layers = cfg.get("trainable_layers")
        if trainable_layers:
            for layer_idx in trainable_layers:
                if 0 <= layer_idx < len(self.whisper_encoder.layers):
                    for param in self.whisper_encoder.layers[layer_idx].parameters():
                        param.requires_grad = True

    def _apply_qwen_freeze(self, cfg: Dict) -> None:
        frozen_blocks = cfg.get("frozen_blocks", [])
        if not frozen_blocks:
            return
        for name, module in self.qwen_model.named_modules():
            parts = name.split(".")
            if len(parts) >= 2 and parts[-2] == "layers":
                try:
                    block_idx = int(parts[-1])
                except ValueError:
                    continue
                if block_idx in frozen_blocks:
                    for param in module.parameters():
                        param.requires_grad = False

    def _pool_chunks(self, adapter_hidden: torch.Tensor, chunk_bounds: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        if chunk_bounds.numel() == 0:
            return adapter_hidden.new_zeros((adapter_hidden.size(0), 0, adapter_hidden.size(-1)))
        batch_size, max_chunks, _ = chunk_bounds.size()
        pooled = adapter_hidden.new_zeros((batch_size, max_chunks, adapter_hidden.size(-1)))
        for b in range(batch_size):
            spans = chunk_bounds[b]
            valid = chunk_mask[b]
            for idx, flag in enumerate(valid):
                if not bool(flag):
                    continue
                start = int(spans[idx, 0].item())
                end = int(spans[idx, 1].item())
                end = max(end, start + 1)
                segment = adapter_hidden[b, start:end]
                pooled[b, idx] = segment.mean(dim=0)
        return pooled

    def forward(self, batch: Dict) -> Dict:
        input_features = batch["input_features"].to(self.whisper_encoder.device)
        input_ids = batch["input_ids"].to(self.qwen_model.device)
        attention_mask = batch["attention_mask"].to(self.qwen_model.device)
        labels_text = batch["labels_text"].to(self.qwen_model.device)
        chunk_bounds = batch["chunk_bounds"].to(self.whisper_encoder.device)
        chunk_mask = batch["chunk_mask"].to(self.whisper_encoder.device)
        audio_targets = batch["audio_targets"].to(self.qwen_model.device)
        interrupt_labels = batch["interrupt_labels"].to(self.qwen_model.device)
        encoder_outputs = self.whisper_encoder(input_features, return_dict=True)
        hidden_states = encoder_outputs.last_hidden_state
        adapted = self.adapter(hidden_states)
        chunk_states = self._pool_chunks(adapted, chunk_bounds, chunk_mask).to(self.qwen_model.device)
        if self.audio_chunk_token_id is not None:
            embeds = self.qwen_model.get_input_embeddings()(input_ids)
            for b in range(input_ids.size(0)):
                audio_positions = (input_ids[b] == self.audio_chunk_token_id).nonzero(as_tuple=False).squeeze(-1)
                valid_chunks = int(chunk_mask[b].sum().item())
                if audio_positions.numel() == 0 or valid_chunks == 0:
                    continue
                limit = min(audio_positions.numel(), valid_chunks)
                embeds[b, audio_positions[:limit]] = embeds[b, audio_positions[:limit]] + chunk_states[b, :limit]
            textual_outputs = self.qwen_model(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels_text, return_dict=True)
        else:
            textual_outputs = self.qwen_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_text, return_dict=True)
        total_loss = textual_outputs.loss
        losses: Dict[str, torch.Tensor] = {"loss_text": textual_outputs.loss.detach()}
        if chunk_states.size(1) > 0 and audio_targets.size(1) > 0:
            logits = self.audio_head(chunk_states)
            audio_mask = chunk_mask.to(self.qwen_model.device)
            audio_loss_total = torch.tensor(0.0, device=self.qwen_model.device)
            for idx in range(self.num_codebooks):
                target = audio_targets[:, :, idx]
                logit = logits[:, :, idx, :]
                if audio_mask.any():
                    masked_logit = logit[audio_mask]
                    masked_target = target[audio_mask]
                    if masked_target.numel() > 0:
                        audio_loss_total = audio_loss_total + F.cross_entropy(masked_logit, masked_target, ignore_index=-100)
            if audio_mask.any():
                total_loss = total_loss + audio_loss_total
                losses["loss_audio"] = audio_loss_total.detach()
        if self.interrupt_head is not None and (interrupt_labels >= 0).any():
            pooled = chunk_states.mean(dim=1)
            logits_interrupt = self.interrupt_head(pooled)
            valid_mask = interrupt_labels >= 0
            if valid_mask.any():
                interrupt_loss = F.cross_entropy(logits_interrupt[valid_mask], interrupt_labels[valid_mask])
                total_loss = total_loss + interrupt_loss
                losses["loss_interrupt"] = interrupt_loss.detach()
        losses["loss"] = total_loss
        return losses


def _collect_parameters(model: SpeechLLMSystem, cfg: OptimizerConfig) -> List[Dict]:
    groups: List[Dict] = []
    adapter_params = [p for p in model.adapter.parameters() if p.requires_grad]
    if adapter_params:
        groups.append({"params": adapter_params, "lr": cfg.lr_adapter})
    audio_params = [p for p in model.audio_head.parameters() if p.requires_grad]
    if audio_params and cfg.lr_audio_head:
        groups.append({"params": audio_params, "lr": cfg.lr_audio_head})
    if model.interrupt_head is not None and cfg.lr_interrupt_head:
        head_params = [p for p in model.interrupt_head.parameters() if p.requires_grad]
        if head_params:
            groups.append({"params": head_params, "lr": cfg.lr_interrupt_head})
    lora_params: List[torch.nn.Parameter] = []
    qwen_main_params: List[torch.nn.Parameter] = []
    for name, param in model.qwen_model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora" in name and cfg.lr_qwen_lora:
            lora_params.append(param)
        elif cfg.lr_qwen_unfrozen:
            qwen_main_params.append(param)
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg.lr_qwen_lora})
    if qwen_main_params:
        groups.append({"params": qwen_main_params, "lr": cfg.lr_qwen_unfrozen})
    whisper_params = [p for p in model.whisper_encoder.parameters() if p.requires_grad]
    if whisper_params and cfg.lr_whisper_unfrozen:
        groups.append({"params": whisper_params, "lr": cfg.lr_whisper_unfrozen})
    return groups


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: OptimizerConfig, total_steps: int):
    if cfg.scheduler != "cosine":
        return None
    warmup = cfg.warmup_steps
    min_ratio = cfg.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model(model_cfg: Dict, stage_cfg: Dict, tokenizer: AutoTokenizer, total_steps: int):
    system = SpeechLLMSystem(model_cfg, stage_cfg, tokenizer)
    optimizer_cfg = OptimizerConfig(**stage_cfg["optimizer"])
    param_groups = _collect_parameters(system, optimizer_cfg)
    if not param_groups:
        raise ValueError("沒有可訓練參數，請檢查凍結設定與學習率配置")
    opt_cfg = model_cfg.get("optimization", {})
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )
    scheduler = _build_scheduler(optimizer, optimizer_cfg, total_steps)
    return system, optimizer, scheduler


def save_checkpoint(accelerator: Accelerator, model, optimizer, step: int, output_dir: Path) -> None:
    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(ckpt_dir))


def log_metrics(accelerator: Accelerator, metrics: Dict, step: int) -> None:
    accelerator.log(metrics, step=step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech LLM 多階段訓練腳本")
    parser.add_argument("--config", type=str, required=True, help="階段訓練 YAML 路徑")
    parser.add_argument("--model", type=str, required=True, help="模型組態 YAML 路徑")
    parser.add_argument("--data", type=str, required=True, help="資料混合 YAML 路徑")
    parser.add_argument("--stage", type=str, default=None, help="多階段設定下要啟用的 stage 名稱")
    parser.add_argument("--output_dir", type=str, required=True, help="輸出 checkpoint 與紀錄的資料夾")
    parser.add_argument("--log_with", nargs="*", default=None, help="額外 Accelerate logger 名稱")
    parser.add_argument("--resume_from", type=str, default=None, help="欲恢復的 checkpoint 路徑")
    parser.add_argument("--wandb_project", type=str, default="speech-llm", help="wandb 專案名稱")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_stage_cfg = load_yaml(Path(args.config))
    if "stages" in raw_stage_cfg:
        if not args.stage:
            raise ValueError("--stage must be provided when config contains multiple stages")
        if args.stage not in raw_stage_cfg["stages"]:
            raise KeyError(f"Unknown stage '{args.stage}' in {args.config}")
        stage_cfg = raw_stage_cfg["stages"][args.stage]
        stage_name = args.stage
    else:
        stage_cfg = raw_stage_cfg
        stage_name = args.stage or Path(args.config).stem
    model_cfg = load_yaml(Path(args.model))
    data_cfg = load_yaml(Path(args.data))

    optimizer_cfg = OptimizerConfig(**stage_cfg["optimizer"])
    train_cfg = TrainConfig(
        seed=stage_cfg.get("seed", 42),
        precision=stage_cfg.get("precision", "bf16"),
        train_steps=stage_cfg["train_steps"],
        save_interval=stage_cfg["save_interval"],
        log_interval=stage_cfg["log_interval"],
        validation_interval=stage_cfg["validation_interval"],
        resume_from=args.resume_from or stage_cfg.get("resume_from"),
        optimizer=optimizer_cfg,
        batching=stage_cfg["batching"],
        loss=stage_cfg["loss"],
        mixed_precision=stage_cfg.get("mixed_precision"),
        use_deepspeed=stage_cfg.get("use_deepspeed", False),
        accelerate_config=stage_cfg.get("accelerate_config"),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(train_cfg.seed)

    accelerator = build_accelerator(train_cfg, output_dir)
    if args.log_with:
        tracker_cfg = {"stage": stage_name, "stage_config": stage_cfg, "model_config": model_cfg}
        accelerator.init_trackers(args.wandb_project, config=tracker_cfg)

    accelerator.print("載入資料中...")
    try:
        train_loader, eval_loader, tokenizer = create_dataloaders(data_cfg, stage_cfg, model_cfg)
    except Exception as exc:  # noqa: BLE001
        accelerator.print(f"資料載入失敗: {exc}")
        return

    accelerator.print("建構模型中...")
    try:
        model, optimizer, scheduler = build_model(model_cfg, stage_cfg, tokenizer, train_cfg.train_steps)
    except Exception as exc:  # noqa: BLE001
        accelerator.print(f"模型建置失敗: {exc}")
        return

    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)

    global_step = 0
    best_metric = math.inf

    for step, batch in enumerate(train_loader, start=1):
        model.train()
        outputs = model(batch)
        loss = outputs["loss"]
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        if step % train_cfg.log_interval == 0:
            metrics = {f"train/{k}": (v.item() if torch.is_tensor(v) else float(v)) for k, v in outputs.items()}
            log_metrics(accelerator, metrics, step)

        if step % train_cfg.validation_interval == 0:
            accelerator.print("執行驗證...")
            model.eval()
            eval_loss = 0.0
            eval_batches = 0
            with torch.no_grad():
                for eval_batch in eval_loader:
                    eval_outputs = model(eval_batch)
                    eval_loss += float(eval_outputs["loss"].item())
                    eval_batches += 1
            eval_loss /= max(eval_batches, 1)
            log_metrics(accelerator, {"eval/loss": eval_loss}, step)
            if eval_loss < best_metric:
                best_metric = eval_loss
                save_checkpoint(accelerator, model, optimizer, step, output_dir / "best")

        if step % train_cfg.save_interval == 0:
            save_checkpoint(accelerator, model, optimizer, step, output_dir)

        global_step = step
        if step >= train_cfg.train_steps:
            break

    accelerator.print(f"訓練完成，總步數 {global_step}，最佳驗證指標 {best_metric:.4f}")


if __name__ == "__main__":
    main()
