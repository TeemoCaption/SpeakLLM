import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

try:
    import yaml
except ImportError:
    yaml = None

from datasets import load_dataset

# Local imports will be available once supporting modules are filled in.
# Placeholder typing hints keep the skeleton self-documented.


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
        raise ImportError("PyYAML is required but not installed. Run pip install pyyaml. ")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

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


def create_dataloaders(data_cfg: Dict, stage_cfg: Dict) -> Tuple[Iterable, Iterable]:
    # TODO: wire up actual dataset pipeline once data modules are implemented.
    # For now we provide placeholders to unblock training loop development.
    raise NotImplementedError("Dataset pipeline not yet implemented. Fill in create_dataloaders.")


def build_model(model_cfg: Dict, stage_cfg: Dict):
    # TODO: instantiate Whisper encoder, adapters, Qwen backbone, and heads.
    raise NotImplementedError("Model instantiation is pending. Implement uild_model with actual modules.")


def save_checkpoint(accelerator: Accelerator, model, optimizer, step: int, output_dir: Path) -> None:
    ckpt_dir = output_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(ckpt_dir))


def log_metrics(accelerator: Accelerator, metrics: Dict, step: int) -> None:
    accelerator.log(metrics, step=step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Voila-Tokenizer + Whisper + Qwen speech LLM")
    parser.add_argument("--config", type=str, required=True, help="Path to stage training config YAML")
    parser.add_argument("--model", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--data", type=str, required=True, help="Path to data mix YAML")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store checkpoints and logs")
    parser.add_argument("--log_with", nargs="*", default=None, help="Additional loggers for Accelerate")
    parser.add_argument("--resume_from", type=str, default=None, help="Optional checkpoint to resume")
    parser.add_argument("--wandb_project", type=str, default="speech-llm", help="wandb project name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_cfg = load_yaml(Path(args.config))
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
        accelerator.init_trackers(args.wandb_project, config={"stage_config": stage_cfg, "model_config": model_cfg})

    accelerator.print("Loading datasets...")
    try:
        train_loader, eval_loader = create_dataloaders(data_cfg, stage_cfg)
    except NotImplementedError as exc:
        accelerator.print(f"Dataset pipeline missing: {exc}")
        return

    accelerator.print("Building model...")
    try:
        model, optimizer, scheduler = build_model(model_cfg, stage_cfg)
    except NotImplementedError as exc:
        accelerator.print(f"Model construction missing: {exc}")
        return

    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)

    global_step = 0
    best_metric = math.inf

    for step, batch in enumerate(train_loader, start=1):
        model.train()
        outputs = model(**batch)
        loss = outputs["loss"]
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        if step % train_cfg.log_interval == 0:
            metrics = {"train/loss": loss.item()}
            log_metrics(accelerator, metrics, step)

        if step % train_cfg.validation_interval == 0:
            accelerator.print("Running evaluation...")
            model.eval()
            eval_loss = 0.0
            eval_batches = 0
            with torch.no_grad():
                for eval_batch in eval_loader:
                    outputs = model(**eval_batch)
                    eval_loss += outputs["loss"].item()
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

    accelerator.print(f"Training finished at step {global_step} with best metric {best_metric:.4f}")


if __name__ == "__main__":
    main()
