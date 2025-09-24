"""Utility to sanity-check dataset batches for invalid labels."""

import argparse
import itertools
from pathlib import Path
import sys

import torch
import yaml

def _resolve_project_root(start: Path, package: str = "speechllm") -> Path:
    current = start
    candidates = [current] + list(current.parents)
    for candidate in candidates:
        if (candidate / package).exists():
            return candidate
    raise RuntimeError(f"Unable to locate project root containing '{package}' starting from {start}")

PROJECT_ROOT = _resolve_project_root(Path(__file__).resolve().parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from speechllm.data.dataset import SpeechLLMDataset, create_dataloader
    from speechllm.codecs.vocab_manager import VocabManager
    from speechllm.codecs.audio_tokenizer import AudioTokenizer
    from speechllm.align.interleaving import InterleavingGenerator
except ImportError as exc:
    raise SystemExit(
        "Missing optional dependency while importing project modules. "
        "Make sure packages like openai-whisper are installed.\n"
        f"Original error: {exc}"
    )


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloader(cfg):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    vocab_manager = VocabManager(
        base_tokenizer_name=model_cfg["llm_model_name"],
        num_rvq_layers=model_cfg["num_rvq_layers"],
        codebook_size=model_cfg["codebook_size"],
    )
    audio_tokenizer = AudioTokenizer(vocab_manager=vocab_manager)
    interleaving = InterleavingGenerator(
        audio_tokenizer=audio_tokenizer,
        vocab_manager=vocab_manager,
    )

    dataset = SpeechLLMDataset(
        data_file=data_cfg["train_data_file"],
        audio_tokenizer=audio_tokenizer,
        vocab_manager=vocab_manager,
        interleaving_generator=interleaving,
        max_text_length=data_cfg["max_text_length"],
        max_audio_length=data_cfg["max_audio_length"],
        sample_rate=data_cfg["sample_rate"],
        mode_weights=data_cfg["mode_weights"],
        cache_audio_tokens=data_cfg["cache_audio_tokens"],
        cache_dir=data_cfg.get("cache_dir"),
    )

    dataloader = create_dataloader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    return dataset, dataloader


def analyse_batch(batch, batch_idx: int):
    labels = batch["labels"].clone()
    mask = labels != -100
    per_sample_valid = mask.sum(dim=1)

    bad_indices = (per_sample_valid == 0).nonzero(as_tuple=False).view(-1)

    stats = {
        "batch_index": batch_idx,
        "batch_size": labels.size(0),
        "num_all_ignored": bad_indices.numel(),
        "num_nan_labels": torch.isnan(labels).sum().item(),
        "num_inf_labels": torch.isinf(labels).sum().item(),
    }

    if mask.any():
        stats["max_label"] = labels[mask].max().item()
        stats["min_label"] = labels[mask].min().item()
    else:
        stats["max_label"] = None
        stats["min_label"] = None

    return stats, bad_indices


def main():
    parser = argparse.ArgumentParser(description="Check dataset batches for invalid labels")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--max-batches", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    dataset, dataloader = build_dataloader(cfg)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Inspecting first {args.max_batches} batches...\n")

    problems = []
    for batch_idx, batch in itertools.islice(enumerate(dataloader), args.max_batches):
        stats, bad_indices = analyse_batch(batch, batch_idx)
        print(stats)
        if bad_indices.numel():
            problems.append((batch_idx, bad_indices.tolist()))

    if problems:
        print("\nSamples with no valid labels detected:")
        for batch_idx, indices in problems:
            print(f"  batch {batch_idx}: indices {indices}")
        print("\nRemove or fix these samples before training.")
    else:
        print("\nNo invalid samples found in inspected batches.")


if __name__ == "__main__":
    main()
