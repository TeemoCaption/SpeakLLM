"""Run a single forward pass to check for NaNs/inf in model outputs."""

import argparse
from pathlib import Path

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
    from speechllm.models.speechllm import SpeechLLM, SpeechLLMConfig
    from speechllm.data.dataset import SpeechLLMDataset, create_dataloader
    from speechllm.codecs.vocab_manager import VocabManager
    from speechllm.codecs.audio_tokenizer import AudioTokenizer
    from speechllm.align.interleaving import InterleavingGenerator
except ImportError as exc:
    raise SystemExit(
        "Missing project dependency; install required packages first.\n"
        f"Original error: {exc}"
    )


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_components(cfg, batch_size: int):
    model_cfg = SpeechLLMConfig(**cfg["model"])
    model_cfg.mixed_precision = False  # ensure pure float32 for debug

    model = SpeechLLM(model_cfg)
    model.eval()

    data_cfg = cfg["data"]
    vocab_manager = VocabManager(
        base_tokenizer_name=model_cfg.llm_model_name,
        num_rvq_layers=model_cfg.num_rvq_layers,
        codebook_size=model_cfg.codebook_size,
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return model, dataloader


def check_forward(model, batch):
    device = torch.device("cpu")
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            mode=batch["modes"][0],
        )

    report = {}
    for key in ("loss", "logits", "rvq_loss", "hidden_states"):
        value = outputs.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            report[key] = {
                "is_nan": torch.isnan(value).any().item(),
                "is_inf": torch.isinf(value).any().item(),
                "shape": tuple(value.shape),
            }
        else:
            report[key] = "non-tensor"

    return report


def main():
    parser = argparse.ArgumentParser(description="Check SpeechLLM forward pass for NaNs")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Override batch size for the debug dataloader")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    model, dataloader = build_components(cfg, args.batch_size)

    for idx, batch in enumerate(dataloader):
        if idx == args.batch_index:
            print(f"Checking batch {idx}")
            report = check_forward(model, batch)
            for key, info in report.items():
                print(key, info)
            break
    else:
        print(f"Batch index {args.batch_index} not found (dataloader smaller).")


if __name__ == "__main__":
    main()
