"""CLI for Speech-LLM data preparation and model utilities."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_MODELS: Dict[str, Dict[str, str]] = {
    "whisper": {"repo": "openai/whisper-medium", "subdir": "models/whisper-medium"},
    "qwen": {"repo": "Qwen/Qwen2-7B", "subdir": "models/qwen2-7b"},
    "voila": {"repo": "maitrix-org/Voila-Tokenizer", "subdir": "tokenizer/voila_tokenizer"},
}

AUDIO_TOKEN_PREFIXES = ["<L1_", "<L2_", "<L3_", "<L4_"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fetch models
# ---------------------------------------------------------------------------


def cmd_fetch_models(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    revisions: Dict[str, str] = {}
    if args.revision:
        for spec in args.revision:
            if "@" not in spec:
                raise ValueError(f"Invalid revision spec '{spec}', expected format key@revision")
            key, rev = spec.split("@", 1)
            if key not in DEFAULT_MODELS:
                raise ValueError(f"Unknown model key '{key}'")
            revisions[key] = rev

    root = Path(args.root)
    for key in args.items or DEFAULT_MODELS.keys():
        info = DEFAULT_MODELS[key]
        target_dir = root / info["subdir"]
        ensure_dir(target_dir)
        path = snapshot_download(
            repo_id=info["repo"],
            revision=revisions.get(key),
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Fetched {info['repo']} -> {path}")


# ---------------------------------------------------------------------------
# Audio token and vocab helpers
# ---------------------------------------------------------------------------


def cmd_gen_audio_tokens(args: argparse.Namespace) -> None:
    ensure_dir(Path(args.output).parent)
    with open(args.output, "w", encoding="utf-8") as fout:
        for prefix in AUDIO_TOKEN_PREFIXES:
            for idx in range(args.codes):
                fout.write(f"{prefix}{idx:04d}>\n")
    print(f"Wrote audio tokens to {args.output}")


def cmd_extend_vocab(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokens = [line.strip() for line in Path(args.tokens).read_text(encoding="utf-8").splitlines() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    added = tokenizer.add_tokens(tokens, special_tokens=True)
    ensure_dir(Path(args.save))
    tokenizer.save_pretrained(args.save)
    print(f"Added {added} tokens. Saved tokenizer to {args.save}")


# ---------------------------------------------------------------------------
# Dataset downloads (Common Voice, LibriSpeech)
# ---------------------------------------------------------------------------


def cmd_fetch_common_voice(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    import soundfile as sf

    dataset = load_dataset(args.version, args.subset, split=args.split)
    output_dir = Path(args.output)
    audio_dir = output_dir / "wav"
    manifest_path = output_dir / f"{args.split}.jsonl"
    ensure_dir(audio_dir)

    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(dataset):
            wav_path = audio_dir / f"{idx:08d}.wav"
            sf.write(wav_path, item["audio"]["array"], item["audio"]["sampling_rate"], subtype="PCM_16")
            record = {
                "path": str(wav_path.resolve()),
                "text": item.get("sentence", ""),
                "lang": args.subset,
                "speaker_id": item.get("client_id", "unknown"),
                "dataset": "common_voice",
                "split": args.split,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved Common Voice subset to {output_dir}")


def cmd_fetch_librispeech(args: argparse.Namespace) -> None:
    import tarfile
    from urllib.request import urlretrieve

    urls = {
        "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    }
    if args.subset not in urls:
        raise ValueError(f"Unsupported subset '{args.subset}'")

    output_dir = Path(args.output)
    ensure_dir(output_dir)
    archive = output_dir / f"{args.subset}.tar.gz"
    if not archive.exists():
        print(f"Downloading {args.subset}...")
        urlretrieve(urls[args.subset], archive)
    print(f"Extracting {archive}...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=output_dir)
    print(f"LibriSpeech subset available at {output_dir}")


# ---------------------------------------------------------------------------
# Manifest + feature helpers
# ---------------------------------------------------------------------------


def validate_audio(path: Path, target_sr: int = 16000) -> Dict[str, float]:
    import soundfile as sf

    data, sr = sf.read(path)
    if sr != target_sr:
        raise ValueError(f"Expected sample rate {target_sr}, got {sr} for {path}")
    return {"duration": len(data) / sr, "sample_rate": sr}


def pick_field(example: Dict, keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        if key in example and example[key]:
            value = example[key]
            if isinstance(value, (list, tuple)):
                value = value[0]
            return str(value)
    return default


def cmd_make_manifest(args: argparse.Namespace) -> None:
    source = Path(args.input)
    target = Path(args.output)
    ensure_dir(target.parent)

    with source.open("r", encoding="utf-8") as fin, target.open("w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            audio_path = Path(record["path"])
            meta = validate_audio(audio_path)
            record.update({
                "lang": record.get("lang", args.lang),
                "dataset": args.dataset,
                "speaker_id": record.get("speaker_id", "unknown"),
            })
            record.update(meta)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Normalized manifest written to {target}")


def cmd_make_mels(args: argparse.Namespace) -> None:
    import json
    import torch
    import torchaudio

    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    def compute_log_mel(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=sample_rate,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            window_type="hamming",
            dither=0.0,
        )

    with manifest.open("r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            waveform, sr = torchaudio.load(entry["path"])
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                sr = 16000
            mel = compute_log_mel(waveform, sr)
            uid = Path(entry["path"]).stem
            torch.save(mel, output_dir / f"{uid}.pt")
    print(f"Mel features saved to {output_dir}")


def cmd_make_codes(args: argparse.Namespace) -> None:
    import json
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    try:
        from voila_tokenizer import VoilaTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("voila_tokenizer package is required for quantization") from exc

    manifest_path = Path(args.manifest)
    tokenizer_path = Path(args.tokenizer)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    tokenizer = VoilaTokenizer.from_pretrained(str(tokenizer_path))

    with manifest_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            audio, sr = sf.read(entry["path"])
            if hasattr(tokenizer, "sample_rate") and sr != tokenizer.sample_rate:
                import torchaudio

                tensor = torchaudio.functional.resample(
                    torch.from_numpy(audio).unsqueeze(0), sr, tokenizer.sample_rate  # type: ignore[name-defined]
                )
                audio = tensor.squeeze(0).numpy()
                sr = tokenizer.sample_rate
            codes = tokenizer.encode(audio)
            uid = Path(entry["path"]).stem
            if isinstance(codes, tuple):
                np.savez_compressed(output_dir / f"{uid}.npz", codes=codes[0])
            else:
                np.savez_compressed(output_dir / f"{uid}.npz", codes=codes)
    print(f"Quantized codes saved to {output_dir}")


def cmd_make_seq(args: argparse.Namespace) -> None:
    import json
    import numpy as np

    SPECIAL = {
        "audio_start": "<AUDIO_IN>",
        "audio_end": "<AUDIO_IN_END>",
        "assistant_audio_start": "<ASSISTANT_AUDIO>",
        "assistant_audio_end": "</ASSISTANT_AUDIO>",
        "assistant_text_start": "<ASSISTANT_TEXT>",
        "assistant_text_end": "</ASSISTANT_TEXT>",
    }

    codes_dir = Path(args.codes_dir)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    with manifest_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            uid = Path(entry["path"]).stem
            code_path = codes_dir / f"{uid}.npz"
            if not code_path.exists():
                raise FileNotFoundError(f"Missing codebook file {code_path}")
            codes = np.load(code_path)["codes"].tolist()
            sequence = {
                "input_ids": [SPECIAL["audio_start"], SPECIAL["audio_end"]],
                "audio_codes": codes,
                "loss_mask": [0, 0],
                "meta": {
                    "text": entry.get("text", ""),
                    "lang": entry.get("lang", "unknown"),
                    "dataset": entry.get("dataset", "unknown"),
                },
            }
            fout.write(json.dumps(sequence, ensure_ascii=False) + "\n")
    print(f"Sequences written to {output_path}")


# ---------------------------------------------------------------------------
# Synthetic barge-in data
# ---------------------------------------------------------------------------


def cmd_make_barge(args: argparse.Namespace) -> None:
    import json
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    fg_files = list(Path(args.foreground).glob("*.wav"))
    bg_files = list(Path(args.background).glob("*.wav"))
    if not fg_files or not bg_files:
        raise ValueError("Foreground or background directory is empty")

    output_dir = Path(args.output)
    audio_dir = output_dir / "wav"
    manifest_path = output_dir / "barge_in.jsonl"
    ensure_dir(audio_dir)

    rng = np.random.default_rng(args.seed)
    with manifest_path.open("w", encoding="utf-8") as fout:
        for idx in range(args.count):
            fg = rng.choice(fg_files)
            bg = rng.choice(bg_files)
            fg_audio, sr = sf.read(fg)
            bg_audio, _ = sf.read(bg)
            length = min(len(fg_audio), len(bg_audio))
            fg_audio = fg_audio[:length]
            bg_audio = bg_audio[:length]
            fg_power = np.mean(fg_audio ** 2)
            bg_power = np.mean(bg_audio ** 2)
            snr = rng.uniform(args.snr_min, args.snr_max)
            scale = math.sqrt(fg_power / (bg_power * (10 ** (snr / 10)))) if bg_power > 0 else 0.0
            mixed = fg_audio + scale * bg_audio
            wav_path = audio_dir / f"mix_{idx:06d}.wav"
            sf.write(wav_path, mixed, sr, subtype="PCM_16")
            record = {
                "path": str(wav_path.resolve()),
                "text": "",
                "lang": "multi",
                "dataset": "duplex_synthetic",
                "split": "train",
                "label": "barge_in",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated barge-in set at {output_dir}")


# ---------------------------------------------------------------------------
# Tiny zh+en recipe
# ---------------------------------------------------------------------------


RECIPES = [
    {
        "name": "common_voice_zh_tw",
        "hf_id": "mozilla-foundation/common_voice_17_0",
        "config": "zh-TW",
        "split": "train",
        "target_hours": 120,
        "lang": "zh-TW",
        "type": "asr",
        "text_keys": ["sentence", "text", "transcription"],
        "speaker_keys": ["client_id", "speaker", "speaker_id"],
    },
    {
        "name": "common_voice_en",
        "hf_id": "mozilla-foundation/common_voice_17_0",
        "config": "en",
        "split": "train",
        "target_hours": 120,
        "lang": "en",
        "type": "asr",
        "text_keys": ["sentence", "text", "transcription"],
        "speaker_keys": ["client_id", "speaker", "speaker_id"],
    },
    {
        "name": "aishell_3",
        "hf_id": "AISHELL/AISHELL-3",
        "config": None,
        "split": "train",
        "target_hours": 40,
        "lang": "zh",
        "type": "tts",
        "text_keys": ["text", "transcription", "sentence"],
        "speaker_keys": ["speaker", "speaker_id", "speaker_name"],
    },
    {
        "name": "vctk",
        "hf_id": "badayvedat/VCTK",
        "config": None,
        "split": "train",
        "target_hours": 40,
        "lang": "en",
        "type": "tts",
        "text_keys": ["text", "transcription", "sentence"],
        "speaker_keys": ["speaker", "speaker_id"],
    },
]


@dataclass
class FeaturePipelines:
    extract_mel: bool
    quantize: bool
    mel_root: Path
    code_root: Path
    mel_bins: int
    frame_length: int
    frame_shift: int
    tokenizer_path: Optional[Path]


def cmd_make_tiny(args: argparse.Namespace) -> None:
    from datasets import Dataset, load_dataset
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    if args.quantize:
        try:
            from voila_tokenizer import VoilaTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("voila_tokenizer package is required for quantization") from exc
        tokenizer = VoilaTokenizer.from_pretrained(str(Path(args.voila_tokenizer)))
    else:
        tokenizer = None

    feature_cfg = FeaturePipelines(
        extract_mel=not args.no_mel,
        quantize=args.quantize,
        mel_root=Path(args.mel_root),
        code_root=Path(args.codes_root),
        mel_bins=args.mel_bins,
        frame_length=args.frame_length,
        frame_shift=args.frame_shift,
        tokenizer_path=Path(args.voila_tokenizer) if args.quantize else None,
    )

    rng = random.Random(args.seed)
    datasets_root = Path(args.datasets_root)
    manifests_root = Path(args.manifests_root)
    mel_root = feature_cfg.mel_root
    code_root = feature_cfg.code_root
    ensure_dir(datasets_root)
    ensure_dir(manifests_root)
    if feature_cfg.extract_mel:
        ensure_dir(mel_root)
    if feature_cfg.quantize:
        ensure_dir(code_root)

    def to_waveform(array: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        tensor = torch.from_numpy(array).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if sr != target_sr:
            tensor = torchaudio.functional.resample(tensor, sr, target_sr)
            sr = target_sr
        return tensor, sr

    def compute_log_mel(waveform: torch.Tensor, sr: int, cfg: FeaturePipelines) -> torch.Tensor:
        return torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=sr,
            num_mel_bins=cfg.mel_bins,
            frame_length=cfg.frame_length,
            frame_shift=cfg.frame_shift,
            window_type="hamming",
            dither=0.0,
        )

    all_asr: List[Dict] = []
    all_tts: List[Dict] = []

    for recipe in RECIPES:
        dataset: Dataset = load_dataset(recipe["hf_id"], recipe.get("config"), split=recipe.get("split", "train"))
        dataset = dataset.shuffle(seed=args.seed)

        target_seconds = (args.override_hours or recipe.get("target_hours", math.inf)) * 3600
        total_seconds = 0.0

        dataset_dir = datasets_root / recipe["name"] / recipe.get("split", "train")
        ensure_dir(dataset_dir)

        mel_dir = mel_root / recipe["name"] if feature_cfg.extract_mel else None
        code_dir = code_root / recipe["name"] if feature_cfg.quantize else None
        if mel_dir:
            ensure_dir(mel_dir)
        if code_dir:
            ensure_dir(code_dir)

        manifest_entries: List[Dict] = []

        for idx, example in enumerate(dataset):
            if args.limit_items and idx >= args.limit_items:
                break
            if total_seconds >= target_seconds:
                break

            audio_field = example.get("audio") or example.get("speech")
            if audio_field is None:
                continue
            array = np.asarray(audio_field["array"], dtype=np.float32)
            sr = int(audio_field["sampling_rate"])
            waveform, sr = to_waveform(array, sr)
            duration = waveform.shape[-1] / sr
            total_seconds += duration

            speaker_id = pick_field(example, recipe.get("speaker_keys", []), default="unknown")
            text = pick_field(example, recipe.get("text_keys", []), default="")

            uid = f"{recipe['name']}_{idx:08d}"
            audio_path = dataset_dir / f"{uid}.wav"
            ensure_dir(audio_path.parent)
            sf.write(audio_path, waveform.squeeze(0).cpu().numpy(), sr, subtype="PCM_16")

            entry = {
                "uid": uid,
                "path": str(audio_path.resolve()),
                "text": text,
                "lang": recipe.get("lang", "unknown"),
                "speaker_id": speaker_id,
                "dataset": recipe["name"],
                "split": recipe.get("split", "train"),
                "type": recipe.get("type", "asr"),
                "duration": duration,
                "sample_rate": sr,
            }

            if feature_cfg.extract_mel and mel_dir is not None:
                mel = compute_log_mel(waveform, sr, feature_cfg)
                torch.save(mel, mel_dir / f"{uid}.pt")
                entry["mel_path"] = str((mel_dir / f"{uid}.pt").resolve())

            if feature_cfg.quantize and tokenizer is not None and code_dir is not None:
                codes = tokenizer.encode(waveform.squeeze(0).cpu().numpy())  # type: ignore[call-arg]
                if isinstance(codes, tuple):
                    np.savez_compressed(code_dir / f"{uid}.npz", codes=codes[0])
                else:
                    np.savez_compressed(code_dir / f"{uid}.npz", codes=codes)
                entry["codes_path"] = str((code_dir / f"{uid}.npz").resolve())

            manifest_entries.append(entry)

        manifest_path = manifests_root / f"{recipe['name']}.jsonl"
        ensure_dir(manifest_path.parent)
        with manifest_path.open("w", encoding="utf-8") as fout:
            for item in manifest_entries:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"Prepared {recipe['name']} -> {len(manifest_entries)} samples, {total_seconds/3600:.1f} h, manifest {manifest_path}"
        )

        if recipe["type"] == "asr":
            all_asr.extend(manifest_entries)
        else:
            all_tts.extend(manifest_entries)

    def write_aggregate(entries: List[Dict], path: Path) -> None:
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8") as fout:
            for item in entries:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote aggregate manifest {path} ({len(entries)} items)")

    write_aggregate(all_asr, manifests_root / "tiny_asr.jsonl")
    write_aggregate(all_tts, manifests_root / "tiny_tts.jsonl")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speech-LLM data/model utility CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch_models = sub.add_parser("fetch-models", help="Download Whisper, Qwen, Voila tokenizer")
    p_fetch_models.add_argument("--items", nargs="*", choices=list(DEFAULT_MODELS.keys()))
    p_fetch_models.add_argument("--revision", nargs="*")
    p_fetch_models.add_argument("--root", type=str, default=".")
    p_fetch_models.set_defaults(func=cmd_fetch_models)

    p_audio_tokens = sub.add_parser("gen-audio-tokens", help="Generate layered audio token list")
    p_audio_tokens.add_argument("--codes", type=int, default=8192)
    p_audio_tokens.add_argument("--output", type=str, default="tokenizer/audio_tokens.txt")
    p_audio_tokens.set_defaults(func=cmd_gen_audio_tokens)

    p_vocab = sub.add_parser("extend-vocab", help="Extend Qwen tokenizer with new tokens")
    p_vocab.add_argument("--base", type=str, required=True)
    p_vocab.add_argument("--tokens", type=str, required=True)
    p_vocab.add_argument("--save", type=str, required=True)
    p_vocab.set_defaults(func=cmd_extend_vocab)

    p_cv = sub.add_parser("fetch-common-voice", help="Download Common Voice subset")
    p_cv.add_argument("--version", type=str, default="mozilla-foundation/common_voice_17_0")
    p_cv.add_argument("--subset", type=str, default="zh-TW")
    p_cv.add_argument("--split", type=str, default="train")
    p_cv.add_argument("--output", type=str, required=True)
    p_cv.set_defaults(func=cmd_fetch_common_voice)

    p_ls = sub.add_parser("fetch-librispeech", help="Download LibriSpeech subset")
    p_ls.add_argument("--subset", type=str, default="train-clean-100")
    p_ls.add_argument("--output", type=str, required=True)
    p_ls.set_defaults(func=cmd_fetch_librispeech)

    p_manifest = sub.add_parser("make-manifest", help="Normalize manifest JSONL")
    p_manifest.add_argument("--input", type=str, required=True)
    p_manifest.add_argument("--output", type=str, required=True)
    p_manifest.add_argument("--lang", type=str, required=True)
    p_manifest.add_argument("--dataset", type=str, required=True)
    p_manifest.set_defaults(func=cmd_make_manifest)

    p_mels = sub.add_parser("make-mels", help="Extract log-mel features")
    p_mels.add_argument("--manifest", type=str, required=True)
    p_mels.add_argument("--output_dir", type=str, required=True)
    p_mels.set_defaults(func=cmd_make_mels)

    p_codes = sub.add_parser("make-codes", help="Quantize audio with Voila tokenizer")
    p_codes.add_argument("--manifest", type=str, required=True)
    p_codes.add_argument("--tokenizer", type=str, required=True)
    p_codes.add_argument("--output_dir", type=str, required=True)
    p_codes.set_defaults(func=cmd_make_codes)

    p_seq = sub.add_parser("make-seq", help="Assemble multimodal training sequences")
    p_seq.add_argument("--manifest", type=str, required=True)
    p_seq.add_argument("--codes_dir", type=str, required=True)
    p_seq.add_argument("--output", type=str, required=True)
    p_seq.set_defaults(func=cmd_make_seq)

    p_barge = sub.add_parser("make-barge", help="Synthesize barge-in data")
    p_barge.add_argument("--foreground", type=str, required=True)
    p_barge.add_argument("--background", type=str, required=True)
    p_barge.add_argument("--output", type=str, required=True)
    p_barge.add_argument("--count", type=int, default=1000)
    p_barge.add_argument("--snr-min", dest="snr_min", type=float, default=0.0)
    p_barge.add_argument("--snr-max", dest="snr_max", type=float, default=10.0)
    p_barge.add_argument("--seed", type=int, default=7)
    p_barge.set_defaults(func=cmd_make_barge)

    p_tiny = sub.add_parser("make-tiny", help="Download and preprocess tiny zh+en recipe")
    p_tiny.add_argument("--datasets-root", type=str, default="datasets")
    p_tiny.add_argument("--manifests-root", type=str, default="data/manifests")
    p_tiny.add_argument("--mel-root", type=str, default="data/processed/mels")
    p_tiny.add_argument("--codes-root", type=str, default="data/processed/voila_codes")
    p_tiny.add_argument("--voila-tokenizer", type=str, default="tokenizer/voila_tokenizer")
    p_tiny.add_argument("--seed", type=int, default=7)
    p_tiny.add_argument("--limit-items", type=int, default=None)
    p_tiny.add_argument("--override-hours", type=float, default=None)
    p_tiny.add_argument("--no-mel", action="store_true")
    p_tiny.add_argument("--quantize", action="store_true")
    p_tiny.add_argument("--mel-bins", type=int, default=80)
    p_tiny.add_argument("--frame-length", type=int, default=25)
    p_tiny.add_argument("--frame-shift", type=int, default=10)
    p_tiny.set_defaults(func=cmd_make_tiny)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

