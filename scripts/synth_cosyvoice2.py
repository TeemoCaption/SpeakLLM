"""CosyVoice2 語音合成資料腳手架。

此腳本示範如何使用 FunAudioLLM/CosyVoice2-0.5B 產生中英混合的語音資料，
並輸出對應的 metadata JSON。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CosyVoice2 合成資料產生器")
    parser.add_argument("--output", type=Path, required=True, help="輸出資料夾，會包含 wav 與 metadata.json")
    parser.add_argument("--zh-ref-dir", type=Path, required=True, help="中文參考語者 wav 目錄 (AISHELL-3 等)")
    parser.add_argument("--en-ref-dir", type=Path, required=True, help="英文參考語者 wav 目錄 (VCTK 等)")
    parser.add_argument("--zh-texts", type=Path, default=None, help="中文語句文字檔，每行一筆")
    parser.add_argument("--en-texts", type=Path, default=None, help="英文語句文字檔，每行一筆")
    parser.add_argument("--zh-count", type=int, default=3000, help="生成中文語句數量")
    parser.add_argument("--en-count", type=int, default=3000, help="生成英文語句數量")
    parser.add_argument("--zh-emotions", type=str, default="neutral,happy,sad", help="中文情緒隨機候選，逗號分隔")
    parser.add_argument("--en-emotions", type=str, default="neutral,friendly", help="英文情緒隨機候選，逗號分隔")
    parser.add_argument("--rate-range", type=str, default="0.95,1.05", help="語速隨機範圍，格式 min,max")
    parser.add_argument("--seed", type=int, default=7, help="隨機種子")
    parser.add_argument("--cosy-checkpoint", type=str, default="FunAudioLLM/CosyVoice2-0.5B", help="Hugging Face CosyVoice2 checkpoint")
    parser.add_argument("--hf-dataset", type=str, default=None, help="可選 Hugging Face 資料集 (例如 mozilla-foundation/common_voice_17_0)")
    parser.add_argument("--hf-lang", type=str, default="zh-TW", help="使用 HF 資料集時的語言子設定 (若適用)")
    parser.add_argument("--hf-split", type=str, default="train", help="使用 HF 資料集時的 split")
    return parser.parse_args()


def read_lines(path: Optional[Path]) -> List[str]:
    """讀取文字檔，若路徑為 None 則回傳預設樣本。"""

    if path is None:
        return []
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return texts


def gather_refs(root: Path) -> List[Path]:
    """蒐集參考語者 wav 檔案。"""

    if not root.exists():
        raise FileNotFoundError(f"找不到參考語者目錄 {root}")
    files = sorted(root.glob("**/*.wav"))
    if not files:
        raise ValueError(f"{root} 底下沒有找到任何 wav 檔")
    return files


def extend_texts_with_hf(texts: List[str], dataset_id: Optional[str], lang: str, split: str) -> List[str]:
    """若指定 Hugging Face 資料集則附加語句。"""

    if dataset_id is None:
        return texts
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 datasets 套件，請先安裝 pip install datasets") from exc

    ds = load_dataset(dataset_id, lang, split=split)
    key_candidates = ["sentence", "text", "transcription"]
    for row in ds:
        for key in key_candidates:
            if key in row and row[key]:
                texts.append(str(row[key]))
                break
    return texts


def ensure_texts(seed: int, base_texts: List[str], fallback: Iterable[str], target: int) -> List[str]:
    """確保有足夠語句供抽樣，不足時使用預設樣本補齊。"""

    rng = random.Random(seed)
    items = list(base_texts)
    if not items:
        items = list(fallback)
    if not items:
        raise ValueError("沒有可用的語句來源")
    if len(items) >= target:
        return rng.sample(items, target)
    repeated = items * (target // len(items) + 1)
    return rng.sample(repeated, target)


def parse_rate_range(expr: str) -> tuple[float, float]:
    """解析語速範圍字串。"""

    try:
        lower_str, upper_str = expr.split(",")
        lower = float(lower_str)
        upper = float(upper_str)
    except ValueError as exc:
        raise ValueError("rate-range 需為 min,max 形式") from exc
    if lower <= 0 or upper <= 0 or lower > upper:
        raise ValueError("語速範圍需為正數且下限不可大於上限")
    return lower, upper


def build_tts(checkpoint: str):
    """載入 CosyVoice2 模型。"""

    try:
        from cosyvoice2 import CosyVoiceTTS  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "需要 cosyvoice2 套件與相關依賴，請依照 FunAudioLLM/CosyVoice2-0.5B 模型卡指示安裝"
        ) from exc
    return CosyVoiceTTS.from_pretrained(checkpoint)


def synthesize_batch(
    tts,
    texts: List[str],
    refs: List[Path],
    lang: str,
    emotions: List[str],
    rate_range: tuple[float, float],
    output_dir: Path,
    meta: List[dict],
    rng: random.Random,
) -> None:
    """產生一批語音並記錄中繼資料。"""

    for text in texts:
        ref = rng.choice(refs)
        rate = rng.uniform(*rate_range)
        emotion = rng.choice(emotions)
        filename = output_dir / f"{lang}_{rng.randrange(10**12)}.wav"
        tts.save_wav(
            text=text,
            language=lang,
            ref_wav=str(ref),
            emotion=emotion,
            rate=f"{rate:.2f}",
            file_path=str(filename),
        )
        meta.append(
            {
                "path": str(filename.resolve()),
                "text": text,
                "lang": lang,
                "speaker_ref": str(ref.resolve()),
                "emo": emotion,
                "rate": f"{rate:.2f}",
            }
        )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    zh_texts = read_lines(args.zh_texts)
    en_texts = read_lines(args.en_texts)

    zh_texts = extend_texts_with_hf(zh_texts, args.hf_dataset, args.hf_lang, args.hf_split) if args.hf_lang.startswith("zh") else zh_texts
    en_texts = extend_texts_with_hf(en_texts, args.hf_dataset, args.hf_lang, args.hf_split) if args.hf_lang.startswith("en") else en_texts

    zh_defaults = ["請用溫柔語氣說：早安。", "幫我總結這段話，保持口語化風格。", "描述一下今天的天氣並提醒攜帶雨具。"]
    en_defaults = ["Good morning!", "Summarize the following paragraph in two sentences.", "Politely remind the listener about the upcoming meeting."]

    zh_samples = ensure_texts(args.seed, zh_texts, zh_defaults, args.zh_count)
    en_samples = ensure_texts(args.seed + 1, en_texts, en_defaults, args.en_count)

    zh_refs = gather_refs(args.zh_ref_dir)
    en_refs = gather_refs(args.en_ref_dir)

    zh_emotions = [item.strip() for item in args.zh_emotions.split(",") if item.strip()]
    en_emotions = [item.strip() for item in args.en_emotions.split(",") if item.strip()]
    if not zh_emotions or not en_emotions:
        raise ValueError("情緒列表不可為空")

    rate_range = parse_rate_range(args.rate_range)
    tts = build_tts(args.cosy_checkpoint)

    metadata: List[dict] = []
    synthesize_batch(tts, zh_samples, zh_refs, "zh", zh_emotions, rate_range, args.output, metadata, rng)
    synthesize_batch(tts, en_samples, en_refs, "en", en_emotions, rate_range, args.output, metadata, rng)

    meta_path = args.output / "metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成生成 {len(metadata)} 筆語音，metadata 已寫入 {meta_path}")


if __name__ == "__main__":
    main()
