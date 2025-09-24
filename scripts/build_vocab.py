import argparse
from pathlib import Path
from typing import List

from transformers import AutoTokenizer


def read_tokens(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def extend_tokenizer(base_model: str, new_tokens: List[str], save_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    print(f"Added {added} tokens. Saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend Qwen tokenizer with audio/control tokens")
    parser.add_argument("--base", type=str, required=True, help="Base tokenizer model name or path")
    parser.add_argument("--tokens", type=str, required=True, help="Text file with new tokens")
    parser.add_argument("--save", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    tokens = read_tokens(Path(args.tokens))
    extend_tokenizer(args.base, tokens, Path(args.save))


if __name__ == "__main__":
    main()
