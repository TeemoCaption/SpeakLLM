import argparse

PREFIXES = ["<L1_", "<L2_", "<L3_", "<L4_"]


def main():
    parser = argparse.ArgumentParser(description="Generate layered audio codebook tokens")
    parser.add_argument("--codes", type=int, default=8192)
    parser.add_argument("--output", type=str, default="tokenizer/audio_tokens.txt")
    args = parser.parse_args()

    with open(args.output, "w", encoding="utf-8") as fout:
        for prefix in PREFIXES:
            for idx in range(args.codes):
                fout.write(f"{prefix}{idx:04d}>\n")


if __name__ == "__main__":
    main()
