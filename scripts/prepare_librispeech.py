import argparse
import hashlib
import tarfile
from pathlib import Path

import requests

LIBRISPEECH_URLS = {
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with target.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def extract(archive: Path, output_dir: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download and extract LibriSpeech subset")
    parser.add_argument("--subset", type=str, default="train-clean-100")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    url = LIBRISPEECH_URLS[args.subset]
    archive = Path(args.output) / f"{args.subset}.tar.gz"
    if not archive.exists():
        download(url, archive)
    extract(archive, Path(args.output))


if __name__ == "__main__":
    main()
