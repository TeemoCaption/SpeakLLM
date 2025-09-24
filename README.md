# Speech-LLM Starter

This repository scaffolds a multilingual speech-language model pipeline that combines **Voila-Tokenizer**, **Whisper-Medium**, and **Qwen-7B** for unified ASR, TTS, and full-duplex dialogue.

## Layout

`
configs/             # YAML configs for model, data, and training stages
data/                # Processed features, manifests, and sequences
models/              # Downloaded backbone checkpoints (Whisper, Qwen)
tokenizer/           # Tokenizer assets and generated codebook tokens
scripts/             # Data utilities, training, inference, evaluation
logs/, checkpoints/  # Runtime artifacts (git-ignored)
datasets/            # Raw corpora downloaded by helper commands
`

## Environment

- Python 3.10+
- CUDA 12.1+, cuDNN 9+, PyTorch 2.3.1 (pip install torch==2.3.1+cu121 ...)
- Key packages: lash-attn==2.5.6, 	ransformers==4.43.3, datasets==2.20.0,
  ccelerate==0.33.0, itsandbytes==0.43.1, 	orchaudio==2.3.1, peft==0.11.1,
  soundfile, librosa, fmpeg-python, huggingface-hub

`
python -m venv .venv
. .venv/Scripts/activate        # PowerShell: .\.venv\Scripts\Activate.ps1
accelerate config --config_file configs/deepspeed.json
`

## Model & Tokenizer Downloads (automated)

```
python scripts/data_tools.py fetch-models --root .
```

This command fetches the required Hugging Face assets into the repo:
- openai/whisper-medium → models/whisper-medium/
- Qwen/Qwen2-7B → models/qwen2-7b/
- maitrix-org/Voila-Tokenizer → 	okenizer/voila_tokenizer/
{{ ... }}

## Tiny zh+en Recipe (automated download + preprocessing)

Reproduce the bilingual TINY recipe (Common Voice zh-TW/en, AISHELL-3, VCTK) and stage manifests/features:

```
python scripts/data_tools.py make-tiny \
    --datasets-root datasets \
    --manifests-root data/manifests \
    --mel-root data/processed/mels \
    --codes-root data/processed/voila_codes \
    --quantize --voila-tokenizer tokenizer/voila_tokenizer
{{ ... }}

## Manual Data Ingestion

For additional corpora (noise, large-scale ASR/TTS), drop raw audio under datasets/<dataset>/ and normalize manifests with:

```
python scripts/data_tools.py make-manifest --input path/to/raw_manifest.jsonl \
    --output data/manifests/my_dataset.jsonl --lang zh --dataset my_dataset
```

## Core Pipeline

{{ ... }}
6. **Streaming inference**: python scripts/stream_infer.py --checkpoint checkpoints/stage5.
7. **Evaluation & stress**: python scripts/eval_tools.py <asr|tts|qa|barge> ... and python scripts/loadtest.py .

## TODO

- Implement dataset readers/collators in scripts/train_core.py (create_dataloaders) and wire full model construction in  build_model.
- Flesh out scripts/data_tools.py make-seq with tokenizer IDs, attention masks, and loss weighting.
- Integrate CosyVoice2 synthetic TTS generation and duplex augmentation pipelines.
- Expand safety templates in configs/safety.yml per deployment requirements.

## Safety & Compliance

{{ ... }}
