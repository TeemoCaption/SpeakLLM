# Speech-LLM Starter

This repository scaffolds a multilingual speech-language model pipeline that combines **Voila-Tokenizer**, **Whisper-Medium**, and **Qwen-7B** for unified ASR, TTS, and full-duplex dialogue.

## Layout

`
speech-llm/
  configs/             # YAML configs for data, model, and multi-stage training
  data/                # Placeholder directories for raw/processed audio and manifests
  tokenizer/           # Tokenizer assets and generated codebook tokens
  models/              # Adapter/head checkpoints and audio decoder outputs
  scripts/             # Data prep, preprocessing, training, inference, and evaluation scripts
  logs/, checkpoints/  # Runtime artifacts (git-ignored)
`

## Environment

- Python 3.10+
- CUDA 12.1+, cuDNN 9+, PyTorch 2.3.1 (pip install torch==2.3.1+cu121 ...)
- lash-attn==2.5.6, 	ransformers==4.43.3, datasets==2.20.0, ccelerate==0.33.0,
  itsandbytes==0.43.1, 	orchaudio==2.3.1, peft==0.11.1, fmpeg-python, soundfile, jiwer, librosa, equests

`
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt  # populate this file with the packages above
accelerate config --config_file speech-llm/configs/deepspeed_zero3_bf16.json
`

## Quick Start

1. **Download data** using scripts in speech-llm/scripts/prepare_*.py, then normalize manifests with prepare_manifest.py.
2. **Extract features** (extract_features.py) and **quantize audio** (quantize_audio.py).
3. **Extend tokenizer**: generate audio tokens (generate_audio_tokens.py), append to 	okenizer/new_tokens.txt, then run uild_vocab.py.
4. **Stage-wise training** with scripts/train.py and stage configs (	rain_stage1.yaml .. 	rain_stage6.yaml).
5. **Audio decoder** training (	rain_audio_decoder.py) if you do not use Voila’s pretrained vocoder.
6. **Streaming inference** via scripts/infer_streaming.py after Stage 5/6 checkpoints.
7. **Evaluation & load testing** using eval_*.py and load_test.py.

Each stage config references the previous checkpoint in esume_from. Adjust paths once actual checkpoints exist.

## TODO

- Implement dataset readers and collators (create_dataloaders) and the unified model wiring in scripts/train.py.
- Flesh out uild_sequence.py to construct proper multimodal sequences with tokenizer vocab ids.
- Add real evaluation harnesses once model components are complete.
- Integrate logging (W&B/TensorBoard) and mixed precision rules based on hardware.

## Safety & Compliance

safety_rules.yaml defines refusal triggers and fallback templates for Stage 6. You can extend it with organization-specific policies before training the safety head.
