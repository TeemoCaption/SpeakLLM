# Speech-LLM Starter

This repository scaffolds a multilingual speech-language model pipeline that combines **Voila-Tokenizer**, **Whisper-Medium**, and **Qwen-7B** for unified ASR, TTS, and full-duplex dialogue.


## 專案結構

```
SpeechLLM/
├── .vscode/
│   └── settings.json
├── checkpoints/
├── configs/
│   ├── data.yml
│   ├── deepspeed.json
│   ├── model.yml
│   ├── safety.yml
│   └── train.yml
├── datasets/
├── logs/
├── models/
│   ├── adapters/
│   ├── audio_decoder/
│   └── heads/
├── notebooks/
├── scripts/
│   ├── data_tools.py
│   ├── eval_tools.py
│   ├── loadtest.py
│   ├── synth_cosyvoice2.py
│   └── train_core.py
├── tokenizer/
│   ├── new_tokens.txt
│   └── voila_tokenizer/
├── .gitignore
├── README.md
└── requirements.txt
```

## Data Preparation
### 1. 下載語料 / 生成 manifest

- **下載 Common Voice 並生成 manifest**：`python scripts/data_tools.py fetch-common-voice --subset zh-TW --split train --output datasets/common_voice_zh_tw && python scripts/data_tools.py make-manifest --input datasets/common_voice_zh_tw/train.jsonl --output data/manifests/common_voice_zh_tw_train.jsonl --lang zh-TW --dataset common_voice_zh_tw`

- **下載 LibriSpeech 並生成 manifest**：`python scripts/data_tools.py fetch-librispeech --subset train-clean-100 --output datasets/librispeech && python scripts/data_tools.py make-manifest --input datasets/librispeech/train-clean-100.jsonl --output data/manifests/librispeech_train.jsonl --lang en --dataset librispeech`
# AISHELL-3 / VCTK manifest 亦可使用 make-manifest 生成
### 2. 產生 barge-in / 重疊語音

- **合成插話與重疊語音**：`python scripts/data_tools.py make-barge --foreground data/clean_speech --background data/noise_pool --output data/processed/duplex --count 10000 --snr-min 0 --snr-max 10`
指令會輸出 `data/processed/duplex/wav/` 及 `barge_in.jsonl`。將生成的 manifest 路徑填入 `configs/data.yml` 中的 `duplex_synth_zh` 或 `duplex_synth_en`，即可在訓練時混入 20–30% 插話樣本。

### 3. 特徵、量化與序列化

- **Whisper 對數梅爾特徵**：`python scripts/data_tools.py make-mels --manifest data/manifests/common_voice_zh_tw_train.jsonl --output_dir data/processed/mels/common_voice_zh_tw`
- **Voila 量化**：`python scripts/data_tools.py make-codes --manifest data/manifests/common_voice_zh_tw_train.jsonl --tokenizer tokenizer/voila_tokenizer --output_dir data/processed/voila_codes/common_voice_zh_tw`
- **序列組裝**：`python scripts/data_tools.py make-seq --manifest data/manifests/common_voice_zh_tw_train.jsonl --codes_dir data/processed/voila_codes/common_voice_zh_tw --output data/processed/sequences/common_voice_zh_tw.jsonl`

其他語料（如 `librispeech_train.jsonl`、`AISHELL-3`、`VCTK`）依樣操作，將輸出路徑填入 `configs/train.yml` 對應階段即可。

### 4. CosyVoice2 合成語音
裡提供 `train_manifests` 與 `eval_manifests` 陣列，項目結構：

- **manifest**：指向 JSONL 清單檔案絕對或相對路徑。
- **weight**：用於資料混合時的抽樣權重（浮點數，可選）。
- **type**：資料類型標籤（如 `asr`、`tts`、`spoken_dialog`，可選）。


- **text**：目標文字或回答內容。
- **prompt**：使用者輸入（若為單純 ASR 可留空）。
- **audio_codes / codes_path**：Voila-Tokenizer 產生的 4 層 RVQ 索引或其檔案路徑。
- **mel_path**：Whisper 對應的 log-mel 特徵檔案。
- **chunk_spans**：每段 `<AUDIO_CHUNK>` 在原始音訊內的起訖 frame（若缺少會自動依預設長度推估）。
- **interrupt_label**：是否中斷的標籤（Streaming 階段使用，可為 -100 表示忽略）。

若需覆寫欄位名稱，可在階段設定中加入 `data_fields` 對應鍵值。

## 合成語音（CosyVoice2 範例）

本專案提供腳本 `scripts/synth_cosyvoice2.py`，可利用 [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) 產生中英雙語語音。

- **批次合成中英語音**：`python scripts/synth_cosyvoice2.py --output data/synthetic/cosyvoice2 --zh-ref-dir /path/to/AISHELL-3/wav --en-ref-dir /path/to/VCTK/wav --zh-texts resources/zh_prompts.txt --en-texts resources/en_prompts.txt --zh-count 3000 --en-count 3000`

- **參考語者**：`--zh-ref-dir`、`--en-ref-dir` 指向現有資料集 wav 目錄（會隨機抽樣）。
- **語句**：可透過 `--zh-texts` / `--en-texts` 提供自備腳本，或改以 `--hf-dataset` 連結 Common Voice 等 HF corpus。
- **情緒與語速**：`--zh-emotions`、`--en-emotions`、`--rate-range` 可自訂候選。
- **輸出**：腳本生成 wav 檔與 `metadata.json`，可直接納入 `train_manifests`。

> **注意**：CosyVoice2 依賴需依模型卡指引自行安裝，可參考 `requirements.txt` 中的註記。

## TODO

- Flesh out scripts/data_tools.py make-seq with tokenizer IDs, attention masks, and loss weighting.
- Integrate CosyVoice2 synthetic TTS generation and duplex augmentation pipelines.
- Expand safety templates in configs/safety.yml per deployment requirements.

## Safety & Compliance

{{ ... }}
