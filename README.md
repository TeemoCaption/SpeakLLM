# SpeakLLM 中文優先全雙工語音大模型專案

## 專案簡介
- **目標**：打造以 `Whisper Medium`、`Voila-Tokenizer`、`Qwen2.5-7B-Instruct`、`CosyVoice2` 為核心的中文優先、支援英文、全雙工語音 LLM。
- **核心特性**：中文語義優先、Dual-Scale 規劃、Tone-Aware 交錯對齊、Overlap-Aware Loss、code-switch 穩健化、低延遲串流推論。
- **產出**：完整的資料管線、三階段訓練策略、串流推論設計與關鍵超參數建議。

## 目錄結構
```text
.
├── README.md
├── configs/
│   ├── data/
│   │   ├── asr_zh.yaml
│   │   ├── tts_zh.yaml
│   │   └── mixed_cs.yaml
│   ├── train/
│   │   ├── phase0_connector.yaml
│   │   ├── phase1_sft.yaml
│   │   ├── phase2_duplex.yaml
│   │   └── phase3_emotion.yaml
│   └── infer/
│       └── server.yaml
├── docs/
│   ├── architecture.md
│   ├── research_modules.md
│   ├── datasets.md
│   ├── data_processing.md
│   ├── training_schedule.md
│   ├── inference_system.md
│   └── roadmap.md
├── env/
│   ├── environment.yml
│   └── docker/
│       ├── Dockerfile
│       └── entrypoint.sh
├── scripts/
│   ├── data_download/
│   │   ├── download_asr_datasets.py
│   │   ├── download_tts_datasets.py
│   │   └── download_instruction_datasets.py
│   ├── prepare/
│   │   ├── build_manifests.py
│   │   ├── normalize_text_zh.py
│   │   └── resample_split.py
│   ├── tokenizer/
│   │   └── extract_voila_tokens.py
│   ├── export_onnx.sh
│   ├── train_phase0.sh
│   ├── train_phase1.sh
│   ├── train_phase2.sh
│   └── train_phase3.sh
├── src/
│   └── ...  # Whisper connector, Qwen loader, CosyVoice streamer 等核心模組
├── tests/
│   ├── test_streaming.py
│   ├── test_barge_in.py
│   └── test_metrics.py
└── proto/
    └── duplex.proto
```

## 快速開始
- 建立 Conda 環境：
  ```bash
  conda env create -f env/environment.yml && conda activate voice-duplex-zh
  ```
- 或使用 Pip 安裝：
  ```bash
  pip install -r requirements.txt
  ```
- 複製環境變數樣板並填入 `HF_TOKEN`：
  ```bash
  copy .env.example .env
  ```
- PowerShell 可改用：
  ```powershell
  Copy-Item .env.example .env
  ```
- 下載 ASR 語料：
  ```bash
  python scripts/data_download/download_asr_datasets.py --config configs/data/asr_zh.yaml
  ```
- 下載 TTS 語料：
  ```bash
  python scripts/data_download/download_tts_datasets.py --config configs/data/tts_zh.yaml
  ```
- 產生訓練清單：
  ```bash
  python scripts/prepare/build_manifests.py --config configs/data/asr_zh.yaml --output data/manifests/asr_train.jsonl
  ```
- 啟動 Phase 0 訓練：
  ```bash
  bash scripts/train_phase0.sh configs/train/phase0_connector.yaml
  ```
- 啟動測試伺服器：
  ```bash
  python -m runtime.server
  ```

## 文件導覽
- **`docs/architecture.md`**：系統總體架構與模組說明。
- **`docs/research_modules.md`**：Dual-Scale、Tone-Aware、Overlap-Aware、code-switch 等新方法詳述。
- **`docs/datasets.md`**：資料集來源、授權、載入方式與使用場景。
- **`docs/data_processing.md`**：資料清理、切分、增強與對齊流程。
- **`docs/training_schedule.md`**：Phase 0–3 詳細訓練步驟、損失、評估指標。
- **`docs/inference_system.md`**：推論管線、搶話機制、串流工程設計。
- **`docs/roadmap.md`**：里程碑、風險評估與部署建議。

## 主要依賴與工具
- **深度學習框架**：PyTorch、transformers、peft。
- **資料處理**：torchaudio、datasets、pypinyin/g2pC、webrtcvad。
- **語音模型**：`maitrix-org/Voila-Tokenizer`、`whisper-medium`、`Qwen2.5-7B-Instruct`、`CosyVoice2`。

## 貢獻指南
- **分支流程**：feature 分支開發 → pull request → code review → merge。
- **風格要求**：遵守 PEP8，註解使用繁體中文且不加數字標號，文件使用 Markdown。
- **測試建議**：提供單元測試與端到端驗證腳本，特別關注延遲、語音品質與搶話行為。

## 授權說明
- **專案**：MIT License。
- **資料集**：請遵守各資料集原始授權（如 `WenetSpeech` 學術非商、`Common Voice` CC0、`CoVoST2` CC-BY-NC-4.0）。
