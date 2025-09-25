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

## 完整訓練流程
1. 下載所有語料（ASR / TTS / 指令與混碼）：
   ```bash
   python scripts/data_download/download_asr_datasets.py --config configs/data/asr_zh.yaml
   python scripts/data_download/download_tts_datasets.py --config configs/data/tts_zh.yaml
   python scripts/data_download/download_instruction_datasets.py --config configs/data/mixed_cs.yaml
   ```
2. 生成資料清單與對齊資訊：
   ```bash
   python scripts/prepare/build_manifests.py --config configs/data/asr_zh.yaml --output data/manifests/asr_train.jsonl
   python scripts/prepare/build_manifests.py --config configs/data/tts_zh.yaml --output data/manifests/tts_train.jsonl
   python scripts/prepare/normalize_text_zh.py --manifest data/manifests/asr_train.jsonl --output data/manifests/asr_train_normalized.jsonl --conversion t2s
   python scripts/prepare/resample_split.py --manifest data/manifests/asr_train_normalized.jsonl --output_dir data/resampled --target_sr 16000 --vad True
   python scripts/tokenizer/extract_voila_tokens.py --manifest data/manifests/tts_train.jsonl --output data/manifests/tts_voila_codes.jsonl --checkpoint maitrix-org/Voila-Tokenizer
   ```
3. 進行四階段訓練（確保 `configs/train/*.yaml` 指向正確的 checkpoint 與清單）：
   ```bash
   bash scripts/train_phase0.sh configs/train/phase0_connector.yaml
   bash scripts/train_phase1.sh configs/train/phase1_sft.yaml
   bash scripts/train_phase2.sh configs/train/phase2_duplex.yaml
   bash scripts/train_phase3.sh configs/train/phase3_emotion.yaml
   ```
4. 匯出推論模型與測試（可選）：
   ```bash
   bash scripts/export_onnx.sh onnx_exports
   python -m runtime.server --config configs/infer/server.yaml
   python -m runtime.client_cli stream --audio examples/sample.wav
   ```
5. 記錄與監控：
   - 每階段輸出會寫入 `outputs/phase*`，可在 Weights & Biases (`wandb_project=voice-duplex-zh`) 上查看損失與延遲指標。
   - 使用 `tests/` 內測試在關鍵變更後驗證基礎功能：
     ```bash
     pytest tests
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
