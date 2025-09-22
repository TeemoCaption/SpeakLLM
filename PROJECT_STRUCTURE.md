# SpeechLLM 專案結構總覽

## 📁 完整目錄結構

```
SpeechLLM/
├── speechllm/                          # 核心程式碼模組
│   ├── __init__.py                     # 主模組初始化
│   ├── models/                         # 模型定義
│   │   ├── __init__.py
│   │   ├── speechllm.py               # 主 SpeechLLM 模型
│   │   ├── whisper_encoder.py         # Whisper 編碼器 + 投影
│   │   ├── qformer.py                 # Q-Former 聚合器
│   │   └── audio_transformer.py       # 音訊 Transformer（階層式 RVQ）
│   ├── codecs/                        # 音訊編解碼
│   │   ├── __init__.py
│   │   ├── rvq_codec.py              # RVQ 編解碼器
│   │   ├── audio_tokenizer.py        # 音訊 tokenizer
│   │   └── vocab_manager.py          # 詞彙表管理（含中文特殊 token）
│   ├── align/                         # 對齊和交錯
│   │   ├── __init__.py
│   │   ├── alignment.py              # 基礎對齊功能
│   │   ├── interleaving.py           # 基礎交錯生成
│   │   ├── chinese_alignment.py      # 中文語音-文字對齊
│   │   ├── chinese_interleaving.py   # 中文交錯序列生成
│   │   └── chinese_text_normalizer.py # 中文文字正規化
│   ├── data/                          # 資料處理
│   │   ├── __init__.py
│   │   ├── dataset.py                # 基礎資料集
│   │   └── chinese_dataset.py        # 中文資料集處理
│   ├── training/                      # 訓練相關
│   │   ├── __init__.py
│   │   ├── trainer.py                # 三階段訓練器
│   │   ├── loss.py                   # 多任務損失函數
│   │   └── optimizer.py              # 優化器和調度器
│   └── inference/                     # 推理相關
│       ├── __init__.py
│       └── engine.py                 # 推理引擎（含串流和全雙工）
├── scripts/                           # 執行腳本
│   ├── __init__.py
│   ├── train.py                      # 通用訓練腳本
│   ├── train_chinese.py              # 中文專用訓練腳本
│   ├── inference.py                  # 推理腳本
│   └── prepare_chinese_data.py       # 中文資料準備腳本
├── configs/                           # 配置文件
│   ├── __init__.py
│   └── default_config.yaml           # 預設配置（含中文優化）
├── examples/                          # 示例和教程
│   ├── chinese_training_example.py   # 中文訓練示例
│   ├── data/                         # 示例資料
│   └── outputs/                      # 示例輸出
├── docs/                             # 文檔
│   ├── CHINESE_GUIDE.md              # 中文使用指南
│   └── API_REFERENCE.md              # API 參考（待創建）
├── tests/                            # 測試文件（待創建）
├── requirements.txt                   # Python 依賴
├── README.md                         # 專案說明
├── PROJECT_STRUCTURE.md             # 本文件
└── LICENSE                           # 授權文件
```

## 🎯 核心模組功能

### 1. 模型架構 (`speechllm/models/`)

#### `speechllm.py` - 主模型
- **功能**：整合所有組件的主要 SpeechLLM 模型
- **特色**：
  - 支援四種任務模式（TITO/AITO/TIAO/AIAO）
  - LoRA 微調支援
  - 混合精度訓練
  - 中文詞彙表擴展

#### `whisper_encoder.py` - 語音編碼器
- **功能**：Whisper encoder + 投影層
- **中文優化**：
  - 使用 Whisper Medium 模型
  - 支援中文語音特徵提取
  - 可凍結預訓練權重

#### `qformer.py` - 特徵聚合器
- **功能**：Q-Former 將時間序列特徵聚合成 LLM token
- **特色**：
  - DiVA 風格初始化
  - 交叉注意力機制
  - 可配置查詢 token 數量

#### `audio_transformer.py` - 音訊生成器
- **功能**：從 LLM 隱狀態生成 RVQ token
- **中文優化**：
  - 階層式生成（L1 語義 → L2-L4 聲學）
  - 多尺度解碼策略
  - 支援並行和序列生成

### 2. 音訊處理 (`speechllm/codecs/`)

#### `rvq_codec.py` - RVQ 編解碼器
- **功能**：4 層殘差向量量化
- **特色**：
  - 語義層（L1）+ 聲學層（L2-L4）
  - 可配置代碼簿大小
  - 支援多種音訊格式

#### `audio_tokenizer.py` - 音訊 Token 化
- **功能**：音訊與文字的統一 token 化
- **中文支援**：
  - 中文聊天格式生成
  - RVQ token 與文字 token 對齊
  - 支援交錯序列

#### `vocab_manager.py` - 詞彙表管理
- **功能**：管理特殊 token 和詞彙表擴展
- **中文優化**：
  - Qwen tokenizer 擴展
  - 中文特殊 token 支援
  - RVQ 代碼簿 token

### 3. 中文對齊系統 (`speechllm/align/`)

#### `chinese_alignment.py` - 中文對齊
- **功能**：中文語音-文字精確對齊
- **特色**：
  - 拼音音節級對齊
  - Whisper 時間戳提取
  - DTW 動態時間規劃
  - 繁簡轉換支援

#### `chinese_interleaving.py` - 中文交錯生成
- **功能**：生成中文語音-文字交錯序列
- **特色**：
  - 音節級交錯標註
  - 4 層 RVQ 對齊策略
  - 支援多種對齊單位

#### `chinese_text_normalizer.py` - 文字正規化
- **功能**：中文文字預處理和正規化
- **特色**：
  - 數字正規化（一千二 → 1200）
  - 量詞統一化
  - 多口音處理
  - 英文詞彙保留

### 4. 資料處理 (`speechllm/data/`)

#### `chinese_dataset.py` - 中文資料集
- **功能**：處理中文語音資料集
- **支援資料集**：
  - AISHELL-1/2/4
  - WenetSpeech
  - Common Voice zh-CN/zh-TW
  - THCHS-30
  - MagicData

### 5. 訓練系統 (`speechllm/training/`)

#### `trainer.py` - 三階段訓練器
- **Stage A**：中文輸入對齊 + KL 蒸餾
- **Stage B**：中文語音生成訓練
- **Stage C**：四種模式聯合訓練
- **中文優化**：
  - 階層式 RVQ 損失權重
  - 多口音混合訓練
  - 只對 `<assistant>` 段計算 loss

#### `loss.py` - 損失函數
- **多任務損失**：文字 + RVQ + 對齊 + KL
- **中文優化**：L1 語義層權重更高

#### `optimizer.py` - 優化策略
- **功能**：差異化學習率和調度策略
- **特色**：支援不同組件的學習率設置

### 6. 推理引擎 (`speechllm/inference/`)

#### `engine.py` - 推理引擎
- **功能**：完整的推理和對話系統
- **特色**：
  - 四種任務模式推理
  - 串流音訊處理
  - 全雙工對話支援
  - VAD 語音活動檢測

## 🚀 執行腳本功能

### `scripts/train_chinese.py` - 中文訓練
- **功能**：專門針對中文的訓練腳本
- **特色**：
  - 中文組件初始化
  - 三階段訓練流程
  - 中文資料集載入
  - 訓練監控和保存

### `scripts/prepare_chinese_data.py` - 資料準備
- **功能**：中文資料集下載和預處理
- **支援操作**：
  - 資料集下載指南
  - 格式轉換和處理
  - 混合資料集創建
  - 資料驗證

### `scripts/inference.py` - 推理腳本
- **功能**：模型推理和測試
- **模式**：
  - 單輪推理
  - 互動式聊天
  - 串流推理
  - 性能測試

## ⚙️ 配置系統

### `configs/default_config.yaml`
- **模型配置**：LLM + Whisper + Q-Former + Audio Transformer
- **中文配置**：語言設置、對齊策略、文字正規化
- **訓練配置**：三階段參數、損失權重、硬體設置
- **資料配置**：資料集路徑、處理參數、快取設置

## 📚 文檔和示例

### `docs/CHINESE_GUIDE.md`
- **完整中文使用指南**
- **涵蓋內容**：
  - 安裝和環境設置
  - 資料集準備
  - 訓練流程
  - 推理使用
  - 性能優化
  - 常見問題

### `examples/chinese_training_example.py`
- **中文訓練完整示例**
- **演示功能**：
  - 組件初始化
  - 資料處理
  - 模型訓練
  - 推理測試

## 🔧 關鍵設計原則

### 1. 模組化設計
- 每個功能獨立模組
- 清晰的介面定義
- 易於擴展和維護

### 2. 中文優先
- 專門針對中文語音對話優化
- 支援繁簡轉換和多口音
- 拼音音節級精確對齊

### 3. 三階段訓練
- Stage A：聽懂中文
- Stage B：會說中文  
- Stage C：流暢對話

### 4. 多任務統一
- TITO/AITO/TIAO/AIAO 四種模式
- 統一的模型架構
- 靈活的推理介面

### 5. 生產就緒
- 完整的訓練和推理流程
- 詳細的文檔和示例
- 性能優化和錯誤處理

## 🎯 使用流程

### 1. 環境準備
```bash
pip install -r requirements.txt
```

### 2. 資料準備
```bash
python scripts/prepare_chinese_data.py --action all
```

### 3. 模型訓練
```bash
python scripts/train_chinese.py --config configs/default_config.yaml
```

### 4. 模型推理
```bash
python scripts/inference.py --model_path ./outputs/final_model --mode interactive
```

### 5. 自定義開發
- 參考 `examples/chinese_training_example.py`
- 閱讀 `docs/CHINESE_GUIDE.md`
- 根據需求修改配置和代碼

這個專案結構設計確保了代碼的清晰性、可維護性和擴展性，特別針對中文語音對話場景進行了全面優化。
