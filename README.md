# SpeechLLM

基於 Qwen-2/2.5 和 Whisper encoder 的多模態語音大語言模型，支援四種任務模式：TITO（文字到文字）、AITO（音訊到文字）、TIAO（文字到音訊）、AIAO（音訊到音訊）。

## 🌟 主要特性

- **多模態對話**：支援文字和語音的雙向交互
- **四種任務模式**：TITO、AITO、TIAO、AIAO 統一框架
- **交錯序列生成**：實現 Voila 風格的語音-文字交錯標註
- **三階段訓練**：輸入對齊、語音輸出、多任務聯訓
- **RVQ 音訊編碼**：4 層殘差向量量化，語義與聲學分離
- **串流推理**：支援實時語音對話
- **全雙工對話**：支援打斷和並行處理

## 🏗️ 架構設計

### 模型架構
```
輸入端：Whisper Encoder → Q-Former → LLM (Qwen)
輸出端：LLM → Audio Transformer → RVQ Tokens → Audio
```

### 關鍵組件
- **Whisper Encoder**：語音特徵提取
- **Q-Former**：將時間序列特徵聚合成語義 token
- **Qwen LLM**：核心語言理解和生成
- **Audio Transformer**：多尺度 RVQ token 生成
- **RVQ Codec**：4 層殘差向量量化音訊編解碼

## 📦 安裝

### 環境要求
- Python 3.10+
- PyTorch 2.4+
- CUDA 12+
- GPU：A100/H100 (80GB) 或 2-4× 40GB

### 安裝步驟
```bash
# 克隆專案
git clone https://github.com/your-username/SpeechLLM.git
cd SpeechLLM

# 安裝依賴
pip install -r requirements.txt

# 安裝 Flash Attention (可選，提升性能)
pip install flash-attn --no-build-isolation
```

## 🚀 快速開始

### 1. 準備中文資料
```bash
# 準備繁體中文資料（AISHELL、WenetSpeech、Common Voice）
python scripts/prepare_data.py --config configs/default_config.yaml --action all

# 或分步驟執行
python scripts/prepare_data.py --action download  # 下載指南
python scripts/prepare_data.py --action prepare   # 處理資料
python scripts/prepare_data.py --action mixed     # 創建混合資料集
```

```python
# 中文資料格式範例
{
  "sample_id": "aishell_001",
  "mode": "AIAO",
  "input_audio_path": "input_zh.wav",
  "output_audio_path": "output_zh.wav",
  "output_text": "你好，這是中文回應。",
  "speaker_id": "SSB0005",
  "dataset_source": "AISHELL"
}
```

### 2. 訓練模型
```bash
# 使用預設配置訓練
python scripts/train.py --config configs/default_config.yaml

# 從檢查點恢復訓練
python scripts/train.py --config configs/default_config.yaml --resume_from_checkpoint ./outputs/checkpoint-1000
```

### 3. 推理使用
```bash
# 互動式聊天
python scripts/inference.py --model_path ./outputs/final_model --mode interactive

# 單輪推理
python scripts/inference.py --model_path ./outputs/final_model --mode single --input_text "你好，請介紹一下自己"

# 音訊輸入推理
python scripts/inference.py --model_path ./outputs/final_model --mode single --input_audio input.wav
```

### 4. 程式化使用
```python
from speechllm.models.speechllm import SpeechLLM, SpeechLLMConfig
from speechllm.inference.engine import SpeechLLMInferenceEngine
from speechllm.align.alignment import ChineseTextProcessor

# 載入模型（繁體中文優化）
config = SpeechLLMConfig(whisper_model_name="openai/whisper-medium")
model = SpeechLLM(config)
engine = SpeechLLMInferenceEngine(model)

# 繁體中文文字處理
text_processor = ChineseTextProcessor(convert_traditional=False)
processed_text = text_processor.normalize_text("你好，今天天氣如何？")

# 生成中文回應
response = engine.generate_response(
    input_text=processed_text,
    mode="TIAO"  # 文字輸入，音訊輸出
)

print(response["text"])  # 中文文字回應
# response["audio"] 包含中文語音回應
```

## 📊 訓練策略

### 三階段訓練
1. **Stage A - 輸入對齊**：訓練 Whisper encoder 和 Q-Former，實現音訊-文字對齊
2. **Stage B - 語音輸出**：訓練 Audio Transformer，學習從 LLM 隱狀態生成 RVQ tokens
3. **Stage C - 多任務聯訓**：端到端訓練，統一四種任務模式

### 損失函數
- **文字損失**：標準交叉熵損失
- **RVQ 損失**：多層 RVQ token 預測損失
- **對齊損失**：音訊-文字嵌入對齊損失
- **KL 蒸餾損失**：輸出分佈一致性損失

## 🔧 配置說明

### 模型配置
```yaml
model:
  llm_model_name: "Qwen/Qwen2.5-7B-Instruct"
  whisper_model_name: "openai/whisper-medium"  # 中文優化
  num_query_tokens: 32
  num_rvq_layers: 4
  codebook_size: 256
```

### 中文優化訓練配置
```yaml
training:
  batch_size: 4
  learning_rate: 1e-4
  stage_a_epochs: 4  # 中文輸入對齊階段
  stage_b_epochs: 4  # 中文語音輸出階段  
  stage_c_epochs: 6  # 中文多任務聯訓階段
  rvq_loss_weight: 0.6  # 提高 RVQ 損失權重
  alignment_loss_weight: 0.4  # 提高對齊損失權重
  kl_loss_weight: 0.3  # 提高 KL 蒸餾權重
  rvq_layer_weights: [3.0, 1.5, 1.0, 1.0]  # L1 語義層 > L2-L4 聲學層
```

### 推理配置
```yaml
inference:
  max_length: 512
  temperature: 0.8
  top_k: 50
  top_p: 0.9
```

## 📁 專案結構

```
SpeechLLM/
├── speechllm/                 # 核心程式碼
│   ├── models/                # 模型定義
│   │   ├── speechllm.py      # 主模型
│   │   ├── whisper_encoder.py # Whisper 編碼器
│   │   ├── qformer.py        # Q-Former
│   │   └── audio_transformer.py # 音訊 Transformer
│   ├── codecs/               # 音訊編解碼
│   │   ├── rvq_codec.py      # RVQ 編解碼器
│   │   ├── audio_tokenizer.py # 音訊 tokenizer
│   │   └── vocab_manager.py   # 詞彙表管理
│   ├── align/                # 對齊和交錯
│   │   ├── alignment.py      # 語音-文字對齊
│   │   └── interleaving.py   # 交錯序列生成
│   ├── data/                 # 資料處理
│   │   └── dataset.py        # 資料集定義
│   ├── training/             # 訓練相關
│   │   ├── trainer.py        # 訓練器
│   │   ├── loss.py          # 損失函數
│   │   └── optimizer.py      # 優化器
│   └── inference/            # 推理相關
│       └── engine.py         # 推理引擎
├── scripts/                  # 執行腳本
│   ├── train.py             # 訓練腳本
│   └── inference.py         # 推理腳本
├── configs/                  # 配置文件
│   └── default_config.yaml  # 預設配置
└── requirements.txt          # 依賴列表
```

## 🎯 任務模式

| 模式 | 輸入 | 輸出 | 應用場景 |
|------|------|------|----------|
| TITO | 文字 | 文字 | 傳統文字對話 |
| AITO | 音訊 | 文字 | 語音轉文字、語音問答 |
| TIAO | 文字 | 音訊 | 文字轉語音、語音合成 |
| AIAO | 音訊 | 音訊 | 語音對話、口譯 |

## 🔬 技術細節

### 繁體中文優化策略
- **Whisper Medium**：使用更大的 Whisper 模型提升繁體中文識別準確度
- **拼音音節對齊**：以拼音音節（含聲調）為基本對齊單位，如 "ni3 hao3 ma5"
- **繁體中文保持**：保持原始繁體中文，不進行繁簡轉換
- **中文分詞**：支援 jieba 和 pkuseg 分詞器，適配繁體中文
- **DTW 對齊**：可選的動態時間規劃對齊，提升對齊精度

### RVQ 編碼策略
- **4 層 RVQ**：第 1 層偏語義，第 2-4 層偏聲學
- **交錯對齊**：每個中文 token 重複 4 次對齊 RVQ 層數
- **多尺度解碼**：LLM 管語義，Audio Transformer 管語音生成

### 特殊 Token
```python
# 角色標籤
<human>, <assistant>, <eos>

# 模式標籤  
<AUDIO>, <TEXT>, <AUDIO_START>, <AUDIO_END>

# RVQ 代碼簿 token
<A_L1_000> ~ <A_L4_255>

# 說話人嵌入 token (可選)
<CHAT_REF_START>, <CHAT_REF>, <CHAT_REF_END>
```

## 📈 性能指標

### 推理性能
- **首音延遲**：< 500ms
- **串流延遲**：< 200ms  
- **音質**：支援 16/24 kHz 採樣率
- **GPU 記憶體**：7B 模型約需 16GB

### 訓練效率
- **收斂速度**：使用 DiVA 初始化技巧加速收斂
- **記憶體優化**：支援梯度檢查點和混合精度
- **分散式訓練**：支援多 GPU 訓練

## 🛠️ 開發指南

### 自定義模型
```python
# 繼承基礎配置
class CustomSpeechLLMConfig(SpeechLLMConfig):
    def __init__(self):
        super().__init__()
        self.custom_param = "value"

# 自定義模型
class CustomSpeechLLM(SpeechLLM):
    def __init__(self, config):
        super().__init__(config)
        # 添加自定義組件
```

### 添加新的音訊編解碼器
```python
class CustomAudioCodec(nn.Module):
    def encode(self, audio):
        # 實現編碼邏輯
        pass
    
    def decode(self, tokens):
        # 實現解碼邏輯  
        pass
```

## 🤝 貢獻指南

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權

本專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 文件

## 📊 中文資料集支援

### 支援的繁體中文資料集
- **AISHELL-1/2/4**：高品質中文朗讀和會議語音（可適配繁體）
- **WenetSpeech**：大規模中文語音資料集
- **Common Voice zh-TW**：繁體中文群眾錄音，台灣口音
- **Common Voice zh-HK**：繁體中文群眾錄音，香港口音
- **THCHS-30**：清華中文語音資料庫（可適配繁體）

### 資料集配置步驟

#### 1. 修改配置檔案
下載並解壓資料集後，請修改 `configs/default_config.yaml` 中的路徑：

```yaml
data:
  chinese_datasets:
    - type: "AISHELL"
      data_dir: "/home/user/data/AISHELL-1"  # 修改為實際路徑
      splits: ["train", "dev", "test"]
    - type: "CommonVoice"
      data_dir: "/home/user/data/common_voice/zh-TW"  # 修改為實際路徑
      language: "zh-TW"
      splits: ["train", "dev", "test"]
    - type: "WenetSpeech"
      data_dir: "/home/user/data/wenetspeech"  # 修改為實際路徑
      splits: ["train", "dev", "test"]
```

#### 2. 資料集使用範例
```bash
# 下載和準備 AISHELL-1
python scripts/prepare_data.py --dataset AISHELL --action prepare

# 創建混合資料集
python scripts/prepare_data.py --action mixed

# 驗證資料集完整性
python scripts/prepare_data.py --action validate
```

#### 3. 新增自定義資料集
若要新增 AISHELL-3 等其他資料集：
1. 在 `configs/default_config.yaml` 中新增配置
2. 在 `scripts/prepare_data.py` 中新增對應的處理函式

## 🙏 致謝

- **DiVA**: 輸入對齊和 Q-Former 初始化策略
- **Voila**: 交錯序列生成和多尺度解碼
- **Freeze-Omni**: 全雙工對話和狀態管理
- **Qwen**: 強大的中文語言模型基座
- **Whisper**: 優秀的多語言語音識別模型
- **AISHELL/WenetSpeech/Common Voice**: 優質的中文語音資料集

## 📞 聯繫方式

- 專案主頁：https://github.com/your-username/SpeechLLM
- 問題回報：https://github.com/your-username/SpeechLLM/issues
- 電子郵件：your-email@example.com

## 🔄 更新日誌

### v0.2.0 (2024-12-XX) - 繁體中文優化版
- **🇹🇼 完整繁體中文支援**：針對繁體中文語音對話全面優化
- **Whisper Medium**：使用更大模型提升繁體中文識別準確度
- **拼音音節對齊**：基於拼音音節的精確時間對齊
- **繁體中文保持**：保持原始繁體中文，不進行繁簡轉換
- **階層式 RVQ**：L1 語義層 > L2-L4 聲學層的權重策略
- **多口音支援**：支援台灣、香港等繁體中文地區口音
- **繁體中文資料集**：完整支援 Common Voice zh-TW/zh-HK 等
- **文字正規化**：數字、量詞、標點符號智能處理（適配繁體）
- **三階段訓練**：專門針對繁體中文的訓練策略優化

### v0.1.0 (2024-01-XX)
- 初始版本發布
- 實現基礎四模態對話功能
- 支援三階段訓練策略
- 提供完整的訓練和推理腳本

## 🇹🇼 繁體中文使用快速入門

### 準備繁體中文環境
```bash
# 安裝繁體中文處理依賴
pip install jieba pypinyin

# 1. 檢視資料集下載指南
python scripts/prepare_data.py --action download

# 2. 下載並解壓資料集後，修改 configs/default_config.yaml 中的 data_dir 路徑

# 3. 處理各資料集
python scripts/prepare_data.py --action prepare

# 4. 創建混合訓練資料集
python scripts/prepare_data.py --action mixed

# 5. 開始繁體中文訓練
python scripts/train.py --config configs/default_config.yaml
```

### 繁體中文推理示例
```python
from speechllm.align.alignment import ChineseTextProcessor

# 繁體中文文字處理（不轉換）
processor = ChineseTextProcessor(convert_traditional=False)
text = processor.normalize_text("你好，今天天氣如何？")

# 繁體中文語音對話
response = engine.generate_response(input_text=text, mode="TIAO")
```
