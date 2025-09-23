# 日誌系統說明

本目錄用於存放 SpeechLLM 專案的日誌檔案。

## 日誌檔案類型

### 資料準備日誌
- 檔案格式：`prepare_data_YYYYMMDD_HHMMSS.log`
- 內容：資料集下載、處理、驗證的詳細記錄
- 產生腳本：`scripts/prepare_data.py`

### 訓練日誌
- 檔案格式：`train_YYYYMMDD_HHMMSS.log`
- 內容：模型訓練過程、損失變化、檢查點保存等記錄
- 產生腳本：`scripts/train.py`

### 推理日誌
- 檔案格式：`inference_YYYYMMDD_HHMMSS.log`
- 內容：模型推理、語音生成、評估結果等記錄

## 日誌格式

所有日誌檔案採用統一格式：
```
YYYY-MM-DD HH:MM:SS - 模組名稱 - 日誌等級 - 訊息內容
```

## 日誌等級

- **INFO**: 一般資訊記錄
- **WARNING**: 警告訊息（不影響執行但需注意）
- **ERROR**: 錯誤訊息（影響執行）
- **DEBUG**: 除錯資訊（詳細執行過程）

## 日誌保留

建議定期清理舊的日誌檔案，保留最近 30 天的記錄即可。
