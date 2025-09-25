# 推論與系統工程

## 串流推論管線
1. 麥克風輸入 16 kHz → 20 ms hop → 線上分幀。
2. `Whisper Medium` encoder（凍結）輸出 768 維隱向量。
3. `Connector`（Linear 768→384 + 2 層 Q-Former，downsample ratio=3）產生 256–384 維序列。
4. 序列交錯文字/語音 token，輸入 `Qwen2.5-7B-Instruct`（LoRA）。
5. 每 150–300 ms `Qwen` 產生文字片段與 prosody embedding，並更新 `Duplex Controller` 機率。
6. `CosyVoice2` 串流 TTS 接收文字 + prosody，使用 320–640 ms chunk 生成音訊，首包目標 ≤ 200 ms。

## 搶話與音量控制
- **偵測條件**：VAD 驗證輸入音量 > -35 dBFS，持續 120 ms。
- **處理流程**：
  - 立即將當前輸出音量降低 8–12 dB。
  - 將最新語音摘要、語義狀態、輸出進度送入 `Duplex Controller`。
  - `Duplex Controller` 判定 `continue`、`hold` 或 `barge-in`：
    - `hold`：暫停 `CosyVoice2` 播放並待輸入結束。
    - `barge-in`：提前生成澄清或反問，或完全停止輸出並轉為文字回覆。

## 緩衝與容錯
- 輸出端設置 120–200 ms jitter buffer，吸收網路抖動。
- 若偵測長時間延遲或網路不穩，退化為文字回覆保底，同時保留語音上下文，恢復後無縫銜接。

## 部署建議
- **硬體**：單卡 24 GB GPU 可運行 `Whisper` + `Qwen` + `CosyVoice2` 串流；如需更低延遲，可將 `Whisper` 前端移至專用 GPU/CPU。
- **服務拆分**：
  - `frontend_service`：音訊獲取與前處理。
  - `llm_service`：`Qwen` + `Duplex Controller`。
  - `tts_service`：`CosyVoice2` 串流合成。
  - `monitor_service`：延遲、搶話事件、音質監控。
- **監控指標**：端到端延遲、首包時間、搶話成功率、tone 對齊偏差、MOS、語音輸出中斷率。

## 安全與隱私
- 提供語音資料匿名化與加密傳輸，確保僅保留必要特徵。
- 設計刪除請求流程，允許用戶移除歷史音訊。
- 監控攻擊向量，如重複觸發 barge-in 試圖干擾輸出。
