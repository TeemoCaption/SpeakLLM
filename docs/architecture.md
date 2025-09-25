# 系統架構總覽

## 高階模組
```mermaid
flowchart LR
    subgraph Frontend[感知前端]
        mic[(麥克風輸入)] --> fe[16 kHz / 20 ms hop]
        fe --> whisper[Whisper Medium Encoder]
        whisper --> connector[Connector (Linear + Q-Former)]
    end

    subgraph LLM[理解與規劃]
        connector --> qwen[Qwen2.5-7B-Instruct (LoRA)]
        qwen --> duplex[Duplex Controller]
        qwen --> text[文字 / Prosody Token]
    end

    subgraph Output[語音輸出]
        text --> cosyA[CosyVoice2 串流 TTS]
        text --> voila[Voila RVQ 語義 / 聲學碼]
        voila --> cosyB[Flow-Matching / HiFi-GAN 對齊]
        cosyA --> speaker[(語音播放)]
    end

    duplex --> cosyA
    duplex --> controllerLog[策略日誌]
```

## 模組說明
- 感知前端：以凍結的 `Whisper Medium` 抽取 768 維語音特徵，`Connector` 透過 `Linear` 與 2 層 `Q-Former` 將每 3 個時間步聚合至 256–384 維。
- 理解與規劃：`Qwen2.5-7B-Instruct` 維持 32k context，負責產生文字 token、韻律提示與策略信號；`Duplex Controller` 以 2 層 MLP 輸出 `{continue, hold, barge-in}` 機率。
- 語音輸出：主路徑使用 `CosyVoice2` 串流合成；輔助路徑以 `Voila-Tokenizer` 的 RVQ 語義碼作研究與多任務損失。

## 雙流交錯流程
1. `Connector` 產出的音訊向量與文字 token 交錯輸入 `Qwen`，並保留 tone tag、停頓標記。
2. `Qwen` 將輸出拆為文字片段與 prosody embedding，並同步輸出策略頭機率。
3. `CosyVoice2` 每 320–640 ms chunk 更新，達成 ≈150–200 ms 首包延遲。
4. 若偵測到輸入語音超過 -35 dBFS 且持續 120 ms，`Duplex Controller` 觸發音量衰減並判定是否搶話。

## 主要資料流
- ASR 鏈維持 16 kHz，供 `Whisper` 與 `Connector` 使用。
- TTS 鏈重採樣至 24 kHz，對應 `CosyVoice2` 與多說話人資料集。
- Tone 與語言標記由 `g2pC`、語言識別器生成，與語音向量一同進入 `Qwen`。

## 擴充接口
- `Duplex Controller` 可插入額外策略（如對話狀態機、RL Policy）。
- `CosyVoice2` 可替換為其他低延遲 TTS，只需遵守文字 + prosody 的輸入介面。
- `Voila-Tokenizer` 可更新至新版 RVQ，保持與交錯對齊損失一致即可。
