# 訓練階段與損失設計

## Phase 0：Connector 蒸餾對齊
- **目的**：讓凍結的 `Whisper Medium` 與 `Qwen2.5-7B-Instruct` 之間的語義橋樑穩定，避免後續 SFT 崩潰。
- **訓練設定**：
  - 凍結 `Whisper` encoder 與 `Qwen` 主幹。
  - 只訓練 `Connector`（Linear + Q-Former）與小型情緒/韻律 head。
  - 批量 32、梯度累積 4、4×A100-80G、bfloat16。
  - 學習率：`Connector` 1e-4，小 head 1e-5；cosine decay、warmup 10%。
- **損失**：
  - `L_kl`：對教師 LLM logits 做 KL 蒸餾。
  - `L_align`：`CTC` 或 `DTW`，將語音時間步對齊文字 token。
  - `L_roundtrip`：語音→文字→語音的簡化閉環正則。
- **評估**：蒸餾 perplexity、對齊準確率、延遲與記憶體占用。

## Phase 1：半雙工 SFT（中文優先）
- **資料**：`WenetSpeech`、`AISHELL`、`Common Voice zh-TW`、人工 QA/任務指令、`VoiceAssistant-400K` 中文版。
- **步驟**：
  1. 先訓練文字輸出（不合成），強化任務理解與答覆。
  2. 引入 `CosyVoice2` 離線合成，驗證文本→語音。
- **訓練設定**：
  - `LoRA` r=8、α=16、dropout 0.05，adapter LR 1e-4，warmup 3%。
  - 目標任務步數 10–30k，視資料量調整。
- **損失**：`L_ce`（文字交叉熵）+ 蒸餾正則（保持 Phase 0 對齊）。
- **評估**：中文 CER/WER、任務完成率、離線 TTS 音質。

## Phase 2：全雙工（交錯對齊 + 早期說話）
- **資料**：Phase 1 語料 + `MagicData-RAMC`、`RealTalk-CN`、`LLaSM-Audio-Instructions`、自建重疊語音資料。
- **訓練重點**：
  - 導入 Tone-aware 交錯序列（文字/tone/停頓交錯）。
  - 訓練 `Duplex Controller` 與 `Overlap-Aware Loss`。
  - `CosyVoice2` 串流模式，每 300–600 ms 更新。
- **損失**：
  - `L_ce` + `L_ctc/dtw`（交錯對齊）。
  - `L_overlap`（KL 或 contrastive）。
- **評估**：響應延遲（實測 RTF）、搶話成功率、語義一致性、tone 對齊偏差。

## Phase 3：情緒/說話人泛化
- **資料**：`AISHELL-3`、`WenetSpeech4TTS`、`CSEMOTIONS`、`ESD`、`LibriTTS-R`、`VCTK`。
- **訓練重點**：
  - 完成 Dual-Scale Planning（文字 + Prosody）。
  - 強化情緒/語者控制，與 `CosyVoice2` flow-matching/HiFi-GAN 對齊。
- **損失**：
  - `L_flow`（flow-matching 對數似然）。
  - `L_rvq`（Voila RVQ-CE）。
  - `L_prosody`（Prosody vs 梅爾/基頻 MSE/Huber）。
  - 語者正則或分類損失。
- **評估**：MOS/CMOS、情緒辨識準確率、Speaker EER、延遲維持。

## 訓練策略總結
- **混碼控制**：全程維持 zh:en ≈ 7:3、zh-TW 佔 15–25%，粵語視需求引入。
- **增強節奏**：優先在 Phase 0–1 使用 Premium/Standard 品質資料，Phase 3 才導入 Basic 擴增多樣性。
- **監控指標**：延遲、搶話精準度、tone 對齊偏差、語種分布、音質評分。
