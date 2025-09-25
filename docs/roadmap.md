# 路線圖與超參數建議

## 里程碑
- **M1（第 4 週）**：完成資料清理、自動對齊管線、Phase 0 蒸餾驗證。
- **M2（第 8 週）**：完成 Phase 1 半雙工 SFT，提供中文優先文字→語音 demo。
- **M3（第 12 週）**：完成 Phase 2 全雙工訓練，實現串流搶話，延遲與穩定度達標。
- **M4（第 16 週）**：完成 Phase 3 情緒/語者泛化與 Dual-Scale 規劃。
- **M5（第 20 週）**：整合評估、部署 PoC、撰寫技術白皮書。

## 超參數與設定總覽
- **Whisper Medium**：滑窗 8 秒、stride 4 秒、16 kHz、20 ms hop，保持凍結。
- **Connector**：Linear 768→384、2 層 Q-Former（8 頭、隱層 1024）、downsample ratio 3、激活 GELU、LayerNorm。
- **Qwen2.5-7B-Instruct**：LoRA r=8、α=16、dropout 0.05；adapter LR 1e-4，輸入線性 LR 5e-6，cosine decay，warmup 3%，總步數 10–30k；context 長度 32k。
- **Duplex Controller**：2 層 MLP（512→256→3），dropout 0.1，交叉熵損失。
- **CosyVoice2 串流**：首包 ≤ 200 ms，chunk 320–640 ms，flow-matching 步長與官方一致，保持 24 kHz。
- **Tone-Aware 對齊**：CTC blank penalty 1.0、tone tag attention bias +0.3。
- **Overlap Loss**：`KL` 溫度 0.7、margin 0.2，僅對重疊片段啟用。
- **Code-switch 正則**：語種占比目標向量 `[0.7, 0.3]`，正則係數 0.1。

## 風險與對策
- **資料授權**：針對 `WenetSpeech`、`WenetSpeech4TTS` 等學術授權資料，建立使用審查流程；`CoVoST2` 僅限研究用途。
- **模型延遲**：若端到端延遲 > 1 秒，可考慮 `Qwen` 推理裁剪或部署多服務分擔。
- **搶話誤判**：持續蒐集 `Duplex Controller` 錯誤案例，加入 offline RL 或人類回饋修正。
- **情緒泛化**：若情緒生成不穩，提升 `CSEMOTIONS`、`ESD` 權重，或引入 SER 模型做自動標注。

## 評估與驗收
- **延遲指標**：首包、平均 RTF、重啟恢復時間。
- **語音品質**：MOS/CMOS、tone accuracy、prosody RMSE。
- **對話表現**：搶話成功率、澄清應答率、任務完成率。
- **混碼能力**：zh/en CER、語種占比偏差、語言切換延遲。
- **部署檢查**：資安稽核、資料遺忘流程、日誌匿名化。
