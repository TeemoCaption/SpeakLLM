# 資料集清單與使用策略

## ASR 與理解資料
- **`wenet-e2e/wenetspeech`**：中文 10k+ 小時，多域，支援 `streaming=True`，需同意條款。建議作 Phase 0 蒸餾與 Phase 1 半雙工訓練主力。
- **`AISHELL/AISHELL-1`**：178 小時普通話，乾淨錄音，適合初期對齊與驗證。
- **`AISHELL/AISHELL-3`**：85 小時多說話人 TTS 含拼音標註，可作 Tone-Aware 對齊與多說話人支援。
- **`mozilla-foundation/common_voice_17_0` (`zh-TW`, `zh-CN`, `yue`)**：CC0，涵蓋台灣口音與粵語。適合訓練搶話場景與多口音魯棒性。
- **`ASLP-lab/WenetSpeech-Yue`**：21.8k 小時粵語，可強化粵語或混碼場景。
- **`google/fleurs` (`zh_tw`, `zh_cn`, `yue_hk`)**：多語語音資料，適合語種辨識與零樣本對齊。
- **`facebook/covost2` (`zh-CN_en`, `en_zh-CN`)**：跨語種語音翻譯，授權 CC-BY-NC-4.0，適用於跨語義對齊與混碼訓練。
- **`BAAI/CS-Dialogue`**：104 小時中英混碼對話，提供 code-switch 正則的實際語料。
- **`gpt-omni/VoiceAssistant-400K`**：英語單輪語音指令，支援 Phase 1 行為蒸餾，需結合自建中文資料。
- **`LinkSoul/LLaSM-Audio-Instructions`**：中英雙語語音指令，適用於 Phase 2 全雙工對齊。
- **`BAAI/RealTalk-CN`**：150 小時中文任務導向對話，含重疊語音場景。

## TTS 與情緒資料
- **`AISHELL/AISHELL-3`**：多說話人中文 TTS，提供韻律與說話人標註。
- **`Wenetspeech4TTS/WenetSpeech4TTS`**：12,800 小時對齊資料，分 Premium/Standard/Basic 子集，附 DNSMOS 品質指標。
- **`mythicinfinity/libritts_r`**：英文音質修復版，多說話人 TTS，適合作為英文補強。
- **`badayvedat/VCTK`**：英文多口音，補足說話人多樣性與口音控制。
- **`AIDC-AI/CSEMOTIONS`**：七類中文情緒語音，支援 Phase 3 情緒建模。
- **`ESD`（官方渠道）**：中英情緒語音，需透過 GitHub/Kaggle 取得。

## Tokenizer 與模型資源
- **`maitrix-org/Voila-Tokenizer`**：MIT 授權，提供 RVQ 語義/聲學碼，支援交錯對齊與 Dual-Scale 規劃。
- **`CosyVoice2`**：官方論文、倉庫與 0.5B 權重，可下載串流推論示例。

## 資料使用建議
- **混碼比例**：全語料維持 zh:en ≈ 7:3，zh-TW 佔中文 15–25%，粵語依需求 10–15%。
- **授權注意**：`WenetSpeech`、`WenetSpeech4TTS`、`Voila-Tokenizer` 等需遵守 MIT/學術許可；`CoVoST2` 為非商用。
- **載入流程**：統一使用 Hugging Face `datasets`，以 `streaming=True` 處理大規模語料，並透過 `config` 選擇子集。
- **評估資料**：建立多域測試集（中文噪聲、粵語、英文、混碼、情緒）以評估延遲與音質。
