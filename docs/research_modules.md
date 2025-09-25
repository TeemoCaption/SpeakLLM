# 研究增量模組

## 語義-聲學雙尺度規劃（Dual-Scale Planning）
- **目標**：讓 `Qwen2.5-7B-Instruct` 同步輸出文字與語義-聲學提示向量，涵蓋 tone1–4、輕聲、停連、韻尾、斷句、情緒 arousal/valence。
- **核心機制**：
  - 使用 `Voila-Tokenizer` 第 1 層語義碼作 anchor，其餘層作稀疏聲學碼。
  - Prosody head 與文字 decoder 並行，輸出對應 `CosyVoice2` 的節奏與情緒向量。
  - 維持文字 token 與語音語義碼的 1:1 交錯。
- **損失函數**：
  - `L_flow`：`CosyVoice2` flow-matching / 對數似然。
  - `L_rvq`：`Voila` RVQ-CE（語義層 + 稀疏聲學層）。
  - `L_prosody`：語義提示與梅爾頻譜、基頻的 `MSE/Huber`。
  - 建議權重 `0.5 * L_flow + 0.3 * L_rvq + 0.2 * L_prosody`。

## 中文聲調感知交錯對齊（Tone-Aware Interleaved Alignment）
- **目標**：強化 tone tag、停頓、語氣詞在 `LLM` 序列中的對齊，提升搶話與跟讀穩定性。
- **核心機制**：
  - 使用 `g2pC` 產生拼音+聲調、停連標記，插入文字序列形成交錯 token。
  - `Whisper` encoder 特徵以 `CTC` 或 `DTW` 對齊交錯序列。
  - 在注意力層加入 tone-aware bias，保證 tone tag 對應正確時間步。
- **損失**：`L_ctc` 或 `L_dtw`，搭配 `L_ce`（文字交叉熵）。
- **輸出**：tone-aware 表示可直接傳遞給 `CosyVoice2` 調整韻律。

## 重疊語音一致性懲罰（Overlap-Aware Loss）
- **目標**：在全雙工情境下，避免模型的輸出語義與輸入語音衝突。
- **核心機制**：
  - 透過 VAD 標記輸入語音片段，與輸出 token 對齊，若偵測衝突則施加懲罰。
  - 利用 `Duplex Controller` 標註資料（`continue/hold/barge-in`）指導策略。
- **損失**：
  - `L_overlap`：`KL` 或 margin-based contrastive（輸入語義 `z_in` vs 輸出 `z_out`）。
  - 與 `L_ce` 聯合訓練。
- **效果**：鼓勵模型在衝突時選擇澄清或暫停，降低聽覺干擾。

## 中英混碼穩健化（Code-Switch Robustness）
- **目標**：維持中文優先前提下的中英混碼能力。
- **核心機制**：
  - 依語句插入語言標記 token（`lang_zh`、`lang_en`）。
  - 控制資料比例 zh:en ≈ 7:3，zh-TW 佔中文 15–25%。
  - 引入語種占比正則 `L_lang`（`KL` 或 `L2`）。
- **輔助特徵**：`Connector` 接收語言識別輸入，`CosyVoice2` prosody head 接收語言標記以調整韻律。
- **評估**：code-switch CER/WER、語言切換延遲、語種分布偏差。
