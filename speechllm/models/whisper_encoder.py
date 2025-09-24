"""
Whisper Encoder 模組
用於語音特徵提取和語音-文字對齊
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import whisper
from transformers import WhisperModel, WhisperConfig
import numpy as np
from speechllm.utils.whisper_compat import get_mel_filters


class WhisperAudioEncoder(nn.Module):
    """
    基於 Whisper 的音訊編碼器
    提取語音特徵並支援與 LLM 的對齊
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-medium",  # 預設使用 medium 模型以支援中文
        freeze_encoder: bool = False,
        output_dim: Optional[int] = None,
        use_cross_attention_weights: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.use_cross_attention_weights = use_cross_attention_weights
        
        # 載入 Whisper 模型
        self.whisper_model = WhisperModel.from_pretrained(model_name)
        self.config = self.whisper_model.config
        
        # 只使用 encoder 部分
        self.encoder = self.whisper_model.encoder
        
        # 凍結 encoder 參數（可選）
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 特徵維度
        self.hidden_size = self.config.d_model
        
        # 輸出投影層（可選）
        if output_dim and output_dim != self.hidden_size:
            self.output_projection = nn.Linear(self.hidden_size, output_dim)
            self.output_dim = output_dim
        else:
            self.output_projection = None
            self.output_dim = self.hidden_size
        
        # 位置編碼（用於長音訊）
        self.max_position_embeddings = self.config.max_source_positions
        
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            input_features: 音訊特徵 [batch, n_mels, time]
            attention_mask: 注意力遮罩
            output_attentions: 是否輸出注意力權重
            output_hidden_states: 是否輸出隱藏狀態
            
        Returns:
            outputs: 包含編碼特徵的字典
        """
        # Whisper encoder 前向傳播
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # 獲取最後一層隱藏狀態
        hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # 輸出投影（可選）
        if self.output_projection is not None:
            hidden_states = self.output_projection(hidden_states)
        
        outputs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask
        }
        
        # 添加額外輸出
        if output_attentions:
            outputs["attentions"] = encoder_outputs.attentions
        
        if output_hidden_states:
            outputs["all_hidden_states"] = encoder_outputs.hidden_states
        
        return outputs
    
    def extract_features(self, audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        從原始音訊提取特徵
        
        Args:
            audio: 原始音訊 [batch, time] 或 [batch, 1, time]
            sample_rate: 採樣率
            
        Returns:
            features: 音訊特徵 [batch, n_mels, time]
        """
        # 確保音訊格式正確
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # [batch, 1, time] -> [batch, time]
        
        batch_size = audio.shape[0]
        features = []
        
        for i in range(batch_size):
            # 使用 whisper 的特徵提取
            mel = whisper.log_mel_spectrogram(audio[i].cpu().numpy())
            features.append(torch.from_numpy(mel))
        
        # 堆疊並移到正確設備
        features = torch.stack(features).to(audio.device)
        
        return features
    
    def get_cross_attention_weights(
        self,
        audio_features: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        獲取交叉注意力權重（用於初始化 Q-Former）
        
        Args:
            audio_features: 音訊特徵
            text_tokens: 文字 token
            
        Returns:
            attention_weights: 交叉注意力權重
        """
        if not self.use_cross_attention_weights:
            return None
        
        # 使用完整的 Whisper 模型獲取交叉注意力權重
        with torch.no_grad():
            outputs = self.whisper_model(
                input_features=audio_features,
                decoder_input_ids=text_tokens,
                output_attentions=True
            )
            
            # 獲取交叉注意力權重
            cross_attentions = outputs.cross_attentions
            
            # 平均所有層和頭的注意力權重
            attention_weights = torch.stack(cross_attentions).mean(dim=(0, 2))  # [batch, dec_len, enc_len]
        
        return attention_weights


class AudioFeatureExtractor(nn.Module):
    """
    音訊特徵提取器
    將原始音訊轉換為 Whisper 可處理的特徵
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        sample_rate: int = 16000,
        normalize: bool = True
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.normalize = normalize
        
        # Mel 濾波器組 (支援多種 Whisper 版本)
        mel_filters = get_mel_filters(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft)
        self.mel_filters = torch.from_numpy(mel_filters).float()
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        提取 mel 頻譜特徵
        
        Args:
            audio: 原始音訊 [batch, time]
            
        Returns:
            mel_features: Mel 頻譜特徵 [batch, n_mels, time_frames]
        """
        batch_size = audio.shape[0]
        device = audio.device
        
        # 移動 mel 濾波器到正確設備
        if self.mel_filters.device != device:
            self.mel_filters = self.mel_filters.to(device)
        
        mel_features = []
        
        for i in range(batch_size):
            # STFT
            stft = torch.stft(
                audio[i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft).to(device),
                return_complex=True
            )
            
            # 功率譜
            magnitude = stft.abs() ** 2
            
            # Mel 濾波
            mel = torch.matmul(self.mel_filters, magnitude)
            
            # 對數變換
            log_mel = torch.clamp(mel, min=1e-10).log10()
            
            # 正規化
            if self.normalize:
                log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
                log_mel = (log_mel + 4.0) / 4.0
            
            mel_features.append(log_mel)
        
        return torch.stack(mel_features)


class WhisperEncoderWithProjection(nn.Module):
    """
    帶投影的 Whisper 編碼器
    整合特徵提取、編碼和投影
    """
    
    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-medium",  # 使用 medium 模型支援中文
        output_dim: int = 4096,  # Qwen 的隱藏維度
        freeze_whisper: bool = False,
        use_feature_extractor: bool = True
    ):
        super().__init__()
        
        self.use_feature_extractor = use_feature_extractor
        
        # 特徵提取器
        if use_feature_extractor:
            self.feature_extractor = AudioFeatureExtractor()
        
        # Whisper 編碼器
        self.whisper_encoder = WhisperAudioEncoder(
            model_name=whisper_model_name,
            freeze_encoder=freeze_whisper,
            output_dim=output_dim
        )
        
        self.output_dim = output_dim
        
    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            audio: 原始音訊 [batch, time] 或 mel 特徵 [batch, n_mels, time]
            attention_mask: 注意力遮罩
            return_attention_weights: 是否返回注意力權重
            
        Returns:
            outputs: 編碼輸出
        """
        # 特徵提取
        if self.use_feature_extractor and audio.dim() == 2:
            # 原始音訊 -> mel 特徵
            input_features = self.feature_extractor(audio)
        else:
            # 已經是 mel 特徵
            input_features = audio
        
        # Whisper 編碼
        outputs = self.whisper_encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=return_attention_weights
        )
        
        return outputs
    
    def get_output_dim(self) -> int:
        """獲取輸出維度"""
        return self.output_dim


if __name__ == "__main__":
    # 測試 Whisper 編碼器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 創建編碼器
    encoder = WhisperEncoderWithProjection(
        whisper_model_name="openai/whisper-medium",  # 使用 medium 模型
        output_dim=4096,
        freeze_whisper=False
    ).to(device)
    
    print(f"Whisper 編碼器初始化完成")
    print(f"輸出維度: {encoder.get_output_dim()}")
    print(f"參數量: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # 創建測試音訊
    batch_size = 2
    audio_length = 16000 * 3  # 3 秒音訊
    test_audio = torch.randn(batch_size, audio_length).to(device)
    
    print(f"\n測試音訊形狀: {test_audio.shape}")
    
    # 前向傳播
    with torch.no_grad():
        outputs = encoder(test_audio)
    
    print(f"編碼輸出形狀: {outputs['hidden_states'].shape}")
    print(f"注意力遮罩形狀: {outputs['attention_mask'].shape if outputs['attention_mask'] is not None else None}")
    
    # 測試特徵提取器
    print(f"\n測試特徵提取器:")
    feature_extractor = AudioFeatureExtractor()
    mel_features = feature_extractor(test_audio)
    print(f"Mel 特徵形狀: {mel_features.shape}")
    
    # 測試直接使用 mel 特徵
    with torch.no_grad():
        outputs_from_mel = encoder(mel_features)
    
    print(f"從 Mel 特徵編碼輸出形狀: {outputs_from_mel['hidden_states'].shape}")
    
    # 檢查輸出是否一致
    diff = torch.abs(outputs['hidden_states'] - outputs_from_mel['hidden_states']).max()
    print(f"輸出差異: {diff.item():.6f}")
