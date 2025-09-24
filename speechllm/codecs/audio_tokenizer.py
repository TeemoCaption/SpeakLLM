"""
音訊 Tokenizer
整合 RVQ codec 和詞彙表管理，提供音訊到 token 的轉換
"""

import logging
import torch
import torchaudio
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import librosa
from pathlib import Path

from .rvq_codec import RVQCodec, RVQConfig
from .vocab_manager import VocabManager
from ..utils.device_utils import resolve_device_with_info


class AudioTokenizer:
    """
    音訊 Tokenizer
    負責將音訊轉換為離散 token，並支援與文字 token 的交錯
    """
    
    def __init__(
        self,
        rvq_config: Optional[RVQConfig] = None,
        vocab_manager: Optional[VocabManager] = None,
        codec_model_path: Optional[str] = None,
        device: Optional[str] = None
    ):

        self.logger = logging.getLogger(self.__class__.__name__)
        resolved_device, device_info = resolve_device_with_info(device, logger=self.logger)
        self.device = resolved_device

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Audio tokenizer device: {device_info}")

        # 初始化 RVQ codec
        if rvq_config is None:
            rvq_config = RVQConfig()
        self.rvq_config = rvq_config

        self.codec = RVQCodec(rvq_config).to(self.device)

        # 載入預訓練的 codec 模型
        if codec_model_path and Path(codec_model_path).exists():
            self.codec.load_state_dict(torch.load(codec_model_path, map_location=self.device))
            self.codec.eval()

        # 初始化詞彙表管理器
        if vocab_manager is None:
            vocab_manager = VocabManager()
        self.vocab_manager = vocab_manager

        
    def load_audio(self, audio_path: str, target_sr: Optional[int] = None) -> torch.Tensor:
        """
        載入音訊文件
        
        Args:
            audio_path: 音訊文件路徑
            target_sr: 目標採樣率，默認使用 codec 配置
            
        Returns:
            audio: 音訊張量 [1, time]
        """
        if target_sr is None:
            target_sr = self.rvq_config.sample_rate
        
        # 使用 librosa 載入音訊（支援更多格式）
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # 轉換為 torch tensor
        audio = torch.from_numpy(audio).float()
        
        # 確保是單聲道並添加 batch 維度
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [time] -> [1, time]
        
        return audio
    
    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        預處理音訊
        
        Args:
            audio: 輸入音訊 [channels, time] 或 [time]
            
        Returns:
            processed: 處理後的音訊 [1, 1, time]
        """
        # 確保是 float 類型
        audio = audio.float()
        
        # 轉換為單聲道
        if audio.dim() == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # 正規化到 [-1, 1]
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()
        
        # 添加 batch 維度
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # [1, channels, time]
        
        return audio
    
    def encode_audio(self, audio: Union[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        編碼音訊為 RVQ 索引
        
        Args:
            audio: 音訊路徑或音訊張量
            
        Returns:
            indices: RVQ 索引列表 [num_layers, seq_len]
        """
        # 載入音訊
        if isinstance(audio, str):
            audio = self.load_audio(audio)
        
        # 預處理
        audio = self.preprocess_audio(audio).to(self.device)
        
        # 編碼
        with torch.no_grad():
            indices = self.codec.encode(audio)
        
        # 移到 CPU
        indices = [idx.cpu() for idx in indices]
        
        return indices
    
    def decode_audio(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """
        從 RVQ 索引解碼音訊
        
        Args:
            indices: RVQ 索引列表
            
        Returns:
            audio: 重建的音訊 [1, 1, time]
        """
        # 移到設備
        indices = [idx.to(self.device) for idx in indices]
        
        # 解碼
        with torch.no_grad():
            audio = self.codec.decode(indices)
        
        return audio.cpu()
    
    def audio_to_tokens(self, audio: Union[str, torch.Tensor]) -> List[str]:
        """
        將音訊轉換為 token 字串
        
        Args:
            audio: 音訊路徑或音訊張量
            
        Returns:
            tokens: token 字串列表
        """
        # 編碼為 RVQ 索引
        indices = self.encode_audio(audio)
        
        # 轉換為 numpy 以便處理
        indices_np = [idx.squeeze(0).numpy() for idx in indices]  # 移除 batch 維度
        
        # 轉換為 token
        tokens = self.vocab_manager.rvq_codes_to_tokens(indices_np)
        
        return tokens
    
    def tokens_to_audio(self, tokens: List[str]) -> torch.Tensor:
        """
        將 token 字串轉換為音訊
        
        Args:
            tokens: token 字串列表
            
        Returns:
            audio: 重建的音訊
        """
        # 轉換為 RVQ 索引
        indices_np = self.vocab_manager.tokens_to_rvq_codes(tokens)
        
        # 轉換為 torch tensor 並添加 batch 維度
        indices = [torch.from_numpy(np.array(idx)).unsqueeze(0) for idx in indices_np]
        
        # 解碼為音訊
        audio = self.decode_audio(indices)
        
        return audio
    
    def create_chat_format(
        self,
        text: str,
        audio_path: Optional[str] = None,
        role: str = "human",
        mode: str = "TITO"  # TITO, AITO, TIAO, AIAO
    ) -> List[str]:
        """
        創建聊天格式的 token 序列
        
        Args:
            text: 文字內容
            audio_path: 音訊文件路徑（可選）
            role: 角色標籤
            mode: 任務模式
            
        Returns:
            tokens: 格式化的 token 序列
        """
        tokens = []
        
        # 添加角色標籤
        tokens.append(f"<{role}>")
        
        # 根據模式處理輸入
        if mode in ["TITO", "TIAO"]:  # 文字輸入
            tokens.append("<TEXT>")
            # 文字 tokenize（使用基礎 tokenizer）
            text_tokens = self.vocab_manager.extended_tokenizer.tokenize(text)
            tokens.extend(text_tokens)
            
        elif mode in ["AITO", "AIAO"]:  # 音訊輸入
            if audio_path is None:
                raise ValueError("音訊輸入模式需要提供 audio_path")
            
            tokens.append("<AUDIO>")
            tokens.append("<AUDIO_START>")
            
            # 音訊 tokenize
            audio_tokens = self.audio_to_tokens(audio_path)
            tokens.extend(audio_tokens)
            
            tokens.append("<AUDIO_END>")
        
        return tokens
    
    def create_response_format(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        mode: str = "TITO"
    ) -> List[str]:
        """
        創建回應格式的 token 序列
        
        Args:
            text: 文字回應
            audio_path: 音訊回應路徑
            mode: 任務模式
            
        Returns:
            tokens: 格式化的回應 token 序列
        """
        tokens = []
        
        # 添加助手標籤
        tokens.append("<assistant>")
        
        # 根據模式處理輸出
        if mode in ["TITO", "AITO"]:  # 文字輸出
            if text is None:
                raise ValueError("文字輸出模式需要提供 text")
            
            tokens.append("<TEXT>")
            text_tokens = self.vocab_manager.extended_tokenizer.tokenize(text)
            tokens.extend(text_tokens)
            
        elif mode in ["TIAO", "AIAO"]:  # 音訊輸出
            if audio_path is None:
                raise ValueError("音訊輸出模式需要提供 audio_path")
            
            tokens.append("<AUDIO>")
            tokens.append("<AUDIO_START>")
            
            audio_tokens = self.audio_to_tokens(audio_path)
            tokens.extend(audio_tokens)
            
            tokens.append("<AUDIO_END>")
        
        # 添加結束標籤
        tokens.append("<eos>")
        
        return tokens
    
    def create_interleaved_sequence(
        self,
        input_tokens: List[str],
        response_tokens: List[str]
    ) -> List[str]:
        """
        創建交錯序列（Voila 策略）
        
        Args:
            input_tokens: 輸入 token 序列
            response_tokens: 回應 token 序列
            
        Returns:
            interleaved: 交錯的 token 序列
        """
        # 分離文字和音訊 token
        text_tokens = []
        audio_tokens = []
        
        in_audio = False
        for token in input_tokens + response_tokens:
            if token == "<AUDIO_START>":
                in_audio = True
                continue
            elif token == "<AUDIO_END>":
                in_audio = False
                continue
            
            if in_audio and token.startswith("<A_L"):
                audio_tokens.append(token)
            elif not in_audio and not token.startswith("<"):
                text_tokens.append(token)
        
        # 創建交錯序列
        return self.vocab_manager.create_interleaved_sequence(text_tokens, audio_tokens)
    
    def save_audio(self, audio: torch.Tensor, save_path: str, sample_rate: Optional[int] = None):
        """
        保存音訊文件
        
        Args:
            audio: 音訊張量
            save_path: 保存路徑
            sample_rate: 採樣率
        """
        if sample_rate is None:
            sample_rate = self.rvq_config.sample_rate
        
        # 確保音訊格式正確
        if audio.dim() == 3:
            audio = audio.squeeze(0)  # 移除 batch 維度
        
        # 保存
        torchaudio.save(save_path, audio, sample_rate)


if __name__ == "__main__":
    # 測試音訊 tokenizer
    tokenizer = AudioTokenizer()
    
    print("音訊 Tokenizer 初始化完成")
    print(f"設備: {tokenizer.device}")
    print(f"RVQ 配置: {tokenizer.rvq_config}")
    
    # 創建測試音訊
    test_audio = torch.randn(1, 16000)  # 1 秒音訊
    
    print(f"\n測試音訊形狀: {test_audio.shape}")
    
    # 測試音訊到 token 轉換
    tokens = tokenizer.audio_to_tokens(test_audio)
    print(f"音訊 token 數量: {len(tokens)}")
    print(f"前 10 個 token: {tokens[:10]}")
    
    # 測試 token 到音訊轉換
    reconstructed = tokenizer.tokens_to_audio(tokens)
    print(f"重建音訊形狀: {reconstructed.shape}")
    
    # 測試聊天格式
    print("\n測試聊天格式:")
    
    # TITO 模式
    tito_tokens = tokenizer.create_chat_format(
        text="你好，請問今天天氣如何？",
        role="human",
        mode="TITO"
    )
    print(f"TITO 輸入 token: {tito_tokens}")
    
    tito_response = tokenizer.create_response_format(
        text="今天天氣很好，陽光明媚。",
        mode="TITO"
    )
    print(f"TITO 回應 token: {tito_response}")
    
    # 測試交錯序列
    interleaved = tokenizer.create_interleaved_sequence(tito_tokens, tito_response)
    print(f"交錯序列長度: {len(interleaved)}")
    print(f"交錯序列前 20 個 token: {interleaved[:20]}")
