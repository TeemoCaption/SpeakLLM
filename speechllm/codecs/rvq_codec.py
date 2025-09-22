"""
RVQ (Residual Vector Quantization) Codec
實現 4 層 RVQ 的音訊編解碼器，支援語義和聲學分離
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class RVQConfig:
    """RVQ 配置"""
    num_layers: int = 4  # RVQ 層數
    codebook_size: int = 256  # 每層代碼簿大小
    embedding_dim: int = 512  # 嵌入維度
    commitment_loss_weight: float = 0.25  # 承諾損失權重
    semantic_layer: int = 0  # 語義層索引（第 1 層）
    sample_rate: int = 16000  # 採樣率
    hop_length: int = 320  # 跳躍長度（20ms at 16kHz）


class VectorQuantizer(nn.Module):
    """向量量化器"""
    
    def __init__(self, codebook_size: int, embedding_dim: int, commitment_weight: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        
        # 代碼簿嵌入
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/codebook_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch, seq_len, dim]
            
        Returns:
            quantized: 量化後的特徵
            indices: 量化索引
            losses: 損失字典
        """
        batch_size, seq_len, dim = x.shape
        
        # 展平輸入
        x_flat = x.view(-1, dim)  # [batch*seq_len, dim]
        
        # 計算距離
        distances = torch.cdist(x_flat, self.embedding.weight)  # [batch*seq_len, codebook_size]
        
        # 找到最近的代碼
        indices = torch.argmin(distances, dim=-1)  # [batch*seq_len]
        
        # 量化
        quantized_flat = self.embedding(indices)  # [batch*seq_len, dim]
        quantized = quantized_flat.view(batch_size, seq_len, dim)
        
        # 計算損失
        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        
        # 直通估計器
        quantized = x + (quantized - x).detach()
        
        losses = {
            "commitment_loss": commitment_loss * self.commitment_weight,
            "codebook_loss": codebook_loss,
            "total_loss": commitment_loss * self.commitment_weight + codebook_loss
        }
        
        return quantized, indices.view(batch_size, seq_len), losses


class ResidualVectorQuantizer(nn.Module):
    """殘差向量量化器"""
    
    def __init__(self, config: RVQConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        
        # 創建多層量化器
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                codebook_size=config.codebook_size,
                embedding_dim=config.embedding_dim,
                commitment_weight=config.commitment_loss_weight
            ) for _ in range(config.num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向傳播
        
        Args:
            x: 輸入特徵 [batch, seq_len, dim]
            
        Returns:
            quantized: 最終量化特徵
            all_indices: 所有層的量化索引
            losses: 損失字典
        """
        residual = x
        quantized = torch.zeros_like(x)
        all_indices = []
        total_losses = {"commitment_loss": 0, "codebook_loss": 0, "total_loss": 0}
        
        for i, quantizer in enumerate(self.quantizers):
            # 量化殘差
            q, indices, losses = quantizer(residual)
            
            # 累加量化結果
            quantized = quantized + q
            all_indices.append(indices)
            
            # 累加損失
            for key in total_losses:
                total_losses[key] = total_losses[key] + losses[key]
            
            # 更新殘差
            residual = residual - q
        
        return quantized, all_indices, total_losses
    
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """編碼：獲取量化索引"""
        _, indices, _ = self.forward(x)
        return indices
    
    def decode(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """解碼：從量化索引重建特徵"""
        quantized = torch.zeros(
            indices[0].shape[0], indices[0].shape[1], self.config.embedding_dim,
            device=indices[0].device, dtype=torch.float32
        )
        
        for i, (quantizer, layer_indices) in enumerate(zip(self.quantizers, indices)):
            q = quantizer.embedding(layer_indices)
            quantized = quantized + q
        
        return quantized


class AudioEncoder(nn.Module):
    """音訊編碼器"""
    
    def __init__(self, config: RVQConfig):
        super().__init__()
        self.config = config
        
        # 1D 卷積編碼器
        self.encoder = nn.Sequential(
            # 第一層：降採樣
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # 第二層
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # 第三層
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            # 第四層
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            # 投影到嵌入維度
            nn.Conv1d(512, config.embedding_dim, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        編碼音訊
        
        Args:
            x: 音訊波形 [batch, 1, time]
            
        Returns:
            features: 編碼特徵 [batch, seq_len, embedding_dim]
        """
        # 卷積編碼
        features = self.encoder(x)  # [batch, embedding_dim, seq_len]
        
        # 轉置到 [batch, seq_len, embedding_dim]
        features = features.transpose(1, 2)
        
        return features


class AudioDecoder(nn.Module):
    """音訊解碼器"""
    
    def __init__(self, config: RVQConfig):
        super().__init__()
        self.config = config
        
        # 1D 轉置卷積解碼器
        self.decoder = nn.Sequential(
            # 投影層
            nn.Conv1d(config.embedding_dim, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            # 第一層：上採樣
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            # 第二層
            nn.ConvTranspose1d(256, 128, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # 第三層
            nn.ConvTranspose1d(128, 64, kernel_size=6, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # 第四層：輸出
            nn.ConvTranspose1d(64, 1, kernel_size=8, stride=2, padding=3),
            nn.Tanh()  # 輸出範圍 [-1, 1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        解碼特徵為音訊
        
        Args:
            x: 量化特徵 [batch, seq_len, embedding_dim]
            
        Returns:
            audio: 重建音訊 [batch, 1, time]
        """
        # 轉置到 [batch, embedding_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 轉置卷積解碼
        audio = self.decoder(x)  # [batch, 1, time]
        
        return audio


class RVQCodec(nn.Module):
    """RVQ 音訊編解碼器"""
    
    def __init__(self, config: RVQConfig):
        super().__init__()
        self.config = config
        
        # 編碼器和解碼器
        self.encoder = AudioEncoder(config)
        self.decoder = AudioDecoder(config)
        
        # RVQ 量化器
        self.quantizer = ResidualVectorQuantizer(config)
        
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        前向傳播：編碼 -> 量化 -> 解碼
        
        Args:
            audio: 輸入音訊 [batch, 1, time]
            
        Returns:
            reconstructed: 重建音訊
            indices: 量化索引
            losses: 損失字典
        """
        # 編碼
        features = self.encoder(audio)
        
        # 量化
        quantized, indices, losses = self.quantizer(features)
        
        # 解碼
        reconstructed = self.decoder(quantized)
        
        # 添加重建損失
        recon_loss = F.l1_loss(reconstructed, audio)
        losses["reconstruction_loss"] = recon_loss
        losses["total_loss"] = losses["total_loss"] + recon_loss
        
        return reconstructed, indices, losses
    
    def encode(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """編碼音訊為量化索引"""
        features = self.encoder(audio)
        indices = self.quantizer.encode(features)
        return indices
    
    def decode(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """從量化索引解碼音訊"""
        quantized = self.quantizer.decode(indices)
        audio = self.decoder(quantized)
        return audio
    
    def get_semantic_codes(self, audio: torch.Tensor) -> torch.Tensor:
        """獲取語義層代碼（第 1 層）"""
        indices = self.encode(audio)
        return indices[self.config.semantic_layer]
    
    def get_acoustic_codes(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """獲取聲學層代碼（第 2-4 層）"""
        indices = self.encode(audio)
        return indices[1:]  # 跳過語義層


if __name__ == "__main__":
    # 測試 RVQ Codec
    config = RVQConfig()
    codec = RVQCodec(config)
    
    # 創建測試音訊（1 秒，16kHz）
    batch_size = 2
    audio_length = 16000
    test_audio = torch.randn(batch_size, 1, audio_length)
    
    print(f"輸入音訊形狀: {test_audio.shape}")
    
    # 前向傳播
    reconstructed, indices, losses = codec(test_audio)
    
    print(f"重建音訊形狀: {reconstructed.shape}")
    print(f"量化層數: {len(indices)}")
    print(f"每層索引形狀: {[idx.shape for idx in indices]}")
    
    print("\n損失:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # 測試編碼解碼
    print("\n編碼解碼測試:")
    encoded_indices = codec.encode(test_audio)
    decoded_audio = codec.decode(encoded_indices)
    
    print(f"編碼索引形狀: {[idx.shape for idx in encoded_indices]}")
    print(f"解碼音訊形狀: {decoded_audio.shape}")
    
    # 測試語義和聲學代碼分離
    semantic_codes = codec.get_semantic_codes(test_audio)
    acoustic_codes = codec.get_acoustic_codes(test_audio)
    
    print(f"\n語義代碼形狀: {semantic_codes.shape}")
    print(f"聲學代碼層數: {len(acoustic_codes)}")
    print(f"聲學代碼形狀: {[code.shape for code in acoustic_codes]}")
    
    print(f"\n模型參數量: {sum(p.numel() for p in codec.parameters()):,}")
