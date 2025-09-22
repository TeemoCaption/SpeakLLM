"""
SpeechLLM 損失函數
實現多任務損失，包含文字損失、RVQ 損失、對齊損失和 KL 蒸餾損失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class SpeechLLMLoss(nn.Module):
    """
    SpeechLLM 多任務損失函數
    
    包含：
    1. 文字生成損失（交叉熵）
    2. RVQ token 預測損失
    3. 輸入對齊損失（DiVA 風格）
    4. KL 蒸餾損失
    """
    
    def __init__(
        self,
        text_weight: float = 1.0,
        rvq_weight: float = 0.5,
        alignment_weight: float = 0.3,
        kl_weight: float = 0.2,
        label_smoothing: float = 0.1,
        ignore_index: int = -100
    ):
        super().__init__()
        
        self.text_weight = text_weight
        self.rvq_weight = rvq_weight
        self.alignment_weight = alignment_weight
        self.kl_weight = kl_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        
        # 文字損失（帶標籤平滑）
        self.text_loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        
        # RVQ 損失
        self.rvq_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # KL 散度損失
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        
    def forward(
        self,
        text_logits: torch.Tensor,
        text_labels: torch.Tensor,
        rvq_predictions: Optional[List[torch.Tensor]] = None,
        rvq_labels: Optional[List[torch.Tensor]] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        audio_distribution: Optional[torch.Tensor] = None,
        text_distribution: Optional[torch.Tensor] = None,
        mode: str = "TITO"
    ) -> Dict[str, torch.Tensor]:
        """
        計算多任務損失
        
        Args:
            text_logits: 文字預測 logits [batch, seq_len, vocab_size]
            text_labels: 文字標籤 [batch, seq_len]
            rvq_predictions: RVQ 預測 logits 列表
            rvq_labels: RVQ 標籤列表
            audio_embeddings: 音訊嵌入
            text_embeddings: 文字嵌入
            audio_distribution: 音訊分佈（用於 KL 蒸餾）
            text_distribution: 文字分佈（用於 KL 蒸餾）
            mode: 任務模式
            
        Returns:
            losses: 損失字典
        """
        losses = {}
        total_loss = 0
        
        # 1. 文字生成損失
        if text_logits is not None and text_labels is not None:
            text_loss = self.text_loss_fn(
                text_logits.view(-1, text_logits.size(-1)),
                text_labels.view(-1)
            )
            losses["text_loss"] = text_loss
            total_loss += self.text_weight * text_loss
        
        # 2. RVQ 損失（音訊輸出模式）
        if rvq_predictions is not None and rvq_labels is not None and mode in ["TIAO", "AIAO"]:
            rvq_loss = self._compute_rvq_loss(rvq_predictions, rvq_labels)
            losses["rvq_loss"] = rvq_loss
            total_loss += self.rvq_weight * rvq_loss
        
        # 3. 輸入對齊損失（DiVA 風格）
        if audio_embeddings is not None and text_embeddings is not None:
            alignment_loss = self._compute_alignment_loss(audio_embeddings, text_embeddings)
            losses["alignment_loss"] = alignment_loss
            total_loss += self.alignment_weight * alignment_loss
        
        # 4. KL 蒸餾損失
        if audio_distribution is not None and text_distribution is not None:
            kl_loss = self._compute_kl_loss(audio_distribution, text_distribution)
            losses["kl_loss"] = kl_loss
            total_loss += self.kl_weight * kl_loss
        
        losses["total_loss"] = total_loss
        return losses
    
    def _compute_rvq_loss(
        self,
        predictions: List[torch.Tensor],
        labels: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        計算 RVQ 損失（中文優化：階層式權重）
        對每層 RVQ 分別計算交叉熵損失，語義層權重更高
        """
        total_rvq_loss = 0
        num_layers = len(predictions)
        
        # 中文優化：階層式權重分配
        # L1（語義層）權重最高，L2-L4（聲學層）權重遞減
        layer_weights = [3.0, 1.5, 1.0, 1.0]  # L1, L2, L3, L4
        
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            # 計算當前層的損失
            layer_loss = self.rvq_loss_fn(
                pred.view(-1, pred.size(-1)),
                label.view(-1)
            )
            
            # 應用階層式權重
            layer_weight = layer_weights[i] if i < len(layer_weights) else 1.0
            total_rvq_loss += layer_weight * layer_loss
        
        # 加權平均
        total_weight = sum(layer_weights[:num_layers])
        return total_rvq_loss / total_weight
    
    def _compute_alignment_loss(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        計算輸入對齊損失（DiVA 風格）
        使音訊和文字嵌入在語義空間中對齊
        """
        # 確保維度匹配
        if audio_embeddings.size(-1) != text_embeddings.size(-1):
            # 如果維度不匹配，使用餘弦相似度
            audio_norm = F.normalize(audio_embeddings, p=2, dim=-1)
            text_norm = F.normalize(text_embeddings, p=2, dim=-1)
            
            # 計算餘弦相似度
            similarity = torch.sum(audio_norm * text_norm, dim=-1)
            
            # 對齊損失：最大化相似度
            alignment_loss = 1.0 - similarity.mean()
        else:
            # 維度匹配時使用 MSE 損失
            alignment_loss = F.mse_loss(audio_embeddings, text_embeddings)
        
        return alignment_loss
    
    def _compute_kl_loss(
        self,
        audio_distribution: torch.Tensor,
        text_distribution: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 KL 蒸餾損失
        使音訊輸入和文字輸入產生相似的輸出分佈
        """
        # 確保分佈是 log-softmax 格式
        if audio_distribution.dim() == 3:
            audio_log_prob = F.log_softmax(audio_distribution, dim=-1)
            text_prob = F.softmax(text_distribution, dim=-1)
        else:
            audio_log_prob = F.log_softmax(audio_distribution, dim=-1)
            text_prob = F.softmax(text_distribution, dim=-1)
        
        # 計算 KL 散度
        kl_loss = self.kl_loss_fn(audio_log_prob, text_prob)
        
        return kl_loss


class ContrastiveLoss(nn.Module):
    """
    對比學習損失
    用於音訊-文字對齊
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        計算對比損失
        
        Args:
            audio_features: 音訊特徵 [batch, dim]
            text_features: 文字特徵 [batch, dim]
            
        Returns:
            loss: 對比損失
        """
        # 正規化特徵
        audio_features = F.normalize(audio_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 計算相似度矩陣
        similarity_matrix = torch.matmul(audio_features, text_features.T) / self.temperature
        
        # 創建標籤（對角線為正樣本）
        batch_size = audio_features.size(0)
        labels = torch.arange(batch_size, device=audio_features.device)
        
        # 計算交叉熵損失
        loss_audio_to_text = F.cross_entropy(similarity_matrix, labels)
        loss_text_to_audio = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_audio_to_text + loss_text_to_audio) / 2


class FocalLoss(nn.Module):
    """
    Focal Loss
    用於處理類別不平衡問題
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        計算 Focal Loss
        
        Args:
            inputs: 預測 logits [batch, num_classes]
            targets: 目標標籤 [batch]
            
        Returns:
            loss: Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class AdaptiveLossWeighting(nn.Module):
    """
    自適應損失權重
    根據訓練進度動態調整不同損失的權重
    """
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_rate: float = 0.01
    ):
        super().__init__()
        self.initial_weights = initial_weights
        self.adaptation_rate = adaptation_rate
        
        # 可學習的權重參數
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.log(torch.tensor(weight)))
            for name, weight in initial_weights.items()
        })
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        計算加權總損失
        
        Args:
            losses: 各項損失
            
        Returns:
            weighted_loss: 加權總損失
        """
        total_loss = 0
        weights = {}
        
        for name, loss in losses.items():
            if name in self.log_weights:
                weight = torch.exp(self.log_weights[name])
                weights[name] = weight
                total_loss += weight * loss
        
        # 記錄當前權重（用於監控）
        self.current_weights = weights
        
        return total_loss


class PerplexityMetric:
    """困惑度指標"""
    
    @staticmethod
    def compute(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
        """
        計算困惑度
        
        Args:
            logits: 預測 logits
            labels: 真實標籤
            ignore_index: 忽略的索引
            
        Returns:
            perplexity: 困惑度值
        """
        # 計算交叉熵損失
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index
        )
        
        # 困惑度 = exp(loss)
        perplexity = torch.exp(loss).item()
        
        return perplexity


if __name__ == "__main__":
    # 測試損失函數
    print("測試 SpeechLLM 損失函數")
    
    # 創建測試資料
    batch_size = 2
    seq_len = 50
    vocab_size = 1000
    codebook_size = 256
    embed_dim = 512
    
    # 文字相關
    text_logits = torch.randn(batch_size, seq_len, vocab_size)
    text_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # RVQ 相關
    rvq_predictions = [torch.randn(batch_size, seq_len, codebook_size) for _ in range(4)]
    rvq_labels = [torch.randint(0, codebook_size, (batch_size, seq_len)) for _ in range(4)]
    
    # 嵌入相關
    audio_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    text_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    
    # 分佈相關
    audio_distribution = torch.randn(batch_size, seq_len, vocab_size)
    text_distribution = torch.randn(batch_size, seq_len, vocab_size)
    
    print("測試資料創建完成")
    
    # 測試主損失函數
    loss_fn = SpeechLLMLoss()
    
    losses = loss_fn(
        text_logits=text_logits,
        text_labels=text_labels,
        rvq_predictions=rvq_predictions,
        rvq_labels=rvq_labels,
        audio_embeddings=audio_embeddings,
        text_embeddings=text_embeddings,
        audio_distribution=audio_distribution,
        text_distribution=text_distribution,
        mode="AIAO"
    )
    
    print("\n損失計算結果:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.4f}")
    
    # 測試對比損失
    print("\n測試對比損失:")
    contrastive_loss_fn = ContrastiveLoss()
    audio_features = torch.randn(batch_size, embed_dim)
    text_features = torch.randn(batch_size, embed_dim)
    
    contrastive_loss = contrastive_loss_fn(audio_features, text_features)
    print(f"對比損失: {contrastive_loss.item():.4f}")
    
    # 測試困惑度
    print("\n測試困惑度:")
    perplexity = PerplexityMetric.compute(text_logits, text_labels)
    print(f"困惑度: {perplexity:.2f}")
    
    # 測試 Focal Loss
    print("\n測試 Focal Loss:")
    focal_loss_fn = FocalLoss()
    focal_loss = focal_loss_fn(text_logits.view(-1, vocab_size), text_labels.view(-1))
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    print("\n損失函數測試完成！")
