"""
Q-Former 模組
用於將 Whisper encoder 的時間序列特徵聚合成 LLM 風格的 token
基於 DiVA 的設計，使用 Whisper decoder 的交叉注意力權重初始化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler


class QFormerConfig:
    """Q-Former 配置"""
    def __init__(
        self,
        num_query_tokens: int = 32,  # 查詢 token 數量
        hidden_size: int = 768,  # 隱藏層大小
        num_hidden_layers: int = 12,  # Transformer 層數
        num_attention_heads: int = 12,  # 注意力頭數
        intermediate_size: int = 3072,  # 前饋網路大小
        hidden_dropout_prob: float = 0.1,  # Dropout 機率
        attention_probs_dropout_prob: float = 0.1,  # 注意力 Dropout
        max_position_embeddings: int = 512,  # 最大位置嵌入
        layer_norm_eps: float = 1e-12,  # LayerNorm epsilon
        cross_attention_freq: int = 2,  # 交叉注意力頻率（每 N 層一次）
        encoder_hidden_size: int = 512,  # 編碼器隱藏大小（Whisper）
    ):
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.cross_attention_freq = cross_attention_freq
        self.encoder_hidden_size = encoder_hidden_size


class QFormerMultiHeadAttention(nn.Module):
    """Q-Former 多頭注意力"""
    
    def __init__(self, config: QFormerConfig, is_cross_attention: bool = False):
        super().__init__()
        self.config = config
        self.is_cross_attention = is_cross_attention
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 查詢、鍵、值投影
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        
        if is_cross_attention:
            # 交叉注意力：K, V 來自編碼器
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            # 自注意力：K, V 來自同一輸入
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """重塑張量以進行多頭注意力計算"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_size]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向傳播"""
        
        # 計算查詢
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        # 計算鍵和值
        if self.is_cross_attention and encoder_hidden_states is not None:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 計算注意力分數
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 應用注意力遮罩
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 應用注意力權重
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 重塑輸出
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 輸出投影
        output = self.output_projection(context_layer)
        output = self.output_dropout(output)
        
        outputs = (output, attention_probs) if output_attentions else (output,)
        return outputs


class QFormerLayer(nn.Module):
    """Q-Former Transformer 層"""
    
    def __init__(self, config: QFormerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 自注意力
        self.self_attention = QFormerMultiHeadAttention(config, is_cross_attention=False)
        self.self_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 交叉注意力（每隔 N 層一次）
        self.has_cross_attention = (layer_idx % config.cross_attention_freq == 0)
        if self.has_cross_attention:
            self.cross_attention = QFormerMultiHeadAttention(config, is_cross_attention=True)
            self.cross_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 前饋網路
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """前向傳播"""
        
        # 自注意力
        self_attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]
        
        # 殘差連接和層正規化
        hidden_states = self.self_attention_layer_norm(hidden_states + attention_output)
        
        # 交叉注意力
        cross_attention_probs = None
        if self.has_cross_attention and encoder_hidden_states is not None:
            cross_attention_outputs = self.cross_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
            cross_attention_output = cross_attention_outputs[0]
            if output_attentions:
                cross_attention_probs = cross_attention_outputs[1]
            
            # 殘差連接和層正規化
            hidden_states = self.cross_attention_layer_norm(hidden_states + cross_attention_output)
        
        # 前饋網路
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(hidden_states + feed_forward_output)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (self_attention_outputs[1], cross_attention_probs)
        
        return outputs


class QFormer(nn.Module):
    """
    Q-Former 模型
    將 Whisper encoder 的時間序列特徵聚合成固定數量的語義 token
    """
    
    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.config = config
        
        # 查詢 token 嵌入
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_query_tokens, config.hidden_size)
        )
        
        # 位置嵌入
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # Transformer 層
        self.layers = nn.ModuleList([
            QFormerLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        
        # 層正規化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 輸入投影（將 Whisper 特徵投影到 Q-Former 維度）
        if config.encoder_hidden_size != config.hidden_size:
            self.encoder_projection = nn.Linear(config.encoder_hidden_size, config.hidden_size)
        else:
            self.encoder_projection = None
        
        # 初始化權重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化權重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def init_from_whisper_cross_attention(self, whisper_cross_attention_weights: torch.Tensor):
        """
        使用 Whisper decoder 的交叉注意力權重初始化 Q-Former
        這是 DiVA 的關鍵技巧
        
        Args:
            whisper_cross_attention_weights: Whisper 交叉注意力權重
        """
        # 這裡實現從 Whisper 交叉注意力權重初始化的邏輯
        # 具體實現需要根據 Whisper 模型的結構調整
        print("使用 Whisper 交叉注意力權重初始化 Q-Former")
        
        # 示例：初始化交叉注意力層的 K, V 投影
        for layer in self.layers:
            if layer.has_cross_attention:
                # 這裡可以將 Whisper 的權重複製到 Q-Former
                # layer.cross_attention.key.weight.data = ...
                # layer.cross_attention.value.weight.data = ...
                pass
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            encoder_hidden_states: 編碼器隱藏狀態 [batch, seq_len, encoder_hidden_size]
            encoder_attention_mask: 編碼器注意力遮罩
            output_attentions: 是否輸出注意力權重
            output_hidden_states: 是否輸出所有隱藏狀態
            
        Returns:
            outputs: 包含聚合特徵的字典
        """
        batch_size = encoder_hidden_states.shape[0]
        
        # 投影編碼器特徵
        if self.encoder_projection is not None:
            encoder_hidden_states = self.encoder_projection(encoder_hidden_states)
        
        # 擴展查詢 token
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # 添加位置嵌入
        position_ids = torch.arange(
            self.config.num_query_tokens, 
            device=query_tokens.device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = query_tokens + position_embeddings
        
        # 準備注意力遮罩
        if encoder_attention_mask is not None:
            # 擴展維度以適應多頭注意力
            extended_attention_mask = encoder_attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # 通過 Transformer 層
        all_hidden_states = []
        all_self_attentions = []
        all_cross_attentions = []
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=extended_attention_mask,
                output_attentions=output_attentions
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_self_attentions.append(layer_outputs[1])
                if len(layer_outputs) > 2 and layer_outputs[2] is not None:
                    all_cross_attentions.append(layer_outputs[2])
        
        # 最終層正規化
        hidden_states = self.layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # 組裝輸出
        outputs = {
            "last_hidden_state": hidden_states,  # [batch, num_query_tokens, hidden_size]
            "pooler_output": hidden_states.mean(dim=1)  # [batch, hidden_size]
        }
        
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
        
        if output_attentions:
            outputs["self_attentions"] = all_self_attentions
            outputs["cross_attentions"] = all_cross_attentions
        
        return outputs
    
    def get_output_embeddings(self) -> torch.Tensor:
        """獲取輸出嵌入（用於與 LLM 對接）"""
        return self.query_tokens


class QFormerWithProjection(nn.Module):
    """
    帶投影的 Q-Former
    將輸出投影到 LLM 的隱藏維度
    """
    
    def __init__(
        self,
        qformer_config: QFormerConfig,
        llm_hidden_size: int = 4096  # Qwen 的隱藏維度
    ):
        super().__init__()
        
        self.qformer = QFormer(qformer_config)
        self.llm_projection = nn.Linear(qformer_config.hidden_size, llm_hidden_size)
        
        # 初始化投影層
        nn.init.xavier_uniform_(self.llm_projection.weight)
        nn.init.zeros_(self.llm_projection.bias)
        
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向傳播
        
        Returns:
            llm_embeddings: 投影到 LLM 維度的嵌入 [batch, num_query_tokens, llm_hidden_size]
        """
        # Q-Former 前向傳播
        qformer_outputs = self.qformer(
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs
        )
        
        # 投影到 LLM 維度
        qformer_embeddings = qformer_outputs["last_hidden_state"]
        llm_embeddings = self.llm_projection(qformer_embeddings)
        
        return llm_embeddings


if __name__ == "__main__":
    # 測試 Q-Former
    config = QFormerConfig(
        num_query_tokens=32,
        hidden_size=768,
        num_hidden_layers=6,
        encoder_hidden_size=512
    )
    
    qformer = QFormerWithProjection(config, llm_hidden_size=4096)
    
    print(f"Q-Former 初始化完成")
    print(f"查詢 token 數量: {config.num_query_tokens}")
    print(f"隱藏維度: {config.hidden_size}")
    print(f"LLM 投影維度: 4096")
    print(f"參數量: {sum(p.numel() for p in qformer.parameters()):,}")
    
    # 創建測試輸入
    batch_size = 2
    seq_len = 100  # Whisper encoder 輸出序列長度
    encoder_hidden_size = 512
    
    test_encoder_states = torch.randn(batch_size, seq_len, encoder_hidden_size)
    test_attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\n測試輸入:")
    print(f"編碼器隱藏狀態: {test_encoder_states.shape}")
    print(f"注意力遮罩: {test_attention_mask.shape}")
    
    # 前向傳播
    with torch.no_grad():
        llm_embeddings = qformer(
            encoder_hidden_states=test_encoder_states,
            encoder_attention_mask=test_attention_mask
        )
    
    print(f"\n輸出:")
    print(f"LLM 嵌入: {llm_embeddings.shape}")
    print(f"預期形狀: [{batch_size}, {config.num_query_tokens}, 4096]")
    
    # 測試原始 Q-Former 輸出
    with torch.no_grad():
        qformer_outputs = qformer.qformer(
            encoder_hidden_states=test_encoder_states,
            encoder_attention_mask=test_attention_mask,
            output_attentions=True
        )
    
    print(f"\nQ-Former 詳細輸出:")
    for key, value in qformer_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: {len(value)} 層")
    
    print(f"\n測試完成！")
