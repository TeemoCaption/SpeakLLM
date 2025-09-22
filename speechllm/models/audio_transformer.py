"""
音訊 Transformer 模組
用於從 LLM 隱藏狀態生成 RVQ token
實現 Voila 的多尺度分工策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import math
from dataclasses import dataclass


@dataclass
class AudioTransformerConfig:
    """音訊 Transformer 配置"""
    num_layers: int = 6  # Transformer 層數
    hidden_size: int = 512  # 隱藏維度
    num_attention_heads: int = 8  # 注意力頭數
    intermediate_size: int = 2048  # 前饋網路大小
    dropout_prob: float = 0.1  # Dropout 機率
    layer_norm_eps: float = 1e-12  # LayerNorm epsilon
    max_position_embeddings: int = 2048  # 最大位置嵌入
    
    # RVQ 相關配置
    num_rvq_layers: int = 4  # RVQ 層數
    codebook_size: int = 256  # 代碼簿大小
    
    # LLM 相關配置
    llm_hidden_size: int = 4096  # LLM 隱藏維度
    use_cross_attention: bool = True  # 是否使用交叉注意力
    
    # 多尺度策略
    hierarchical_generation: bool = True  # 是否使用階層式生成
    semantic_layer_idx: int = 0  # 語義層索引


class MultiHeadAttention(nn.Module):
    """多頭注意力機制"""
    
    def __init__(self, config: AudioTransformerConfig, is_cross_attention: bool = False):
        super().__init__()
        self.config = config
        self.is_cross_attention = is_cross_attention
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 查詢投影
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 鍵值投影
        if is_cross_attention:
            # 交叉注意力：K, V 來自 LLM
            self.key = nn.Linear(config.llm_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.llm_hidden_size, self.all_head_size)
        else:
            # 自注意力
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.dropout_prob)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """重塑張量以進行多頭注意力計算"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向傳播"""
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        if self.is_cross_attention and key_value_states is not None:
            key_layer = self.transpose_for_scores(self.key(key_value_states))
            value_layer = self.transpose_for_scores(self.value(key_value_states))
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 計算注意力分數
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        output = self.output_projection(context_layer)
        
        outputs = (output, attention_probs) if output_attentions else (output,)
        return outputs


class AudioTransformerLayer(nn.Module):
    """音訊 Transformer 層"""
    
    def __init__(self, config: AudioTransformerConfig):
        super().__init__()
        self.config = config
        
        # 自注意力
        self.self_attention = MultiHeadAttention(config, is_cross_attention=False)
        self.self_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 交叉注意力（條件於 LLM）
        if config.use_cross_attention:
            self.cross_attention = MultiHeadAttention(config, is_cross_attention=True)
            self.cross_attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 前饋網路
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout_prob)
        )
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        llm_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        llm_attention_mask: Optional[torch.Tensor] = None,
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
        hidden_states = self.self_attention_layer_norm(hidden_states + attention_output)
        
        # 交叉注意力
        cross_attention_probs = None
        if self.config.use_cross_attention and llm_hidden_states is not None:
            cross_attention_outputs = self.cross_attention(
                hidden_states=hidden_states,
                key_value_states=llm_hidden_states,
                attention_mask=llm_attention_mask,
                output_attentions=output_attentions
            )
            cross_attention_output = cross_attention_outputs[0]
            if output_attentions:
                cross_attention_probs = cross_attention_outputs[1]
            
            hidden_states = self.cross_attention_layer_norm(hidden_states + cross_attention_output)
        
        # 前饋網路
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(hidden_states + feed_forward_output)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (self_attention_outputs[1], cross_attention_probs)
        
        return outputs


class RVQTokenPredictor(nn.Module):
    """RVQ Token 預測器"""
    
    def __init__(self, config: AudioTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 預測頭
        self.prediction_head = nn.Linear(config.hidden_size, config.codebook_size)
        
        # 初始化
        nn.init.xavier_uniform_(self.prediction_head.weight)
        nn.init.zeros_(self.prediction_head.bias)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        預測 RVQ token
        
        Args:
            hidden_states: 隱藏狀態 [batch, seq_len, hidden_size]
            
        Returns:
            logits: 預測 logits [batch, seq_len, codebook_size]
        """
        return self.prediction_head(hidden_states)


class AudioTransformer(nn.Module):
    """
    音訊 Transformer
    從 LLM 隱藏狀態生成 RVQ token
    """
    
    def __init__(self, config: AudioTransformerConfig):
        super().__init__()
        self.config = config
        
        # 輸入投影（從 LLM 維度投影到音訊 Transformer 維度）
        self.input_projection = nn.Linear(config.llm_hidden_size, config.hidden_size)
        
        # 位置嵌入
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        
        # Transformer 層
        self.layers = nn.ModuleList([
            AudioTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # 層正規化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # RVQ 預測頭（每層一個）
        self.rvq_predictors = nn.ModuleList([
            RVQTokenPredictor(config, i) for i in range(config.num_rvq_layers)
        ])
        
        # 階層式生成的中間層（先語義 L1，再聲學 L2-L4）
        if config.hierarchical_generation:
            # 語義層（L1）到聲學層的投影
            self.semantic_to_acoustic_projection = nn.Linear(config.hidden_size, config.hidden_size)
            
            # 聲學層之間的投影
            self.intermediate_projections = nn.ModuleList([
                nn.Linear(config.hidden_size, config.hidden_size) 
                for _ in range(config.num_rvq_layers - 2)  # L2->L3, L3->L4
            ])
            
            # 語義層特殊處理
            self.semantic_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.acoustic_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
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
    
    def forward(
        self,
        llm_hidden_states: torch.Tensor,
        llm_attention_mask: Optional[torch.Tensor] = None,
        target_rvq_codes: Optional[List[torch.Tensor]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            llm_hidden_states: LLM 隱藏狀態 [batch, seq_len, llm_hidden_size]
            llm_attention_mask: LLM 注意力遮罩
            target_rvq_codes: 目標 RVQ codes（訓練時使用）
            output_attentions: 是否輸出注意力權重
            output_hidden_states: 是否輸出隱藏狀態
            
        Returns:
            outputs: 包含預測結果的字典
        """
        batch_size, seq_len = llm_hidden_states.shape[:2]
        
        # 輸入投影
        hidden_states = self.input_projection(llm_hidden_states)
        
        # 添加位置嵌入
        position_ids = torch.arange(seq_len, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # 準備注意力遮罩
        if llm_attention_mask is not None:
            extended_attention_mask = llm_attention_mask[:, None, None, :]
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
                llm_hidden_states=llm_hidden_states,
                attention_mask=extended_attention_mask,
                llm_attention_mask=extended_attention_mask,
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
        
        # 階層式生成 RVQ token 預測（先語義 L1，再聲學 L2-L4）
        rvq_predictions = []
        current_hidden = hidden_states
        
        for i, predictor in enumerate(self.rvq_predictors):
            # 預測當前層的 RVQ token
            logits = predictor(current_hidden)
            rvq_predictions.append(logits)
            
            # 階層式生成：先語義後聲學
            if self.config.hierarchical_generation and i < len(self.rvq_predictors) - 1:
                # 獲取預測的 token（訓練時使用真實值，推理時使用預測值）
                if target_rvq_codes is not None and i < len(target_rvq_codes):
                    # 訓練時使用真實值
                    predicted_tokens = target_rvq_codes[i]
                else:
                    # 推理時使用預測值
                    predicted_tokens = torch.argmax(logits, dim=-1)
                
                # 將預測的 token 嵌入
                token_embeddings = F.embedding(predicted_tokens, predictor.prediction_head.weight.t())
                
                if i == 0:
                    # L1（語義）-> L2（聲學）的特殊處理
                    semantic_features = self.semantic_layer_norm(token_embeddings)
                    acoustic_features = self.semantic_to_acoustic_projection(semantic_features)
                    current_hidden = current_hidden + self.acoustic_layer_norm(acoustic_features)
                else:
                    # L2->L3, L3->L4 的聲學層間投影
                    projection_idx = i - 1  # 調整索引
                    if projection_idx < len(self.intermediate_projections):
                        acoustic_features = self.intermediate_projections[projection_idx](token_embeddings)
                        current_hidden = current_hidden + acoustic_features
        
        # 組裝輸出
        outputs = {
            "rvq_predictions": rvq_predictions,  # List of [batch, seq_len, codebook_size]
            "last_hidden_state": hidden_states
        }
        
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
        
        if output_attentions:
            outputs["self_attentions"] = all_self_attentions
            outputs["cross_attentions"] = all_cross_attentions
        
        return outputs
    
    def generate_rvq_tokens(
        self,
        llm_hidden_states: torch.Tensor,
        llm_attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> List[torch.Tensor]:
        """
        生成 RVQ token
        
        Args:
            llm_hidden_states: LLM 隱藏狀態
            llm_attention_mask: 注意力遮罩
            temperature: 溫度參數
            top_k: Top-k 採樣
            top_p: Top-p 採樣
            
        Returns:
            generated_tokens: 生成的 RVQ token 列表
        """
        with torch.no_grad():
            outputs = self.forward(
                llm_hidden_states=llm_hidden_states,
                llm_attention_mask=llm_attention_mask
            )
            
            generated_tokens = []
            
            for logits in outputs["rvq_predictions"]:
                # 應用溫度
                logits = logits / temperature
                
                # Top-k 採樣
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Top-p 採樣
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累積機率超過 top_p 的 token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # 採樣
                probs = F.softmax(logits, dim=-1)
                tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])
                generated_tokens.append(tokens)
            
            return generated_tokens


if __name__ == "__main__":
    # 測試音訊 Transformer
    config = AudioTransformerConfig(
        num_layers=6,
        hidden_size=512,
        num_attention_heads=8,
        llm_hidden_size=4096,
        num_rvq_layers=4,
        codebook_size=256
    )
    
    audio_transformer = AudioTransformer(config)
    
    print(f"音訊 Transformer 初始化完成")
    print(f"層數: {config.num_layers}")
    print(f"隱藏維度: {config.hidden_size}")
    print(f"RVQ 層數: {config.num_rvq_layers}")
    print(f"代碼簿大小: {config.codebook_size}")
    print(f"參數量: {sum(p.numel() for p in audio_transformer.parameters()):,}")
    
    # 創建測試輸入
    batch_size = 2
    seq_len = 50
    llm_hidden_size = 4096
    
    test_llm_states = torch.randn(batch_size, seq_len, llm_hidden_size)
    test_attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\n測試輸入:")
    print(f"LLM 隱藏狀態: {test_llm_states.shape}")
    print(f"注意力遮罩: {test_attention_mask.shape}")
    
    # 前向傳播
    with torch.no_grad():
        outputs = audio_transformer(
            llm_hidden_states=test_llm_states,
            llm_attention_mask=test_attention_mask,
            output_attentions=True
        )
    
    print(f"\n輸出:")
    print(f"RVQ 預測數量: {len(outputs['rvq_predictions'])}")
    for i, pred in enumerate(outputs['rvq_predictions']):
        print(f"  層 {i+1}: {pred.shape}")
    
    print(f"最終隱藏狀態: {outputs['last_hidden_state'].shape}")
    
    # 測試生成
    print(f"\n測試生成:")
    generated_tokens = audio_transformer.generate_rvq_tokens(
        llm_hidden_states=test_llm_states,
        llm_attention_mask=test_attention_mask,
        temperature=0.8,
        top_k=50
    )
    
    print(f"生成的 token 數量: {len(generated_tokens)}")
    for i, tokens in enumerate(generated_tokens):
        print(f"  層 {i+1}: {tokens.shape}")
    
    print(f"\n測試完成！")
