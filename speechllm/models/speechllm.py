"""
SpeechLLM 主模型
整合 Qwen LLM、Whisper encoder、Q-Former 和音訊 Transformer
實現多模態語音對話系統
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import warnings

from .whisper_encoder import WhisperEncoderWithProjection
from .qformer import QFormerWithProjection, QFormerConfig
from .audio_transformer import AudioTransformer, AudioTransformerConfig
from ..codecs.vocab_manager import VocabManager
from ..codecs.audio_tokenizer import AudioTokenizer


@dataclass
class SpeechLLMConfig:
    """SpeechLLM 配置"""
    # LLM 配置
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    freeze_llm: bool = False
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    
    # Whisper encoder 配置
    whisper_model_name: str = "openai/whisper-base"
    freeze_whisper: bool = True
    
    # Q-Former 配置
    num_query_tokens: int = 32
    qformer_hidden_size: int = 768
    qformer_num_layers: int = 6
    
    # 音訊 Transformer 配置
    audio_transformer_layers: int = 6
    audio_transformer_hidden_size: int = 512
    
    # RVQ 配置
    num_rvq_layers: int = 4
    codebook_size: int = 256
    
    # 訓練配置
    use_gradient_checkpointing: bool = True
    mixed_precision: bool = True


class SpeechLLM(nn.Module):
    """
    SpeechLLM 主模型
    
    架構：
    1. 輸入端：Whisper encoder -> Q-Former -> LLM
    2. 輸出端：LLM -> 音訊 Transformer -> RVQ tokens
    3. 支援四種任務模式：TITO, AITO, TIAO, AIAO
    """
    
    def __init__(self, config: SpeechLLMConfig):
        super().__init__()
        self.config = config
        
        # 初始化詞彙表管理器
        self.vocab_manager = VocabManager(
            base_tokenizer_name=config.llm_model_name,
            num_rvq_layers=config.num_rvq_layers,
            codebook_size=config.codebook_size
        )
        
        # 載入並擴展 LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_name,
            torch_dtype=torch.float16 if config.mixed_precision else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 擴展詞彙表
        self.llm.resize_token_embeddings(len(self.vocab_manager.extended_tokenizer))
        
        # 凍結 LLM（可選）
        if config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # LoRA 配置（可選）
        if config.use_lora:
            self._setup_lora()
        
        # 獲取 LLM 隱藏維度
        self.llm_hidden_size = self.llm.config.hidden_size
        
        # Whisper encoder
        self.whisper_encoder = WhisperEncoderWithProjection(
            whisper_model_name=config.whisper_model_name,
            output_dim=config.qformer_hidden_size,
            freeze_whisper=config.freeze_whisper
        )
        
        # Q-Former
        qformer_config = QFormerConfig(
            num_query_tokens=config.num_query_tokens,
            hidden_size=config.qformer_hidden_size,
            num_hidden_layers=config.qformer_num_layers,
            encoder_hidden_size=config.qformer_hidden_size
        )
        self.qformer = QFormerWithProjection(qformer_config, self.llm_hidden_size)
        
        # 音訊 Transformer
        audio_transformer_config = AudioTransformerConfig(
            num_layers=config.audio_transformer_layers,
            hidden_size=config.audio_transformer_hidden_size,
            llm_hidden_size=self.llm_hidden_size,
            num_rvq_layers=config.num_rvq_layers,
            codebook_size=config.codebook_size
        )
        self.audio_transformer = AudioTransformer(audio_transformer_config)
        
        # 音訊 tokenizer
        self.audio_tokenizer = AudioTokenizer(vocab_manager=self.vocab_manager)
        
        # 狀態分類頭（全雙工用）
        self.state_classifier = nn.Linear(self.llm_hidden_size, 3)  # 0: 續聽, 1: 需打斷, 2: 不打斷
        
        # 梯度檢查點
        if config.use_gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
        
    def _setup_lora(self):
        """設置 LoRA"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            self.llm = get_peft_model(self.llm, lora_config)
            print(f"LoRA 配置完成，可訓練參數: {self.llm.print_trainable_parameters()}")
            
        except ImportError:
            warnings.warn("PEFT 未安裝，跳過 LoRA 配置")
            self.config.use_lora = False
    
    def encode_audio(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        編碼音訊為 LLM 可理解的嵌入
        
        Args:
            audio: 音訊輸入 [batch, time] 或 [batch, n_mels, time]
            attention_mask: 注意力遮罩
            
        Returns:
            audio_embeddings: 音訊嵌入 [batch, num_query_tokens, llm_hidden_size]
        """
        # Whisper 編碼
        whisper_outputs = self.whisper_encoder(
            audio=audio,
            attention_mask=attention_mask
        )
        
        # Q-Former 聚合
        audio_embeddings = self.qformer(
            encoder_hidden_states=whisper_outputs["hidden_states"],
            encoder_attention_mask=whisper_outputs["attention_mask"]
        )
        
        return audio_embeddings
    
    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        audio_embeddings: Optional[torch.Tensor] = None,
        audio_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        準備輸入嵌入，將音訊嵌入插入到對應位置
        
        Args:
            input_ids: 輸入 token ID [batch, seq_len]
            audio_embeddings: 音訊嵌入 [batch, num_audio_tokens, hidden_size]
            audio_positions: 音訊嵌入插入位置
            
        Returns:
            inputs_embeds: 混合嵌入 [batch, total_seq_len, hidden_size]
        """
        # 獲取文字嵌入
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        if audio_embeddings is None or audio_positions is None:
            return text_embeddings
        
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device
        
        # 為每個批次樣本插入音訊嵌入
        mixed_embeddings = []
        
        for b in range(batch_size):
            text_emb = text_embeddings[b]  # [seq_len, hidden_size]
            audio_emb = audio_embeddings[b]  # [num_audio_tokens, hidden_size]
            
            # 找到音訊標記位置
            audio_token_id = self.vocab_manager.get_token_id("<AUDIO>")
            audio_mask = (input_ids[b] == audio_token_id)
            
            if audio_mask.any() and len(audio_positions) > b:
                # 在指定位置插入音訊嵌入
                pos = audio_positions[b]
                mixed_emb = torch.cat([
                    text_emb[:pos],
                    audio_emb,
                    text_emb[pos:]
                ], dim=0)
            else:
                mixed_emb = text_emb
            
            mixed_embeddings.append(mixed_emb)
        
        # 填充到相同長度
        max_len = max(emb.shape[0] for emb in mixed_embeddings)
        padded_embeddings = []
        
        for emb in mixed_embeddings:
            if emb.shape[0] < max_len:
                padding = torch.zeros(
                    max_len - emb.shape[0], emb.shape[1],
                    device=device, dtype=emb.dtype
                )
                emb = torch.cat([emb, padding], dim=0)
            padded_embeddings.append(emb)
        
        return torch.stack(padded_embeddings)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        rvq_labels: Optional[List[torch.Tensor]] = None,
        mode: str = "TITO",
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播
        
        Args:
            input_ids: 輸入 token ID
            inputs_embeds: 輸入嵌入（可選）
            attention_mask: 注意力遮罩
            audio: 音訊輸入
            audio_attention_mask: 音訊注意力遮罩
            labels: 文字標籤
            rvq_labels: RVQ 標籤
            mode: 任務模式
            output_attentions: 是否輸出注意力
            output_hidden_states: 是否輸出隱藏狀態
            return_dict: 是否返回字典
            
        Returns:
            outputs: 模型輸出
        """
        # 處理音訊輸入
        audio_embeddings = None
        if audio is not None and mode in ["AITO", "AIAO"]:
            audio_embeddings = self.encode_audio(audio, audio_attention_mask)
        
        # 準備輸入嵌入
        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.prepare_inputs_embeds(
                    input_ids=input_ids,
                    audio_embeddings=audio_embeddings
                )
            else:
                raise ValueError("必須提供 input_ids 或 inputs_embeds")
        
        # LLM 前向傳播
        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        outputs = {
            "loss": llm_outputs.loss,
            "logits": llm_outputs.logits,
            "hidden_states": llm_outputs.hidden_states,
            "attentions": llm_outputs.attentions
        }
        
        # 音訊輸出生成
        if mode in ["TIAO", "AIAO"]:
            audio_outputs = self.audio_transformer(
                llm_hidden_states=llm_outputs.hidden_states[-1],
                llm_attention_mask=attention_mask,
                target_rvq_codes=rvq_labels,
                output_attentions=output_attentions
            )
            
            outputs["rvq_predictions"] = audio_outputs["rvq_predictions"]
            outputs["audio_hidden_states"] = audio_outputs["last_hidden_state"]
            
            # 計算 RVQ 損失
            if rvq_labels is not None:
                rvq_loss = 0
                for pred, target in zip(audio_outputs["rvq_predictions"], rvq_labels):
                    rvq_loss += F.cross_entropy(
                        pred.view(-1, pred.size(-1)),
                        target.view(-1),
                        ignore_index=-100
                    )
                outputs["rvq_loss"] = rvq_loss / len(rvq_labels)
                
                # 總損失
                if outputs["loss"] is not None:
                    outputs["loss"] = outputs["loss"] + outputs["rvq_loss"]
                else:
                    outputs["loss"] = outputs["rvq_loss"]
        
        # 狀態分類（全雙工用）
        if llm_outputs.hidden_states is not None:
            state_logits = self.state_classifier(llm_outputs.hidden_states[-1])
            outputs["state_logits"] = state_logits
        
        return outputs
    
    def generate_speech(
        self,
        input_ids: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        生成語音回應
        
        Args:
            input_ids: 輸入 token
            audio: 音訊輸入（可選）
            max_length: 最大生成長度
            temperature: 溫度參數
            top_k: Top-k 採樣
            top_p: Top-p 採樣
            do_sample: 是否採樣
            
        Returns:
            generated_outputs: 生成結果
        """
        with torch.no_grad():
            # 編碼音訊
            audio_embeddings = None
            if audio is not None:
                audio_embeddings = self.encode_audio(audio)
            
            # 準備輸入
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                audio_embeddings=audio_embeddings
            )
            
            # 生成文字
            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.vocab_manager.extended_tokenizer.pad_token_id,
                eos_token_id=self.vocab_manager.get_token_id("<eos>"),
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            
            # 生成音訊 token
            last_hidden_states = generated_ids.hidden_states[-1][-1]  # 最後一層的最後一個狀態
            
            rvq_tokens = self.audio_transformer.generate_rvq_tokens(
                llm_hidden_states=last_hidden_states.unsqueeze(0),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            return {
                "generated_ids": generated_ids.sequences,
                "rvq_tokens": rvq_tokens,
                "hidden_states": generated_ids.hidden_states
            }
    
    def get_trainable_parameters(self) -> int:
        """獲取可訓練參數數量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_pretrained(self, save_directory: str):
        """保存模型"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存模型權重
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # 保存配置
        import json
        config_dict = {
            "llm_model_name": self.config.llm_model_name,
            "whisper_model_name": self.config.whisper_model_name,
            "num_query_tokens": self.config.num_query_tokens,
            "num_rvq_layers": self.config.num_rvq_layers,
            "codebook_size": self.config.codebook_size
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # 保存詞彙表
        self.vocab_manager.save_vocab(os.path.join(save_directory, "vocab.json"))
        
        print(f"模型已保存到 {save_directory}")


if __name__ == "__main__":
    # 測試 SpeechLLM
    config = SpeechLLMConfig(
        llm_model_name="Qwen/Qwen2.5-7B-Instruct",
        use_lora=True,
        freeze_llm=True
    )
    
    print("初始化 SpeechLLM...")
    model = SpeechLLM(config)
    
    print(f"模型初始化完成")
    print(f"總參數量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可訓練參數量: {model.get_trainable_parameters():,}")
    
    # 創建測試輸入
    batch_size = 1
    seq_len = 50
    audio_length = 16000 * 3  # 3 秒音訊
    
    test_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    test_audio = torch.randn(batch_size, audio_length)
    test_attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\n測試輸入:")
    print(f"Input IDs: {test_input_ids.shape}")
    print(f"Audio: {test_audio.shape}")
    print(f"Attention mask: {test_attention_mask.shape}")
    
    # 測試前向傳播（AITO 模式）
    print(f"\n測試 AITO 模式:")
    with torch.no_grad():
        outputs = model(
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            audio=test_audio,
            mode="AITO"
        )
    
    print(f"Logits: {outputs['logits'].shape}")
    if outputs['loss'] is not None:
        print(f"Loss: {outputs['loss'].item():.4f}")
    
    # 測試 AIAO 模式
    print(f"\n測試 AIAO 模式:")
    mock_rvq_labels = [torch.randint(0, 256, (batch_size, seq_len)) for _ in range(4)]
    
    with torch.no_grad():
        outputs = model(
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
            audio=test_audio,
            rvq_labels=mock_rvq_labels,
            mode="AIAO"
        )
    
    print(f"RVQ predictions: {len(outputs['rvq_predictions'])}")
    for i, pred in enumerate(outputs['rvq_predictions']):
        print(f"  Layer {i+1}: {pred.shape}")
    
    if 'rvq_loss' in outputs:
        print(f"RVQ Loss: {outputs['rvq_loss'].item():.4f}")
    
    print(f"\n測試完成！")
