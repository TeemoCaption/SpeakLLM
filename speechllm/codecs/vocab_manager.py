"""
詞彙表管理器
管理特殊 token 和 RVQ codec token
"""

from typing import Dict, List, Optional, Tuple
import json
import os
from transformers import AutoTokenizer


class VocabManager:
    """
    管理 SpeechLLM 的詞彙表，包含：
    1. 角色標籤：<human>, <assistant>, <eos>
    2. 模式標籤：<AUDIO>, <TEXT>, <AUDIO_START>, <AUDIO_END>
    3. RVQ 代碼簿 token：<A_L1_000> ~ <A_L4_255>
    4. 說話人嵌入 token：<CHAT_REF_START>, <CHAT_REF>, <CHAT_REF_END>
    """
    
    def __init__(
        self,
        base_tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
        num_rvq_layers: int = 4,
        codebook_size: int = 256,
        add_speaker_tokens: bool = True
    ):
        self.base_tokenizer_name = base_tokenizer_name
        self.num_rvq_layers = num_rvq_layers
        self.codebook_size = codebook_size
        self.add_speaker_tokens = add_speaker_tokens
        
        # 載入基礎 tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
        
        # 定義特殊 token
        self.special_tokens = self._create_special_tokens()
        
        # 擴展詞彙表
        self.extended_tokenizer = self._extend_tokenizer()
        
    def _create_special_tokens(self) -> Dict[str, List[str]]:
        """創建所有特殊 token"""
        special_tokens = {
            # 角色標籤
            "role_tokens": ["<human>", "<assistant>", "<eos>"],
            
            # 模式標籤
            "mode_tokens": ["<AUDIO>", "<TEXT>", "<AUDIO_START>", "<AUDIO_END>"],
            
            # RVQ 代碼簿 token
            "rvq_tokens": [],
            
            # 說話人嵌入 token（可選）
            "speaker_tokens": []
        }
        
        # 生成 RVQ token：<A_L1_000> ~ <A_L4_255>
        for layer in range(1, self.num_rvq_layers + 1):
            for code in range(self.codebook_size):
                token = f"<A_L{layer}_{code:03d}>"
                special_tokens["rvq_tokens"].append(token)
        
        # 說話人嵌入 token
        if self.add_speaker_tokens:
            special_tokens["speaker_tokens"] = [
                "<CHAT_REF_START>", "<CHAT_REF>", "<CHAT_REF_END>"
            ]
        
        return special_tokens
    
    def _extend_tokenizer(self) -> AutoTokenizer:
        """擴展基礎 tokenizer 以包含特殊 token"""
        # 收集所有特殊 token
        all_special_tokens = []
        for token_group in self.special_tokens.values():
            all_special_tokens.extend(token_group)
        
        # 創建新的 tokenizer（Qwen 的 SentencePiece 已覆蓋中英混排）
        extended_tokenizer = AutoTokenizer.from_pretrained(self.base_tokenizer_name)
        
        # 確保有 pad_token
        if extended_tokenizer.pad_token is None:
            extended_tokenizer.pad_token = extended_tokenizer.eos_token
        
        # 添加特殊 token
        num_added = extended_tokenizer.add_special_tokens({
            "additional_special_tokens": all_special_tokens
        })
        
        print(f"添加了 {num_added} 個特殊 token 到 Qwen tokenizer")
        print(f"原始詞彙表大小: {len(extended_tokenizer) - num_added}")
        print(f"擴展後詞彙表大小: {len(extended_tokenizer)}")
        
        return extended_tokenizer
    
    def get_token_id(self, token: str) -> int:
        """獲取 token 的 ID"""
        return self.extended_tokenizer.convert_tokens_to_ids(token)
    
    def get_token_ids(self, tokens: List[str]) -> List[int]:
        """獲取多個 token 的 ID"""
        return self.extended_tokenizer.convert_tokens_to_ids(tokens)
    
    def get_rvq_token_ids(self, layer: int) -> List[int]:
        """獲取特定層的所有 RVQ token ID"""
        layer_tokens = [f"<A_L{layer}_{code:03d}>" for code in range(self.codebook_size)]
        return self.get_token_ids(layer_tokens)
    
    def rvq_codes_to_tokens(self, codes: List[List[int]]) -> List[str]:
        """
        將 RVQ codes 轉換為 token
        
        Args:
            codes: shape [num_layers, seq_len] 的 RVQ codes
            
        Returns:
            List of token strings
        """
        tokens = []
        seq_len = len(codes[0])
        
        for t in range(seq_len):
            for layer in range(self.num_rvq_layers):
                code = codes[layer][t]
                token = f"<A_L{layer+1}_{code:03d}>"
                tokens.append(token)
        
        return tokens
    
    def tokens_to_rvq_codes(self, tokens: List[str]) -> List[List[int]]:
        """
        將 token 轉換為 RVQ codes
        
        Args:
            tokens: List of RVQ token strings
            
        Returns:
            codes: shape [num_layers, seq_len] 的 RVQ codes
        """
        # 過濾出 RVQ token
        rvq_tokens = [t for t in tokens if t.startswith("<A_L") and t.endswith(">")]
        
        # 計算序列長度
        seq_len = len(rvq_tokens) // self.num_rvq_layers
        
        # 初始化 codes
        codes = [[] for _ in range(self.num_rvq_layers)]
        
        # 解析 token
        for i, token in enumerate(rvq_tokens):
            # 解析 token：<A_L{layer}_{code}>
            parts = token[3:-1].split("_")  # 移除 "<A_" 和 ">"
            layer = int(parts[1][1:]) - 1  # L1 -> 0, L2 -> 1, ...
            code = int(parts[2])
            
            codes[layer].append(code)
        
        return codes
    
    def create_interleaved_sequence(self, text_tokens: List[str], audio_tokens: List[str]) -> List[str]:
        """
        創建交錯序列（Voila 策略）
        每個文字 token 重複 4 次以對齊 4 層 RVQ
        
        Args:
            text_tokens: 文字 token 列表
            audio_tokens: 音訊 token 列表
            
        Returns:
            交錯的 token 序列
        """
        interleaved = []
        
        # 文字部分：每個 token 重複 4 次
        for token in text_tokens:
            interleaved.extend([token] * self.num_rvq_layers)
        
        # 音訊部分：直接添加
        interleaved.extend(audio_tokens)
        
        return interleaved
    
    def save_vocab(self, save_path: str):
        """保存詞彙表配置"""
        vocab_config = {
            "base_tokenizer_name": self.base_tokenizer_name,
            "num_rvq_layers": self.num_rvq_layers,
            "codebook_size": self.codebook_size,
            "add_speaker_tokens": self.add_speaker_tokens,
            "special_tokens": self.special_tokens,
            "vocab_size": len(self.extended_tokenizer)
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_config, f, ensure_ascii=False, indent=2)
        
        # 保存 tokenizer
        tokenizer_path = save_path.replace('.json', '_tokenizer')
        self.extended_tokenizer.save_pretrained(tokenizer_path)
    
    @classmethod
    def from_pretrained(cls, vocab_path: str) -> 'VocabManager':
        """從保存的配置載入詞彙表管理器"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return cls(
            base_tokenizer_name=config["base_tokenizer_name"],
            num_rvq_layers=config["num_rvq_layers"],
            codebook_size=config["codebook_size"],
            add_speaker_tokens=config["add_speaker_tokens"]
        )
    
    def get_vocab_info(self) -> Dict:
        """獲取詞彙表資訊"""
        info = {
            "base_vocab_size": len(self.base_tokenizer),
            "extended_vocab_size": len(self.extended_tokenizer),
            "num_special_tokens": len(self.extended_tokenizer) - len(self.base_tokenizer),
            "num_rvq_tokens": self.num_rvq_layers * self.codebook_size,
            "special_token_groups": {
                group: len(tokens) for group, tokens in self.special_tokens.items()
            }
        }
        return info


if __name__ == "__main__":
    # 測試詞彙表管理器
    vocab_manager = VocabManager()
    
    print("詞彙表資訊:")
    info = vocab_manager.get_vocab_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n特殊 token 範例:")
    print(f"  <human> ID: {vocab_manager.get_token_id('<human>')}")
    print(f"  <AUDIO> ID: {vocab_manager.get_token_id('<AUDIO>')}")
    print(f"  <A_L1_000> ID: {vocab_manager.get_token_id('<A_L1_000>')}")
    print(f"  <A_L4_255> ID: {vocab_manager.get_token_id('<A_L4_255>')}")
    
    # 測試 RVQ codes 轉換
    print("\nRVQ codes 轉換測試:")
    test_codes = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    tokens = vocab_manager.rvq_codes_to_tokens(test_codes)
    print(f"  Codes: {test_codes}")
    print(f"  Tokens: {tokens[:12]}...")  # 顯示前 12 個 token
    
    recovered_codes = vocab_manager.tokens_to_rvq_codes(tokens)
    print(f"  Recovered: {recovered_codes}")
    print(f"  Match: {test_codes == recovered_codes}")
