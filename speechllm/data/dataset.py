"""
SpeechLLM 資料集
支援多任務格式：TITO, AITO, TIAO, AIAO
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import random
from pathlib import Path

from ..codecs.audio_tokenizer import AudioTokenizer
from ..codecs.vocab_manager import VocabManager
from ..align.interleaving import InterleavingGenerator


@dataclass
class DataSample:
    """資料樣本"""
    sample_id: str
    mode: str  # TITO, AITO, TIAO, AIAO
    input_text: Optional[str] = None
    input_audio_path: Optional[str] = None
    output_text: Optional[str] = None
    output_audio_path: Optional[str] = None
    metadata: Optional[Dict] = None


class SpeechLLMDataset(Dataset):
    """
    SpeechLLM 資料集
    支援四種任務模式的統一處理
    """
    
    def __init__(
        self,
        data_file: str,
        audio_tokenizer: Optional[AudioTokenizer] = None,
        vocab_manager: Optional[VocabManager] = None,
        interleaving_generator: Optional[InterleavingGenerator] = None,
        max_text_length: int = 512,
        max_audio_length: int = 16000 * 10,  # 10 秒
        sample_rate: int = 16000,
        mode_weights: Optional[Dict[str, float]] = None,
        cache_audio_tokens: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.data_file = data_file
        self.max_text_length = max_text_length
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate
        self.cache_audio_tokens = cache_audio_tokens
        self.cache_dir = cache_dir
        
        # 初始化組件
        self.audio_tokenizer = audio_tokenizer or AudioTokenizer()
        self.vocab_manager = vocab_manager or VocabManager()
        self.interleaving_generator = interleaving_generator or InterleavingGenerator(
            audio_tokenizer=self.audio_tokenizer,
            vocab_manager=self.vocab_manager
        )
        
        # 模式權重（用於平衡不同任務）
        self.mode_weights = mode_weights or {
            "TITO": 0.3,
            "AITO": 0.3, 
            "TIAO": 0.2,
            "AIAO": 0.2
        }
        
        # 載入資料
        self.samples = self._load_data()
        
        # 建立快取目錄
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"載入 {len(self.samples)} 個樣本")
        self._print_data_statistics()
    
    def _load_data(self) -> List[DataSample]:
        """載入資料"""
        samples = []
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            sample = DataSample(
                sample_id=item.get("sample_id", str(len(samples))),
                mode=item["mode"],
                input_text=item.get("input_text"),
                input_audio_path=item.get("input_audio_path"),
                output_text=item.get("output_text"),
                output_audio_path=item.get("output_audio_path"),
                metadata=item.get("metadata", {})
            )
            samples.append(sample)
        
        return samples
    
    def _print_data_statistics(self):
        """印出資料統計"""
        mode_counts = {}
        for sample in self.samples:
            mode_counts[sample.mode] = mode_counts.get(sample.mode, 0) + 1
        
        print("資料統計:")
        for mode, count in mode_counts.items():
            percentage = count / len(self.samples) * 100
            print(f"  {mode}: {count} ({percentage:.1f}%)")
    
    def _get_cache_path(self, sample_id: str, data_type: str) -> str:
        """獲取快取路徑"""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{sample_id}_{data_type}.pt")
    
    def _load_cached_tokens(self, cache_path: str) -> Optional[List[str]]:
        """載入快取的 token"""
        if not cache_path or not os.path.exists(cache_path):
            return None
        
        try:
            return torch.load(cache_path)
        except:
            return None
    
    def _save_cached_tokens(self, cache_path: str, tokens: List[str]):
        """保存 token 到快取"""
        if not cache_path:
            return
        
        try:
            torch.save(tokens, cache_path)
        except:
            pass
    
    def _process_audio(self, audio_path: str, sample_id: str) -> List[str]:
        """處理音訊文件"""
        # 檢查快取
        cache_path = self._get_cache_path(sample_id, "audio_tokens")
        if self.cache_audio_tokens:
            cached_tokens = self._load_cached_tokens(cache_path)
            if cached_tokens is not None:
                return cached_tokens
        
        # 載入和處理音訊
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # 截斷或填充到固定長度
            if len(audio) > self.max_audio_length:
                audio = audio[:self.max_audio_length]
            elif len(audio) < self.max_audio_length:
                audio = np.pad(audio, (0, self.max_audio_length - len(audio)))
            
            # 轉換為 token
            audio_tensor = torch.from_numpy(audio).float()
            tokens = self.audio_tokenizer.audio_to_tokens(audio_tensor)
            
            # 保存快取
            if self.cache_audio_tokens:
                self._save_cached_tokens(cache_path, tokens)
            
            return tokens
            
        except Exception as e:
            print(f"處理音訊文件 {audio_path} 時出錯: {e}")
            # 返回空 token 序列
            return []
    
    def _create_training_sequence(self, sample: DataSample) -> Dict[str, List[str]]:
        """創建訓練序列"""
        input_tokens = []
        output_tokens = []
        
        # 處理輸入
        if sample.mode in ["TITO", "TIAO"]:  # 文字輸入
            if sample.input_text:
                input_tokens = self.audio_tokenizer.create_chat_format(
                    text=sample.input_text,
                    role="human",
                    mode=sample.mode
                )
        
        elif sample.mode in ["AITO", "AIAO"]:  # 音訊輸入
            if sample.input_audio_path:
                input_tokens = self.audio_tokenizer.create_chat_format(
                    text="",
                    audio_path=sample.input_audio_path,
                    role="human",
                    mode=sample.mode
                )
        
        # 處理輸出
        if sample.mode in ["TITO", "AITO"]:  # 文字輸出
            if sample.output_text:
                output_tokens = self.audio_tokenizer.create_response_format(
                    text=sample.output_text,
                    mode=sample.mode
                )
        
        elif sample.mode in ["TIAO", "AIAO"]:  # 音訊輸出
            if sample.output_audio_path:
                # 使用交錯生成器創建輸出序列
                output_tokens = self.interleaving_generator.generate_interleaved_sequence(
                    audio_path=sample.output_audio_path,
                    reference_text=sample.output_text,
                    mode=sample.mode,
                    role="assistant"
                )
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "full_sequence": input_tokens + output_tokens
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """獲取單個樣本"""
        sample = self.samples[idx]
        
        # 創建訓練序列
        sequences = self._create_training_sequence(sample)
        
        # 轉換為 token ID
        full_sequence = sequences["full_sequence"]
        input_length = len(sequences["input_tokens"])
        
        # 截斷序列
        if len(full_sequence) > self.max_text_length:
            full_sequence = full_sequence[:self.max_text_length]
            input_length = min(input_length, self.max_text_length)
        
        # 轉換為 ID
        token_ids = self.vocab_manager.get_token_ids(full_sequence)
        
        # 創建標籤（只對輸出部分計算損失）
        labels = [-100] * input_length + token_ids[input_length:]
        labels = labels[:len(token_ids)]  # 確保長度一致
        
        # 填充到最大長度
        padding_length = self.max_text_length - len(token_ids)
        if padding_length > 0:
            pad_token_id = self.vocab_manager.extended_tokenizer.pad_token_id or 0
            token_ids.extend([pad_token_id] * padding_length)
            labels.extend([-100] * padding_length)
        
        # 創建注意力遮罩
        attention_mask = [1] * (len(token_ids) - padding_length) + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "mode": sample.mode,
            "sample_id": sample.sample_id
        }


class SpeechLLMCollator:
    """
    SpeechLLM 資料整理器
    處理批次資料的填充和對齊
    """
    
    def __init__(
        self,
        vocab_manager: VocabManager,
        pad_to_multiple_of: Optional[int] = None
    ):
        self.vocab_manager = vocab_manager
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = vocab_manager.extended_tokenizer.pad_token_id or 0
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """整理批次資料"""
        # 獲取最大長度
        max_length = max(len(item["input_ids"]) for item in batch)
        
        # 調整到指定倍數
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # 填充所有序列
        input_ids = []
        attention_masks = []
        labels = []
        modes = []
        sample_ids = []
        
        for item in batch:
            # 填充 input_ids
            current_length = len(item["input_ids"])
            if current_length < max_length:
                padding = torch.full((max_length - current_length,), 
                                   self.pad_token_id, dtype=torch.long)
                padded_input_ids = torch.cat([item["input_ids"], padding])
            else:
                padded_input_ids = item["input_ids"][:max_length]
            
            # 填充 attention_mask
            if current_length < max_length:
                padding = torch.zeros(max_length - current_length, dtype=torch.long)
                padded_attention_mask = torch.cat([item["attention_mask"], padding])
            else:
                padded_attention_mask = item["attention_mask"][:max_length]
            
            # 填充 labels
            if current_length < max_length:
                padding = torch.full((max_length - current_length,), -100, dtype=torch.long)
                padded_labels = torch.cat([item["labels"], padding])
            else:
                padded_labels = item["labels"][:max_length]
            
            input_ids.append(padded_input_ids)
            attention_masks.append(padded_attention_mask)
            labels.append(padded_labels)
            modes.append(item["mode"])
            sample_ids.append(item["sample_id"])
        
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
            "modes": modes,
            "sample_ids": sample_ids
        }


def create_dataloader(
    dataset: SpeechLLMDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn: Optional[SpeechLLMCollator] = None
) -> DataLoader:
    """創建資料載入器"""
    
    if collate_fn is None:
        collate_fn = SpeechLLMCollator(dataset.vocab_manager)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    # 測試資料集
    
    # 創建測試資料
    test_data = [
        {
            "sample_id": "test_001",
            "mode": "TITO",
            "input_text": "你好，今天天氣如何？",
            "output_text": "今天天氣很好，陽光明媚。"
        },
        {
            "sample_id": "test_002", 
            "mode": "AITO",
            "input_audio_path": "test_audio.wav",
            "output_text": "我聽到了你的問題。"
        },
        {
            "sample_id": "test_003",
            "mode": "TIAO",
            "input_text": "請說一句話。",
            "output_audio_path": "output_audio.wav",
            "output_text": "這是我的回應。"
        }
    ]
    
    # 保存測試資料
    test_file = "test_data.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"創建測試資料文件: {test_file}")
    
    # 創建資料集
    try:
        dataset = SpeechLLMDataset(
            data_file=test_file,
            max_text_length=256,
            cache_audio_tokens=False  # 測試時不使用快取
        )
        
        print(f"資料集創建成功，包含 {len(dataset)} 個樣本")
        
        # 測試單個樣本
        sample = dataset[0]
        print(f"\n測試樣本:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} {value.dtype}")
            else:
                print(f"  {key}: {value}")
        
        # 測試資料載入器
        dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)
        
        print(f"\n測試批次:")
        for batch in dataloader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {len(value)} items")
            break
        
        print("資料集測試完成！")
        
    except Exception as e:
        print(f"測試過程中出錯: {e}")
    
    finally:
        # 清理測試文件
        if os.path.exists(test_file):
            os.remove(test_file)
