"""
交錯標註產生器
實現 Voila 風格的語音-文字交錯序列生成
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import librosa
import torchaudio

from .alignment import AlignmentSegment, WhisperAligner
from ..codecs.audio_tokenizer import AudioTokenizer
from ..codecs.vocab_manager import VocabManager


@dataclass
class InterleavedSegment:
    """交錯片段"""
    text_tokens: List[str]  # 文字 token（重複 4 次）
    audio_tokens: List[str]  # 音訊 token（RVQ codes）
    start_time: float  # 開始時間
    end_time: float  # 結束時間
    alignment_confidence: float = 1.0  # 對齊置信度


class InterleavingGenerator:
    """
    交錯標註產生器
    
    核心功能：
    1. ASR 對齊獲取詞級時間戳
    2. RVQ codec 將語音轉為離散 token
    3. 時間映射切出對應的 codec token 片段
    4. 文字 token 重複 4 次對齊 RVQ 層數
    5. 組裝交錯序列
    """
    
    def __init__(
        self,
        audio_tokenizer: Optional[AudioTokenizer] = None,
        aligner: Optional[WhisperAligner] = None,
        vocab_manager: Optional[VocabManager] = None,
        num_rvq_layers: int = 4,
        repeat_text_tokens: bool = True
    ):
        # 初始化組件
        self.audio_tokenizer = audio_tokenizer or AudioTokenizer()
        self.aligner = aligner or WhisperAligner()
        self.vocab_manager = vocab_manager or VocabManager()
        
        self.num_rvq_layers = num_rvq_layers
        self.repeat_text_tokens = repeat_text_tokens
        
    def generate_interleaved_sequence(
        self,
        audio_path: str,
        reference_text: Optional[str] = None,
        mode: str = "AIAO",  # TITO, AITO, TIAO, AIAO
        role: str = "assistant"
    ) -> List[str]:
        """
        生成交錯序列
        
        Args:
            audio_path: 音訊文件路徑
            reference_text: 參考文字（可選，用於強制對齊）
            mode: 任務模式
            role: 角色標籤
            
        Returns:
            interleaved_tokens: 交錯的 token 序列
        """
        # 1. ASR 對齊獲取時間戳
        if reference_text:
            # 強制對齊
            alignment_segments = self.aligner.force_align(audio_path, reference_text)
        else:
            # 自動轉錄對齊
            alignment_segments = self.aligner.align_audio_text(audio_path, word_level=True)
        
        # 2. 音訊編碼為 RVQ token
        audio_tokens = self.audio_tokenizer.audio_to_tokens(audio_path)
        
        # 3. 時間映射
        interleaved_segments = self._create_time_aligned_segments(
            alignment_segments, audio_tokens, audio_path
        )
        
        # 4. 組裝完整序列
        full_sequence = self._assemble_full_sequence(
            interleaved_segments, mode, role
        )
        
        return full_sequence
    
    def _create_time_aligned_segments(
        self,
        alignment_segments: List[AlignmentSegment],
        audio_tokens: List[str],
        audio_path: str
    ) -> List[InterleavedSegment]:
        """
        創建時間對齊的交錯片段
        
        Args:
            alignment_segments: 對齊片段
            audio_tokens: 音訊 token 序列
            audio_path: 音訊文件路徑
            
        Returns:
            interleaved_segments: 交錯片段列表
        """
        # 載入音訊獲取時長信息
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration = len(audio) / sr
        
        # 計算音訊 token 的時間解析度
        # 假設 RVQ codec 的跳躍長度為 320 (20ms at 16kHz)
        hop_length = 320
        frame_duration = hop_length / sr  # 每幀時長
        tokens_per_frame = self.num_rvq_layers  # 每幀對應 4 個 RVQ token
        
        interleaved_segments = []
        
        for segment in alignment_segments:
            # 計算對應的音訊 token 範圍
            start_frame = int(segment.start / frame_duration)
            end_frame = int(segment.end / frame_duration)
            
            # 計算 token 索引範圍
            start_token_idx = start_frame * tokens_per_frame
            end_token_idx = end_frame * tokens_per_frame
            
            # 確保索引在有效範圍內
            start_token_idx = max(0, min(start_token_idx, len(audio_tokens)))
            end_token_idx = max(start_token_idx, min(end_token_idx, len(audio_tokens)))
            
            # 提取對應的音訊 token
            segment_audio_tokens = audio_tokens[start_token_idx:end_token_idx]
            
            # 處理文字 token
            text_tokens = self._process_text_tokens(segment.text)
            
            # 創建交錯片段
            interleaved_segment = InterleavedSegment(
                text_tokens=text_tokens,
                audio_tokens=segment_audio_tokens,
                start_time=segment.start,
                end_time=segment.end,
                alignment_confidence=segment.confidence
            )
            
            interleaved_segments.append(interleaved_segment)
        
        return interleaved_segments
    
    def _process_text_tokens(self, text: str) -> List[str]:
        """
        處理文字 token
        
        Args:
            text: 原始文字
            
        Returns:
            processed_tokens: 處理後的 token 列表
        """
        # 使用詞彙表管理器的 tokenizer
        tokens = self.vocab_manager.extended_tokenizer.tokenize(text)
        
        # 如果需要重複 token 以對齊 RVQ 層數
        if self.repeat_text_tokens:
            repeated_tokens = []
            for token in tokens:
                repeated_tokens.extend([token] * self.num_rvq_layers)
            return repeated_tokens
        
        return tokens
    
    def _assemble_full_sequence(
        self,
        interleaved_segments: List[InterleavedSegment],
        mode: str,
        role: str
    ) -> List[str]:
        """
        組裝完整的交錯序列
        
        Args:
            interleaved_segments: 交錯片段
            mode: 任務模式
            role: 角色標籤
            
        Returns:
            full_sequence: 完整的 token 序列
        """
        sequence = []
        
        # 添加角色標籤
        sequence.append(f"<{role}>")
        
        # 根據模式添加標籤
        if mode in ["TIAO", "AIAO"]:  # 音訊輸出模式
            sequence.append("<AUDIO>")
            sequence.append("<AUDIO_START>")
            
            # 交錯文字和音訊 token
            for segment in interleaved_segments:
                # 添加文字 token（重複 4 次）
                sequence.extend(segment.text_tokens)
                
                # 添加對應的音訊 token
                sequence.extend(segment.audio_tokens)
            
            sequence.append("<AUDIO_END>")
            
        elif mode in ["TITO", "AITO"]:  # 文字輸出模式
            sequence.append("<TEXT>")
            
            # 只添加文字 token
            for segment in interleaved_segments:
                # 不重複文字 token（文字輸出不需要對齊 RVQ）
                original_tokens = self.vocab_manager.extended_tokenizer.tokenize(
                    " ".join(segment.text_tokens[::self.num_rvq_layers])  # 取每 4 個中的第一個
                )
                sequence.extend(original_tokens)
        
        # 添加結束標籤
        sequence.append("<eos>")
        
        return sequence
    
    def create_training_sample(
        self,
        input_audio_path: str,
        output_audio_path: Optional[str] = None,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        mode: str = "AIAO"
    ) -> Dict[str, List[str]]:
        """
        創建訓練樣本
        
        Args:
            input_audio_path: 輸入音訊路徑
            output_audio_path: 輸出音訊路徑（可選）
            input_text: 輸入文字（可選）
            output_text: 輸出文字（可選）
            mode: 任務模式
            
        Returns:
            training_sample: 包含輸入和輸出序列的訓練樣本
        """
        input_sequence = []
        output_sequence = []
        
        # 處理輸入
        if mode in ["AITO", "AIAO"]:  # 音訊輸入
            input_sequence = self.audio_tokenizer.create_chat_format(
                text="",
                audio_path=input_audio_path,
                role="human",
                mode=mode
            )
        elif mode in ["TITO", "TIAO"]:  # 文字輸入
            if input_text is None:
                raise ValueError("文字輸入模式需要提供 input_text")
            input_sequence = self.audio_tokenizer.create_chat_format(
                text=input_text,
                role="human",
                mode=mode
            )
        
        # 處理輸出
        if mode in ["TIAO", "AIAO"]:  # 音訊輸出
            if output_audio_path is None:
                raise ValueError("音訊輸出模式需要提供 output_audio_path")
            output_sequence = self.generate_interleaved_sequence(
                audio_path=output_audio_path,
                reference_text=output_text,
                mode=mode,
                role="assistant"
            )
        elif mode in ["TITO", "AITO"]:  # 文字輸出
            if output_text is None:
                raise ValueError("文字輸出模式需要提供 output_text")
            output_sequence = self.audio_tokenizer.create_response_format(
                text=output_text,
                mode=mode
            )
        
        return {
            "input_sequence": input_sequence,
            "output_sequence": output_sequence,
            "full_sequence": input_sequence + output_sequence,
            "mode": mode
        }
    
    def batch_generate_training_data(
        self,
        data_list: List[Dict],
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """
        批量生成訓練資料
        
        Args:
            data_list: 資料列表，每個元素包含必要的路徑和文字
            output_file: 輸出文件路徑（可選）
            
        Returns:
            training_samples: 訓練樣本列表
        """
        training_samples = []
        
        for i, data in enumerate(data_list):
            try:
                sample = self.create_training_sample(**data)
                sample["sample_id"] = i
                training_samples.append(sample)
                
                if (i + 1) % 100 == 0:
                    print(f"已處理 {i + 1}/{len(data_list)} 個樣本")
                    
            except Exception as e:
                print(f"處理第 {i} 個樣本時出錯: {e}")
                continue
        
        # 保存到文件（可選）
        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_samples, f, ensure_ascii=False, indent=2)
            print(f"訓練資料已保存到 {output_file}")
        
        return training_samples
    
    def visualize_interleaving(
        self,
        interleaved_segments: List[InterleavedSegment],
        max_segments: int = 5
    ):
        """視覺化交錯結果"""
        print("交錯序列視覺化:")
        print("=" * 80)
        
        for i, segment in enumerate(interleaved_segments[:max_segments]):
            print(f"\n片段 {i+1}: [{segment.start_time:.2f}s - {segment.end_time:.2f}s]")
            print(f"置信度: {segment.alignment_confidence:.3f}")
            
            # 顯示文字 token
            print(f"文字 token: {segment.text_tokens}")
            
            # 顯示音訊 token（前 10 個）
            audio_preview = segment.audio_tokens[:10]
            if len(segment.audio_tokens) > 10:
                audio_preview.append("...")
            print(f"音訊 token: {audio_preview}")
            
            print("-" * 40)


if __name__ == "__main__":
    # 測試交錯標註產生器
    generator = InterleavingGenerator()
    
    print("交錯標註產生器初始化完成")
    
    # 模擬測試資料
    mock_alignment = [
        AlignmentSegment("你好", 0.0, 0.5, ["你", "好"], 0.9),
        AlignmentSegment("世界", 0.5, 1.0, ["世", "界"], 0.8),
        AlignmentSegment("測試", 1.0, 1.5, ["測", "試"], 0.85)
    ]
    
    # 模擬音訊 token
    mock_audio_tokens = [f"<A_L{(i%4)+1}_{(i//4)%256:03d}>" for i in range(24)]  # 6 幀 × 4 層
    
    print(f"\n模擬對齊片段: {len(mock_alignment)} 個")
    print(f"模擬音訊 token: {len(mock_audio_tokens)} 個")
    
    # 測試文字 token 處理
    test_text = "你好世界"
    processed_tokens = generator._process_text_tokens(test_text)
    print(f"\n文字處理測試:")
    print(f"  原始文字: {test_text}")
    print(f"  處理後 token: {processed_tokens}")
    
    # 測試交錯片段創建
    print(f"\n交錯片段測試:")
    for i, segment in enumerate(mock_alignment):
        text_tokens = generator._process_text_tokens(segment.text)
        print(f"  片段 {i+1}: '{segment.text}' -> {text_tokens}")
    
    # 測試完整序列組裝
    mock_interleaved = [
        InterleavedSegment(
            text_tokens=["你", "你", "你", "你"],
            audio_tokens=mock_audio_tokens[:8],
            start_time=0.0,
            end_time=0.5
        ),
        InterleavedSegment(
            text_tokens=["好", "好", "好", "好"],
            audio_tokens=mock_audio_tokens[8:16],
            start_time=0.5,
            end_time=1.0
        )
    ]
    
    full_sequence = generator._assemble_full_sequence(
        mock_interleaved, mode="AIAO", role="assistant"
    )
    
    print(f"\n完整序列測試 (AIAO 模式):")
    print(f"  序列長度: {len(full_sequence)}")
    print(f"  前 20 個 token: {full_sequence[:20]}")
    
    # 視覺化
    generator.visualize_interleaving(mock_interleaved)
