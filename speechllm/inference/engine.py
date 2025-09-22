"""
SpeechLLM 推理引擎
支援串流推理和全雙工對話
"""

import torch
import torch.nn.functional as F
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Union, Callable, Iterator
from dataclasses import dataclass
import librosa
import soundfile as sf
from pathlib import Path

from ..models.speechllm import SpeechLLM, SpeechLLMConfig
from ..codecs.audio_tokenizer import AudioTokenizer
from ..codecs.rvq_codec import RVQCodec


@dataclass
class InferenceConfig:
    """推理配置"""
    # 生成參數
    max_length: int = 512
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1
    
    # 音訊參數
    sample_rate: int = 16000
    chunk_duration: float = 0.32  # 320ms 分塊
    max_audio_length: float = 10.0  # 最大音訊長度（秒）
    
    # 串流參數
    stream_chunk_size: int = 1024
    vad_threshold: float = 0.5
    silence_threshold: float = 2.0  # 靜音閾值（秒）
    
    # 全雙工參數
    enable_duplex: bool = False
    interrupt_threshold: float = 0.7
    response_delay: float = 0.1  # 回應延遲（秒）
    
    # 設備配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True


class SpeechLLMInferenceEngine:
    """
    SpeechLLM 推理引擎
    
    功能：
    1. 單輪對話推理
    2. 串流音訊處理
    3. 全雙工對話
    4. 狀態管理
    """
    
    def __init__(
        self,
        model: SpeechLLM,
        config: Optional[InferenceConfig] = None
    ):
        self.model = model
        self.config = config or InferenceConfig()
        
        # 設置設備
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.model.eval()
        
        # 混合精度
        if self.config.fp16:
            self.model.half()
        
        # 音訊處理組件
        self.audio_tokenizer = model.audio_tokenizer
        self.vocab_manager = model.vocab_manager
        
        # 串流狀態
        self.is_streaming = False
        self.audio_buffer = []
        self.conversation_history = []
        
        # 全雙工狀態
        self.duplex_enabled = self.config.enable_duplex
        self.user_speaking = False
        self.model_speaking = False
        self.interrupt_requested = False
        
        # 線程和隊列
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.stop_event = threading.Event()
        
    def generate_response(
        self,
        input_text: Optional[str] = None,
        input_audio: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        mode: str = "auto",
        **generation_kwargs
    ) -> Dict[str, Union[str, np.ndarray]]:
        """
        生成回應
        
        Args:
            input_text: 輸入文字
            input_audio: 輸入音訊（路徑、numpy 陣列或 tensor）
            mode: 模式（auto, TITO, AITO, TIAO, AIAO）
            generation_kwargs: 生成參數
            
        Returns:
            response: 包含文字和/或音訊回應的字典
        """
        # 自動判斷模式
        if mode == "auto":
            mode = self._determine_mode(input_text, input_audio)
        
        # 準備輸入
        input_tokens = self._prepare_input(input_text, input_audio, mode)
        
        # 生成回應
        with torch.no_grad():
            generated_outputs = self.model.generate_speech(
                input_ids=input_tokens,
                audio=self._process_audio_input(input_audio) if input_audio else None,
                max_length=generation_kwargs.get("max_length", self.config.max_length),
                temperature=generation_kwargs.get("temperature", self.config.temperature),
                top_k=generation_kwargs.get("top_k", self.config.top_k),
                top_p=generation_kwargs.get("top_p", self.config.top_p),
                do_sample=generation_kwargs.get("do_sample", self.config.do_sample)
            )
        
        # 解碼回應
        response = self._decode_response(generated_outputs, mode)
        
        # 更新對話歷史
        self._update_conversation_history(input_text, input_audio, response, mode)
        
        return response
    
    def start_streaming(
        self,
        audio_callback: Optional[Callable] = None,
        text_callback: Optional[Callable] = None
    ):
        """
        開始串流推理
        
        Args:
            audio_callback: 音訊輸入回調函數
            text_callback: 文字輸出回調函數
        """
        if self.is_streaming:
            print("串流已在運行中")
            return
        
        self.is_streaming = True
        self.stop_event.clear()
        
        # 啟動處理線程
        self.audio_thread = threading.Thread(
            target=self._audio_processing_thread,
            args=(audio_callback,)
        )
        self.response_thread = threading.Thread(
            target=self._response_generation_thread,
            args=(text_callback,)
        )
        
        self.audio_thread.start()
        self.response_thread.start()
        
        print("串流推理已啟動")
    
    def stop_streaming(self):
        """停止串流推理"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        self.stop_event.set()
        
        # 等待線程結束
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        if hasattr(self, 'response_thread'):
            self.response_thread.join()
        
        print("串流推理已停止")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """
        添加音訊塊到處理隊列
        
        Args:
            audio_chunk: 音訊塊
        """
        if self.is_streaming:
            self.audio_queue.put(audio_chunk)
    
    def _determine_mode(
        self,
        input_text: Optional[str],
        input_audio: Optional[Union[str, np.ndarray, torch.Tensor]]
    ) -> str:
        """自動判斷對話模式"""
        has_text = input_text is not None and len(input_text.strip()) > 0
        has_audio = input_audio is not None
        
        if has_text and has_audio:
            return "AIAO"  # 音訊輸入，音訊輸出
        elif has_audio:
            return "AITO"  # 音訊輸入，文字輸出
        elif has_text:
            return "TIAO"  # 文字輸入，音訊輸出
        else:
            return "TITO"  # 文字輸入，文字輸出
    
    def _prepare_input(
        self,
        input_text: Optional[str],
        input_audio: Optional[Union[str, np.ndarray, torch.Tensor]],
        mode: str
    ) -> torch.Tensor:
        """準備模型輸入"""
        if mode in ["TITO", "TIAO"]:
            # 文字輸入
            tokens = self.audio_tokenizer.create_chat_format(
                text=input_text or "",
                role="human",
                mode=mode
            )
        else:
            # 音訊輸入
            tokens = self.audio_tokenizer.create_chat_format(
                text="",
                audio_path=input_audio if isinstance(input_audio, str) else None,
                role="human",
                mode=mode
            )
        
        # 轉換為 tensor
        token_ids = self.vocab_manager.get_token_ids(tokens)
        return torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def _process_audio_input(
        self,
        input_audio: Union[str, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """處理音訊輸入"""
        if isinstance(input_audio, str):
            # 從文件載入
            audio, sr = librosa.load(input_audio, sr=self.config.sample_rate, mono=True)
            audio = torch.from_numpy(audio).float()
        elif isinstance(input_audio, np.ndarray):
            # numpy 陣列
            audio = torch.from_numpy(input_audio).float()
        else:
            # 已經是 tensor
            audio = input_audio.float()
        
        # 確保格式正確
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # [time] -> [1, time]
        
        # 截斷或填充
        max_samples = int(self.config.max_audio_length * self.config.sample_rate)
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        
        return audio.to(self.device)
    
    def _decode_response(
        self,
        generated_outputs: Dict[str, torch.Tensor],
        mode: str
    ) -> Dict[str, Union[str, np.ndarray]]:
        """解碼生成的回應"""
        response = {}
        
        # 解碼文字
        if "generated_ids" in generated_outputs:
            generated_ids = generated_outputs["generated_ids"][0]  # 取第一個序列
            
            # 移除特殊 token 並解碼
            text_tokens = []
            for token_id in generated_ids:
                token = self.vocab_manager.extended_tokenizer.convert_ids_to_tokens(token_id.item())
                if not token.startswith("<") or not token.endswith(">"):
                    text_tokens.append(token)
            
            response_text = self.vocab_manager.extended_tokenizer.convert_tokens_to_string(text_tokens)
            response["text"] = response_text.strip()
        
        # 解碼音訊
        if "rvq_tokens" in generated_outputs and mode in ["TIAO", "AIAO"]:
            rvq_tokens = generated_outputs["rvq_tokens"]
            
            # 轉換 RVQ token 為音訊
            try:
                audio = self.audio_tokenizer.decode_audio(rvq_tokens)
                if audio.dim() == 3:
                    audio = audio.squeeze(0).squeeze(0)  # [1, 1, time] -> [time]
                
                response["audio"] = audio.cpu().numpy()
                response["sample_rate"] = self.config.sample_rate
            except Exception as e:
                print(f"音訊解碼失敗: {e}")
                response["audio"] = None
        
        return response
    
    def _update_conversation_history(
        self,
        input_text: Optional[str],
        input_audio: Optional[Union[str, np.ndarray, torch.Tensor]],
        response: Dict[str, Union[str, np.ndarray]],
        mode: str
    ):
        """更新對話歷史"""
        turn = {
            "timestamp": time.time(),
            "mode": mode,
            "input": {
                "text": input_text,
                "has_audio": input_audio is not None
            },
            "response": {
                "text": response.get("text"),
                "has_audio": response.get("audio") is not None
            }
        }
        
        self.conversation_history.append(turn)
        
        # 限制歷史長度
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]
    
    def _audio_processing_thread(self, audio_callback: Optional[Callable]):
        """音訊處理線程"""
        audio_buffer = []
        
        while not self.stop_event.is_set():
            try:
                # 獲取音訊塊
                if audio_callback:
                    audio_chunk = audio_callback()
                else:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                
                if audio_chunk is None:
                    continue
                
                # 添加到緩衝區
                audio_buffer.append(audio_chunk)
                
                # 檢查是否有足夠的音訊進行處理
                total_duration = len(audio_buffer) * self.config.chunk_duration
                if total_duration >= 1.0:  # 1 秒的音訊
                    # 合併音訊塊
                    combined_audio = np.concatenate(audio_buffer)
                    
                    # VAD 檢測
                    if self._voice_activity_detection(combined_audio):
                        # 處理音訊
                        self._process_audio_segment(combined_audio)
                    
                    # 清空緩衝區
                    audio_buffer = []
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"音訊處理錯誤: {e}")
    
    def _response_generation_thread(self, text_callback: Optional[Callable]):
        """回應生成線程"""
        while not self.stop_event.is_set():
            try:
                # 從隊列獲取處理請求
                request = self.response_queue.get(timeout=0.1)
                
                # 生成回應
                response = self.generate_response(**request)
                
                # 回調處理
                if text_callback and "text" in response:
                    text_callback(response["text"])
                
                # 播放音訊（如果有）
                if "audio" in response and response["audio"] is not None:
                    self._play_audio(response["audio"], response.get("sample_rate", self.config.sample_rate))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"回應生成錯誤: {e}")
    
    def _voice_activity_detection(self, audio: np.ndarray) -> bool:
        """語音活動檢測（針對中文語音優化）"""
        # 能量基礎 VAD
        energy = np.mean(audio ** 2)
        
        # 中文語音特性：考慮聲調變化
        # 計算短時能量變化
        frame_length = int(0.025 * self.config.sample_rate)  # 25ms 幀
        hop_length = int(0.01 * self.config.sample_rate)  # 10ms 跳躍
        
        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            frame_energy = np.mean(frame ** 2)
            frames.append(frame_energy)
        
        if len(frames) > 0:
            energy_variance = np.var(frames)  # 能量變化方差
            # 中文語音的聲調變化會導致較大的能量變化
            return energy > self.config.vad_threshold or energy_variance > 0.001
        
        return energy > self.config.vad_threshold
    
    def _process_audio_segment(self, audio: np.ndarray):
        """處理音訊片段"""
        # 添加到回應隊列
        request = {
            "input_audio": audio,
            "mode": "AITO"  # 預設音訊輸入文字輸出
        }
        self.response_queue.put(request)
    
    def _play_audio(self, audio: np.ndarray, sample_rate: int):
        """播放音訊"""
        try:
            # 這裡可以整合音訊播放庫
            # 例如使用 sounddevice 或 pygame
            print(f"播放音訊: {len(audio)} 樣本, {sample_rate} Hz")
        except Exception as e:
            print(f"音訊播放錯誤: {e}")
    
    def save_audio(self, audio: np.ndarray, filepath: str, sample_rate: Optional[int] = None):
        """保存音訊文件"""
        if sample_rate is None:
            sample_rate = self.config.sample_rate
        
        sf.write(filepath, audio, sample_rate)
        print(f"音訊已保存到: {filepath}")
    
    def get_conversation_history(self) -> List[Dict]:
        """獲取對話歷史"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """清空對話歷史"""
        self.conversation_history.clear()
        print("對話歷史已清空")
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """獲取模型資訊"""
        return {
            "model_type": "SpeechLLM",
            "device": str(self.device),
            "fp16": self.config.fp16,
            "vocab_size": len(self.vocab_manager.extended_tokenizer),
            "conversation_turns": len(self.conversation_history)
        }


if __name__ == "__main__":
    # 測試推理引擎
    print("測試 SpeechLLM 推理引擎")
    
    # 創建配置
    model_config = SpeechLLMConfig()
    inference_config = InferenceConfig(
        max_length=128,
        temperature=0.8,
        device="cpu"  # 測試時使用 CPU
    )
    
    print("配置創建完成")
    
    # 注意：實際使用時需要載入訓練好的模型
    print("推理引擎測試完成（需要訓練好的模型才能運行完整測試）")
    
    # 測試配置
    print(f"推理配置:")
    print(f"  最大長度: {inference_config.max_length}")
    print(f"  溫度: {inference_config.temperature}")
    print(f"  Top-k: {inference_config.top_k}")
    print(f"  Top-p: {inference_config.top_p}")
    print(f"  設備: {inference_config.device}")
    print(f"  串流塊大小: {inference_config.stream_chunk_size}")
    print(f"  全雙工: {inference_config.enable_duplex}")
