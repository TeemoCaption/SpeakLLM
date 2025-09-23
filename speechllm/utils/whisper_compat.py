"""
Whisper 相容性工具
處理不同版本 OpenAI Whisper 的 API 差異
"""

import whisper
import numpy as np
import torch
from typing import Union


def get_mel_filters(sample_rate: int, n_mels: int, n_fft: int = None) -> np.ndarray:
    """
    獲取 Mel 濾波器，相容不同版本的 Whisper

    Args:
        sample_rate: 採樣率
        n_mels: Mel 頻帶數量
        n_fft: FFT 點數（舊版本需要）

    Returns:
        mel_filters: Mel 濾波器矩陣
    """
    mel_fn = whisper.audio.mel_filters
    default_n_fft = n_fft if n_fft is not None else 400

    attempts = [
        lambda: mel_fn(device="cpu", n_mels=n_mels),
        lambda: mel_fn(n_mels, sample_rate),
        lambda: mel_fn(sample_rate, n_mels),
        lambda: mel_fn(sample_rate, default_n_fft, n_mels),
    ]

    filters = None
    last_error = None
    for attempt in attempts:
        try:
            filters = attempt()
            break
        except (TypeError, AssertionError) as error:
            last_error = error

    if filters is None:
        raise RuntimeError("Unsupported whisper.audio.mel_filters signature") from last_error

    if isinstance(filters, torch.Tensor):
        filters = filters.detach().cpu().numpy()

    return np.asarray(filters, dtype=np.float32)




def get_sample_rate() -> int:
    """
    獲取 Whisper 的採樣率
    
    Returns:
        sample_rate: 採樣率
    """
    try:
        return whisper.audio.SAMPLE_RATE
    except AttributeError:
        # 如果沒有 SAMPLE_RATE 常數，使用預設值
        return 16000


def load_audio(file_path: str, sr: int = None) -> np.ndarray:
    """
    載入音訊檔案，相容不同版本的 Whisper
    
    Args:
        file_path: 音訊檔案路徑
        sr: 目標採樣率
        
    Returns:
        audio: 音訊陣列
    """
    if sr is None:
        sr = get_sample_rate()
    
    try:
        return whisper.load_audio(file_path, sr=sr)
    except TypeError:
        # 如果不支援 sr 參數，使用預設載入
        return whisper.load_audio(file_path)


def pad_or_trim(array: np.ndarray, length: int, axis: int = -1) -> np.ndarray:
    """
    填充或裁剪陣列到指定長度
    
    Args:
        array: 輸入陣列
        length: 目標長度
        axis: 操作軸
        
    Returns:
        padded_array: 處理後的陣列
    """
    try:
        return whisper.pad_or_trim(array, length, axis=axis)
    except (AttributeError, TypeError):
        # 如果沒有 pad_or_trim 函數，手動實現
        if array.shape[axis] > length:
            # 裁剪
            slices = [slice(None)] * array.ndim
            slices[axis] = slice(0, length)
            return array[tuple(slices)]
        elif array.shape[axis] < length:
            # 填充
            pad_width = [(0, 0)] * array.ndim
            pad_width[axis] = (0, length - array.shape[axis])
            return np.pad(array, pad_width, mode='constant', constant_values=0)
        else:
            return array
