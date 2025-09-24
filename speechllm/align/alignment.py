"""
語音-文字對齊模組
使用 Whisper 進行 ASR 對齊，獲取詞級時間戳
支援繁體中文拼音音節級對齊
"""

import logging
import whisper
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import librosa
from dtw import dtw
import jieba
import re

from ..utils.whisper_compat import load_audio, get_sample_rate
from ..utils.device_utils import resolve_device_with_info
from pypinyin import pinyin, lazy_pinyin, Style
import opencc
import warnings


@dataclass
class AlignmentSegment:
    """對齊片段"""

    text: str  # 文字內容
    start: float  # 開始時間（秒）
    end: float  # 結束時間（秒）
    tokens: List[str]  # token 列表
    confidence: float = 1.0  # 置信度


@dataclass
class ChineseAlignmentSegment:
    """繁體中文對齊片段"""

    text: str  # 原始文字
    pinyin: str  # 拼音（含聲調）
    start: float  # 開始時間（秒）
    end: float  # 結束時間（秒）
    tokens: List[str]  # token 列表
    confidence: float = 1.0  # 置信度
    unit_type: str = "syllable"  # 對齊單位類型：syllable, character, word




def _linear_time_spans(count: int, total_duration: float) -> List[Tuple[float, float]]:
    """Create evenly spaced time spans covering the full duration."""
    if count <= 0:
        return []
    if total_duration <= 0:
        return [(0.0, 0.0) for _ in range(count)]
    unit = total_duration / count
    spans = []
    for idx in range(count):
        start = unit * idx
        end = min(total_duration, unit * (idx + 1))
        spans.append((start, end))
    return spans


def _compute_dtw_alignment(reference: List[str], observed: List[str]) -> Dict[int, List[int]]:
    """Align reference and observed sequences using DTW and return index mapping."""
    mapping: Dict[int, List[int]] = {i: [] for i in range(len(reference))}
    if not reference or not observed:
        return mapping

    vocabulary = {token: idx for idx, token in enumerate(sorted(set(reference + observed)))}
    ref_array = np.array([vocabulary[token] for token in reference], dtype=np.int32).reshape(-1, 1)
    obs_array = np.array([vocabulary[token] for token in observed], dtype=np.int32).reshape(-1, 1)

    def _dist(x: np.ndarray, y: np.ndarray) -> float:
        return 0.0 if int(x[0]) == int(y[0]) else 1.0

    alignment = dtw(ref_array, obs_array, keep_internals=True, dist=_dist)
    for ref_idx, obs_idx in zip(alignment.index1, alignment.index2):
        r = int(ref_idx) - 1
        o = int(obs_idx) - 1
        if 0 <= r < len(reference) and 0 <= o < len(observed):
            mapping[r].append(o)
    return mapping


def _extract_recognized_words(result: Dict, audio_duration: float) -> List[Dict[str, float]]:
    """Extract recognized words with timing information from Whisper output."""
    words: List[Dict[str, float]] = []

    for word_info in result.get('words', []):
        text = word_info.get('word', '').strip()
        if not text:
            continue
        start = max(0.0, float(word_info.get('start', 0.0)))
        end = min(audio_duration, float(word_info.get('end', start)))
        confidence = float(word_info.get('probability', 1.0))
        words.append({'text': text, 'start': start, 'end': end, 'confidence': confidence})

    if words:
        return words

    for segment in result.get('segments', []):
        text = segment.get('text', '').strip()
        if not text:
            continue
        seg_start = max(0.0, float(segment.get('start', 0.0)))
        seg_end = min(audio_duration, float(segment.get('end', seg_start)))
        tokens = re.findall(r'\S+', text)
        if not tokens:
            continue
        duration = max(seg_end - seg_start, 1e-4)
        per_word = duration / len(tokens)
        confidence = float(segment.get('avg_logprob', 1.0))
        for idx, token in enumerate(tokens):
            start = seg_start + idx * per_word
            end = min(seg_end, start + per_word)
            words.append({'text': token, 'start': start, 'end': end, 'confidence': confidence})

    return words


class WhisperAligner:
    """
    基於 Whisper 的語音-文字對齊器
    獲取詞級時間戳用於交錯標註
    """


    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        language: str = "zh",
        task: str = "transcribe",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        resolved_device, device_info = resolve_device_with_info(device, logger=self.logger)
        self.device = resolved_device
        self.language = language
        self.task = task

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Whisper aligner device: {device_info}")

        # 載入 Whisper 模型
        self.model = whisper.load_model(model_name, device=str(self.device))

        # 分詞器
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            self.model.is_multilingual,
            num_languages=self.model.num_languages,
            language=language,
            task=task,
        )

        def align_audio_text(
            self,
            audio_path: str,
            reference_text: Optional[str] = None,
            word_level: bool = True,
        ) -> List[AlignmentSegment]:
            """
            對齊音訊和文字

            Args:
                audio_path: 音訊文件路徑
                reference_text: 參考文字（可選）
                word_level: 是否進行詞級對齊

            Returns:
                segments: 對齊片段列表
            """
            # 載入音訊
            audio = load_audio(audio_path)

            # 轉錄並獲取時間戳
            result = self.model.transcribe(
                audio,
                language=self.language,
                task=self.task,
                word_timestamps=word_level,
                verbose=False,
            )

            segments = []

            if word_level and "words" in result:
                # 詞級對齊
                for word_info in result["words"]:
                    segment = AlignmentSegment(
                        text=word_info["word"].strip(),
                        start=word_info["start"],
                        end=word_info["end"],
                        tokens=self.tokenizer.encode(word_info["word"].strip()),
                        confidence=word_info.get("probability", 1.0),
                    )
                    segments.append(segment)
            else:
                # 句子級對齊
                for segment_info in result["segments"]:
                    segment = AlignmentSegment(
                        text=segment_info["text"].strip(),
                        start=segment_info["start"],
                        end=segment_info["end"],
                        tokens=self.tokenizer.encode(segment_info["text"].strip()),
                        confidence=segment_info.get("avg_logprob", 1.0),
                    )
                    segments.append(segment)

            return segments


    def force_align(
        self,
        audio_path: str,
        reference_text: str,
        segment_duration: float = 0.02,  # 20ms 對應 Whisper 的時間解析度
    ) -> List[AlignmentSegment]:
        """使用 DTW 強制對齊已知文字和音訊"""
        audio = load_audio(audio_path)
        sample_rate = get_sample_rate()
        audio_duration = len(audio) / sample_rate if sample_rate else 0.0

        result = self.model.transcribe(
            audio,
            language=self.language,
            task=self.task,
            word_timestamps=True,
            verbose=False,
        )

        recognized_words = _extract_recognized_words(result, audio_duration)
        reference_words = self._tokens_to_words(self.tokenizer.encode(reference_text))

        if not reference_words:
            return []

        mapping = _compute_dtw_alignment(
            [word.lower() for word in reference_words],
            [word["text"].lower() for word in recognized_words],
        )

        spans = _linear_time_spans(len(reference_words), audio_duration)
        segments: List[AlignmentSegment] = []

        for idx, word in enumerate(reference_words):
            matched_indices = sorted(set(mapping.get(idx, [])))
            if matched_indices:
                start_time = min(recognized_words[i]["start"] for i in matched_indices)
                end_time = max(recognized_words[i]["end"] for i in matched_indices)
                confidence = float(
                    np.mean([recognized_words[i]["confidence"] for i in matched_indices])
                )
            else:
                start_time, end_time = spans[idx]
                confidence = 0.3

            segments.append(
                AlignmentSegment(
                    text=word,
                    start=max(0.0, start_time),
                    end=min(audio_duration, end_time),
                    tokens=self.tokenizer.encode(word),
                    confidence=confidence,
                )
            )

        return segments

        def _tokens_to_words(self, tokens: List[int]) -> List[str]:
            """將 token 轉換為詞列表"""
            text = self.tokenizer.decode(tokens)
            # 簡單的分詞（可以使用更複雜的分詞器）
            words = re.findall(r"\S+", text)
            return words

        def get_time_to_token_mapping(
            self,
            segments: List[AlignmentSegment],
            hop_length: int = 320,  # 對應 20ms at 16kHz
            sample_rate: int = 16000,
        ) -> Dict[int, List[str]]:
            """
            獲取時間到 token 的映射

            Args:
                segments: 對齊片段
                hop_length: 跳躍長度
                sample_rate: 採樣率

            Returns:
                mapping: 時間幀到 token 的映射
            """
            # 計算總幀數
            if not segments:
                return {}

            max_time = max(seg.end for seg in segments)
            total_frames = int(max_time * sample_rate / hop_length) + 1

            # 初始化映射
            mapping = {i: [] for i in range(total_frames)}

            # 填充映射
            for segment in segments:
                start_frame = int(segment.start * sample_rate / hop_length)
                end_frame = int(segment.end * sample_rate / hop_length)

                # 將 token 分配到對應的時間幀
                for frame in range(start_frame, min(end_frame + 1, total_frames)):
                    mapping[frame].extend(segment.tokens)

            return mapping


class ChineseTextProcessor:
    """
    中文文字處理器
    負責繁簡轉換、分詞、拼音轉換等
    """

    def __init__(
        self,
        convert_traditional: bool = True,  # 是否進行繁簡轉換
        use_tone: bool = True,  # 是否使用聲調
        segment_method: str = "jieba",  # 分詞方法：jieba, pkuseg
    ):
        self.convert_traditional = convert_traditional
        self.use_tone = use_tone
        self.segment_method = segment_method

        # 初始化繁簡轉換器
        if convert_traditional:
            self.converter = opencc.OpenCC("s2t")  # 簡體轉繁體
        else:
            self.converter = None

        # 初始化分詞器
        if segment_method == "jieba":
            jieba.initialize()
        elif segment_method == "pkuseg":
            try:
                import pkuseg

                self.pkuseg_model = pkuseg.pkuseg()
            except ImportError:
                warnings.warn("pkuseg 未安裝，回退到 jieba")
                self.segment_method = "jieba"

    def normalize_text(self, text: str) -> str:
        """中文文字正規化"""
        # 繁簡轉換處理
        if self.converter:
            text = self.converter.convert(text)

        # 移除多餘空白和標點符號的處理
        text = re.sub(r"\s+", "", text)  # 移除所有空白
        text = re.sub(r'[，。！？；：""' "（）【】《》]", "", text)  # 移除標點符號

        return text

    def text_to_pinyin(self, text: str, unit: str = "syllable") -> List[str]:
        """
        將中文文字轉換為拼音

        Args:
            text: 中文文字
            unit: 轉換單位 (syllable, character, word)

        Returns:
            拼音列表
        """
        normalized_text = self.normalize_text(text)

        if unit == "syllable":
            # 音節級：每個字轉為一個拼音音節
            if self.use_tone:
                pinyins = lazy_pinyin(normalized_text, style=Style.TONE3)
            else:
                pinyins = lazy_pinyin(normalized_text, style=Style.NORMAL)
            return pinyins

        elif unit == "character":
            # 字級：直接返回字符列表
            return list(normalized_text)

        elif unit == "word":
            # 詞級：先分詞再轉拼音
            if self.segment_method == "jieba":
                words = list(jieba.cut(normalized_text))
            else:  # pkuseg
                words = self.pkuseg_model.cut(normalized_text)

            if self.use_tone:
                word_pinyins = []
                for word in words:
                    word_pinyin = "".join(lazy_pinyin(word, style=Style.TONE3))
                    word_pinyins.append(word_pinyin)
                return word_pinyins
            else:
                word_pinyins = []
                for word in words:
                    word_pinyin = "".join(lazy_pinyin(word, style=Style.NORMAL))
                    word_pinyins.append(word_pinyin)
                return word_pinyins

        else:
            raise ValueError(f"不支援的單位類型: {unit}")

    def get_character_pinyin_mapping(self, text: str) -> List[Tuple[str, str]]:
        """
        獲取字符和拼音的對應關係

        Returns:
            [(字符, 拼音), ...] 的列表
        """
        normalized_text = self.normalize_text(text)
        characters = list(normalized_text)

        if self.use_tone:
            pinyins = lazy_pinyin(normalized_text, style=Style.TONE3)
        else:
            pinyins = lazy_pinyin(normalized_text, style=Style.NORMAL)

        return list(zip(characters, pinyins))


class ChineseWhisperAligner:
    """
    基於 Whisper 的中文語音-文字對齊器
    支援音節級、字級和詞級對齊，支援多種中文口音
    """

    def __init__(
        self,
        model_name: str = "medium",
        device: Optional[str] = None,
        language: str = "zh",
        alignment_unit: str = "syllable",  # syllable, character, word
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        resolved_device, device_info = resolve_device_with_info(device, logger=self.logger)
        self.device = resolved_device
        self.language = language
        self.alignment_unit = alignment_unit

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Chinese Whisper aligner device: {device_info}")

        # 載入 Whisper 模型
        self.model = whisper.load_model(model_name, device=str(self.device))

        # 初始化中文文字處理器
        self.text_processor = ChineseTextProcessor()

        # 分詞器
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            self.model.is_multilingual,
            num_languages=self.model.num_languages,
            language=language,
            task="transcribe",
        )


        def align_audio_text(
            self,
            audio_path: str,
            reference_text: Optional[str] = None,
            alignment_unit: Optional[str] = None,
        ) -> List[ChineseAlignmentSegment]:
            """
            對齊音訊和中文文字

            Args:
                audio_path: 音訊文件路徑
                reference_text: 參考文字（可選）
                alignment_unit: 對齊單位（可選，覆蓋預設值）

            Returns:
                中文對齊片段列表
            """
            if alignment_unit is None:
                alignment_unit = self.alignment_unit

            # 載入音訊
            audio = load_audio(audio_path)

            if reference_text is None:
                # 自動轉錄
                return self._auto_transcribe_align(audio, alignment_unit)
            else:
                # 強制對齊
                return self._force_align(audio, reference_text, alignment_unit)

        def _auto_transcribe_align(
            self, audio: np.ndarray, alignment_unit: str
        ) -> List[ChineseAlignmentSegment]:
            """自動轉錄並對齊"""
            # 使用 Whisper 轉錄並獲取詞級時間戳
            result = self.model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                word_timestamps=True,
                verbose=False,
            )

            segments = []

            if "words" in result:
                # 處理詞級時間戳
                for word_info in result["words"]:
                    word_text = word_info["word"].strip()

                    # 根據對齊單位處理
                    if alignment_unit == "syllable":
                        # 轉換為拼音音節
                        pinyins = self.text_processor.text_to_pinyin(word_text, "syllable")

                        # 將時間平均分配給每個音節
                        duration = word_info["end"] - word_info["start"]
                        syllable_duration = duration / len(pinyins)

                        for i, pinyin in enumerate(pinyins):
                            start_time = word_info["start"] + i * syllable_duration
                            end_time = start_time + syllable_duration

                            segment = ChineseAlignmentSegment(
                                text=word_text[i] if i < len(word_text) else "",
                                pinyin=pinyin,
                                start=start_time,
                                end=end_time,
                                tokens=[pinyin],
                                confidence=word_info.get("probability", 1.0),
                                unit_type="syllable",
                            )
                            segments.append(segment)

                    elif alignment_unit == "character":
                        # 字級對齊
                        characters = list(word_text)
                        duration = word_info["end"] - word_info["start"]
                        char_duration = duration / len(characters)

                        for i, char in enumerate(characters):
                            start_time = word_info["start"] + i * char_duration
                            end_time = start_time + char_duration

                            # 獲取對應拼音
                            char_pinyin = self.text_processor.text_to_pinyin(
                                char, "syllable"
                            )[0]

                            segment = ChineseAlignmentSegment(
                                text=char,
                                pinyin=char_pinyin,
                                start=start_time,
                                end=end_time,
                                tokens=[char],
                                confidence=word_info.get("probability", 1.0),
                                unit_type="character",
                            )
                            segments.append(segment)

                    else:  # word
                        # 詞級對齊
                        word_pinyin = "".join(
                            self.text_processor.text_to_pinyin(word_text, "syllable")
                        )

                        segment = ChineseAlignmentSegment(
                            text=word_text,
                            pinyin=word_pinyin,
                            start=word_info["start"],
                            end=word_info["end"],
                            tokens=[word_text],
                            confidence=word_info.get("probability", 1.0),
                            unit_type="word",
                        )
                        segments.append(segment)

            return segments

    def _force_align(
        self, audio: np.ndarray, reference_text: str, alignment_unit: str
    ) -> List[ChineseAlignmentSegment]:
        """使用 DTW 強制對齊已知中文文字和音訊"""
        audio_duration = len(audio) / get_sample_rate()

        result = self.model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            word_timestamps=True,
            verbose=False,
        )

        reference_texts, reference_keys, reference_pinyins = self._prepare_reference_units(
            reference_text, alignment_unit
        )
        if not reference_texts:
            return []

        recognized_words = _extract_recognized_words(result, audio_duration)
        observed_units = self._prepare_observed_units(recognized_words, alignment_unit)

        mapping = _compute_dtw_alignment(
            reference_keys,
            [unit["key"] for unit in observed_units],
        )
        spans = _linear_time_spans(len(reference_texts), audio_duration)

        segments: List[ChineseAlignmentSegment] = []
        for idx, unit_text in enumerate(reference_texts):
            matched_indices = sorted(set(mapping.get(idx, [])))
            if matched_indices:
                start_time = min(observed_units[i]["start"] for i in matched_indices)
                end_time = max(observed_units[i]["end"] for i in matched_indices)
                confidence = float(
                    np.mean([observed_units[i]["confidence"] for i in matched_indices])
                )
            else:
                start_time, end_time = spans[idx]
                confidence = 0.3

            if alignment_unit == "syllable":
                tokens = [reference_pinyins[idx]]
            elif alignment_unit == "character":
                tokens = [unit_text]
            else:
                tokens = [reference_pinyins[idx]]

            segments.append(
                ChineseAlignmentSegment(
                    text=unit_text,
                    pinyin=reference_pinyins[idx],
                    start=max(0.0, start_time),
                    end=min(audio_duration, end_time),
                    tokens=tokens,
                    confidence=confidence,
                    unit_type=alignment_unit,
                )
            )

        return segments

    def _prepare_reference_units(
        self, reference_text: str, alignment_unit: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """根據對齊單位生成參考單位及其拼音/鍵值"""
        normalized_text = self.text_processor.normalize_text(reference_text)
        if not normalized_text:
            return [], [], []

        if alignment_unit == "syllable":
            char_pairs = self.text_processor.get_character_pinyin_mapping(reference_text)
            texts = [char for char, _ in char_pairs]
            pinyins = [pinyin for _, pinyin in char_pairs]
            return texts, pinyins, pinyins

        if alignment_unit == "character":
            texts = list(normalized_text)
            pinyins = [
                self.text_processor.text_to_pinyin(char, "syllable")[0]
                for char in texts
            ] if texts else []
            return texts, texts, pinyins

        if self.text_processor.segment_method == "jieba":
            words = list(jieba.cut(normalized_text))
        else:
            words = self.text_processor.pkuseg_model.cut(normalized_text)

        if self.text_processor.use_tone:
            pinyins = ["".join(lazy_pinyin(word, style=Style.TONE3)) for word in words]
        else:
            pinyins = ["".join(lazy_pinyin(word, style=Style.NORMAL)) for word in words]

        return list(words), pinyins, pinyins

    def _prepare_observed_units(
        self, recognized_words: List[Dict[str, float]], alignment_unit: str
    ) -> List[Dict[str, float]]:
        """將 Whisper 轉錄結果展開為指定單位"""
        units: List[Dict[str, float]] = []

        for word in recognized_words:
            raw_text = word["text"]
            normalized = self.text_processor.normalize_text(raw_text)

            if alignment_unit == "syllable":
                char_pairs = self.text_processor.get_character_pinyin_mapping(raw_text)
                if not char_pairs:
                    continue
                duration = max(word["end"] - word["start"], 1e-4)
                per_unit = duration / len(char_pairs)
                for idx, (char, pinyin_value) in enumerate(char_pairs):
                    start = word["start"] + idx * per_unit
                    end = min(word["end"], start + per_unit)
                    units.append(
                        {
                            "key": pinyin_value,
                            "text": char,
                            "pinyin": pinyin_value,
                            "start": start,
                            "end": end,
                            "confidence": word["confidence"],
                        }
                    )
            elif alignment_unit == "character":
                if not normalized:
                    continue
                duration = max(word["end"] - word["start"], 1e-4)
                per_unit = duration / len(normalized)
                for idx, char in enumerate(normalized):
                    pinyin_value = self.text_processor.text_to_pinyin(char, "syllable")[0]
                    start = word["start"] + idx * per_unit
                    end = min(word["end"], start + per_unit)
                    units.append(
                        {
                            "key": char,
                            "text": char,
                            "pinyin": pinyin_value,
                            "start": start,
                            "end": end,
                            "confidence": word["confidence"],
                        }
                    )
            else:
                if not normalized:
                    continue
                if self.text_processor.use_tone:
                    word_pinyin = "".join(lazy_pinyin(normalized, style=Style.TONE3))
                else:
                    word_pinyin = "".join(lazy_pinyin(normalized, style=Style.NORMAL))
                units.append(
                    {
                        "key": word_pinyin,
                        "text": normalized,
                        "pinyin": word_pinyin,
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word["confidence"],
                    }
                )

        return units



class CTCAligner:
    """
    基於 CTC 的對齊器（可選實現）
    提供更精確的幀級對齊
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        # 這裡可以載入 CTC 對齊模型
        # 例如 wav2vec2 + CTC 頭
        pass

    def align(self, audio: torch.Tensor, text: str) -> List[AlignmentSegment]:
        """CTC 對齊實現"""
        # TODO: 實現 CTC 對齊
        raise NotImplementedError("CTC 對齊尚未實現")


class AlignmentPostProcessor:
    """對齊後處理器"""

    @staticmethod
    def smooth_boundaries(
        segments: List[AlignmentSegment],
        min_duration: float = 0.1,
        max_gap: float = 0.05,
    ) -> List[AlignmentSegment]:
        """
        平滑邊界

        Args:
            segments: 原始片段
            min_duration: 最小持續時間
            max_gap: 最大間隙

        Returns:
            smoothed_segments: 平滑後的片段
        """
        if not segments:
            return segments

        smoothed = []

        for i, segment in enumerate(segments):
            # 確保最小持續時間
            duration = segment.end - segment.start
            if duration < min_duration:
                # 擴展到最小持續時間
                center = (segment.start + segment.end) / 2
                new_start = center - min_duration / 2
                new_end = center + min_duration / 2

                segment = AlignmentSegment(
                    text=segment.text,
                    start=max(0, new_start),
                    end=new_end,
                    tokens=segment.tokens,
                    confidence=segment.confidence,
                )

            # 填補小間隙
            if i > 0 and segment.start - smoothed[-1].end <= max_gap:
                # 擴展前一個片段
                smoothed[-1] = AlignmentSegment(
                    text=smoothed[-1].text,
                    start=smoothed[-1].start,
                    end=segment.start,
                    tokens=smoothed[-1].tokens,
                    confidence=smoothed[-1].confidence,
                )

            smoothed.append(segment)

        return smoothed

    @staticmethod
    def filter_by_confidence(
        segments: List[AlignmentSegment], min_confidence: float = 0.5
    ) -> List[AlignmentSegment]:
        """根據置信度過濾片段"""
        return [seg for seg in segments if seg.confidence >= min_confidence]

    @staticmethod
    def merge_short_segments(
        segments: List[AlignmentSegment], min_duration: float = 0.2
    ) -> List[AlignmentSegment]:
        """合併短片段"""
        if not segments:
            return segments

        merged = []
        current_segment = segments[0]

        for segment in segments[1:]:
            # 如果當前片段太短，嘗試與下一個合併
            if (current_segment.end - current_segment.start) < min_duration:
                # 合併片段
                merged_text = current_segment.text + " " + segment.text
                merged_tokens = current_segment.tokens + segment.tokens
                merged_confidence = min(current_segment.confidence, segment.confidence)

                current_segment = AlignmentSegment(
                    text=merged_text,
                    start=current_segment.start,
                    end=segment.end,
                    tokens=merged_tokens,
                    confidence=merged_confidence,
                )
            else:
                merged.append(current_segment)
                current_segment = segment

        merged.append(current_segment)
        return merged


if __name__ == "__main__":
    # 測試對齊器
    aligner = WhisperAligner(model_name="base")

    print("Whisper 對齊器初始化完成")
    print(f"設備: {aligner.device}")
    print(f"語言: {aligner.language}")

    # 創建測試音訊（實際使用時需要真實音訊文件）
    # segments = aligner.align_audio_text("test_audio.wav")

    # 測試強制對齊
    test_text = "你好世界，這是一個測試。"
    print(f"\n測試文字: {test_text}")

    # 模擬對齊結果
    mock_segments = [
        AlignmentSegment("你好", 0.0, 0.5, [1, 2], 0.9),
        AlignmentSegment("世界", 0.5, 1.0, [3, 4], 0.8),
        AlignmentSegment("這是", 1.0, 1.5, [5, 6], 0.85),
        AlignmentSegment("一個", 1.5, 2.0, [7, 8], 0.9),
        AlignmentSegment("測試", 2.0, 2.5, [9, 10], 0.95),
    ]

    print(f"\n模擬對齊結果:")
    for i, seg in enumerate(mock_segments):
        print(
            f"  {i+1}. '{seg.text}' [{seg.start:.2f}s - {seg.end:.2f}s] 置信度: {seg.confidence:.2f}"
        )

    # 測試時間到 token 映射
    mapping = aligner.get_time_to_token_mapping(mock_segments)
    print(f"\n時間映射範例（前 10 幀）:")
    for frame in range(min(10, len(mapping))):
        if mapping[frame]:
            print(f"  幀 {frame}: {mapping[frame]}")

    # 測試後處理
    processor = AlignmentPostProcessor()
    smoothed = processor.smooth_boundaries(mock_segments)
    filtered = processor.filter_by_confidence(smoothed, min_confidence=0.8)

    print(f"\n後處理結果:")
    print(f"  原始片段數: {len(mock_segments)}")
    print(f"  平滑後片段數: {len(smoothed)}")
    print(f"  過濾後片段數: {len(filtered)}")
