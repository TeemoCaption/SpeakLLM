#!/usr/bin/env python3
"""
SpeechLLM 資料集準備腳本
支援 AISHELL、WenetSpeech、Common Voice 等語音資料集
針對繁體中文優化
"""

import os
import sys
import argparse
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
from dataclasses import dataclass
import random
import numpy as np

# 添加專案根目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from speechllm.align.alignment import ChineseTextProcessor


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """設置日誌配置"""
    # 創建日誌目錄
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日誌檔案名稱（包含時間戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"prepare_data_{timestamp}.log")
    
    # 配置日誌格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 設置日誌配置
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 檔案輸出
            logging.StreamHandler(sys.stdout)  # 控制台輸出
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日誌系統已初始化，日誌檔案：{log_file}")
    return logger


@dataclass
class DataSample:
    """資料樣本"""

    sample_id: str
    mode: str  # TITO, AITO, TIAO, AIAO
    input_text: str = None
    input_audio_path: str = None
    output_text: str = None
    output_audio_path: str = None
    speaker_id: str = None
    dataset_source: str = None  # AISHELL, CommonVoice, etc.
    metadata: Dict = None


class DatasetProcessor:
    """資料集處理器"""

    def __init__(self, text_processor=None):
        self.text_processor = text_processor or ChineseTextProcessor(
            convert_traditional=False
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_common_voice(
        self,
        data_dir: str,
        split: str = "train",
        language: str = "zh-TW",
        mode: str = "AITO",
    ):
        """處理 Common Voice 繁體中文資料集"""
        self.logger.info(f"開始處理 Common Voice 資料集：{data_dir}, split={split}, language={language}, mode={mode}")
        samples = []

        # Common Voice 資料結構
        tsv_file = os.path.join(data_dir, f"{split}.tsv")
        audio_dir = os.path.join(data_dir, "clips")

        if not os.path.exists(tsv_file):
            self.logger.error(f"找不到 TSV 檔案：{tsv_file}")
            return samples

        # 讀取 TSV 文件
        try:
            self.logger.info(f"讀取 TSV 檔案：{tsv_file}")
            df = pd.read_csv(tsv_file, sep="\t")
            self.logger.info(f"成功讀取 {len(df)} 筆資料")

            for idx, row in df.iterrows():
                audio_file = row["path"]
                text = row["sentence"]

                audio_path = os.path.join(audio_dir, audio_file)
                if not os.path.exists(audio_path):
                    self.logger.warning(f"音訊檔案不存在，跳過：{audio_path}")
                    continue

                # 正規化文字
                normalized_text = self.text_processor.normalize_text(text)

                sample = DataSample(
                    sample_id=f"cv_{language}_{idx}",
                    mode=mode,
                    input_audio_path=audio_path if mode in ["AITO", "AIAO"] else None,
                    input_text=normalized_text if mode in ["TITO", "TIAO"] else None,
                    output_text=normalized_text if mode in ["AITO", "TITO"] else None,
                    output_audio_path=audio_path if mode in ["TIAO", "AIAO"] else None,
                    speaker_id=row.get("client_id", "unknown"),
                    dataset_source="CommonVoice",
                    metadata={
                        "split": split,
                        "language": language,
                        "age": row.get("age", ""),
                        "gender": row.get("gender", ""),
                        "accent": row.get("accent", ""),
                    },
                )
                samples.append(sample)
                
                # 每處理 1000 筆資料記錄一次進度
                if (idx + 1) % 1000 == 0:
                    self.logger.info(f"已處理 {idx + 1} 筆資料，成功建立 {len(samples)} 個樣本")
                    
        except Exception as e:
            self.logger.error(f"處理 Common Voice 資料時出錯: {e}")

        self.logger.info(f"Common Voice 處理完成，共建立 {len(samples)} 個樣本")
        return samples

    def process_aishell3(self, data_dir: str, split: str = "train", mode: str = "TIAO"):
        """處理 AISHELL-3 TTS 資料集"""
        self.logger.info(f"開始處理 AISHELL-3 資料集：{data_dir}, split={split}, mode={mode}")
        samples = []

        # AISHELL-3 資料結構: train/SSB*/SSB*_*.wav 和 train/label_train-set.txt
        split_dir = os.path.join(data_dir, split)
        label_file = os.path.join(data_dir, split, f"label_{split}-set.txt")

        if not os.path.exists(label_file):
            self.logger.error(f"找不到標籤檔案：{label_file}")
            return samples

        # 讀取標籤文件
        transcripts = {}
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        audio_id = parts[0]
                        text = parts[1]
                        transcripts[audio_id] = text

        # 遍歷音訊文件
        for speaker_dir in os.listdir(split_dir):
            speaker_path = os.path.join(split_dir, speaker_dir)
            if not os.path.isdir(speaker_path) or not speaker_dir.startswith("SSB"):
                continue

            for audio_file in os.listdir(speaker_path):
                if not audio_file.endswith(".wav"):
                    continue

                audio_id = audio_file.replace(".wav", "")
                audio_path = os.path.join(speaker_path, audio_file)

                if audio_id in transcripts:
                    # 正規化文字
                    text = self.text_processor.normalize_text(transcripts[audio_id])

                    sample = DataSample(
                        sample_id=audio_id,
                        mode=mode,
                        input_text=text,  # TTS: 文字輸入
                        output_audio_path=audio_path,  # TTS: 音訊輸出
                        speaker_id=speaker_dir,
                        dataset_source="AISHELL-3",
                    )
                    samples.append(sample)

        return samples

    def process_bznsyp(self, data_dir: str, split: str = "train", mode: str = "TIAO"):
        """處理 BZNSYP (Baker/CSMSC) TTS 資料集"""
        samples = []

        # BZNSYP 資料結構: Wave/*.wav 和 ProsodyLabeling/000001-010000.txt
        wave_dir = os.path.join(data_dir, "Wave")
        label_file = os.path.join(data_dir, "ProsodyLabeling", "000001-010000.txt")

        if not os.path.exists(label_file):
            print(f"  警告: 找不到標籤文件 {label_file}")
            return samples

        # 讀取標籤文件
        transcripts = {}
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        audio_id = parts[0]
                        text = parts[1]
                        # 移除韻律標記
                        text = (
                            text.replace("#1", "")
                            .replace("#2", "")
                            .replace("#3", "")
                            .replace("#4", "")
                        )
                        transcripts[audio_id] = text

        # 遍歷音訊文件
        if os.path.exists(wave_dir):
            for audio_file in os.listdir(wave_dir):
                if not audio_file.endswith(".wav"):
                    continue

                audio_id = audio_file.replace(".wav", "")
                audio_path = os.path.join(wave_dir, audio_file)

                if audio_id in transcripts:
                    # 正規化文字
                    text = self.text_processor.normalize_text(transcripts[audio_id])

                    sample = DataSample(
                        sample_id=audio_id,
                        mode=mode,
                        input_text=text,  # TTS: 文字輸入
                        output_audio_path=audio_path,  # TTS: 音訊輸出
                        speaker_id="baker",  # BZNSYP 是單說話人
                        dataset_source="BZNSYP",
                    )
                    samples.append(sample)

        return samples

    def process_esd(self, data_dir: str, split: str = "train", mode: str = "TIAO"):
        """處理 ESD (Emotion Speech Dataset) 中文部分"""
        samples = []

        # ESD 資料結構: 0011/*.wav 和 0011/train.txt
        # 假設只處理中文說話人 0011
        speaker_dir = os.path.join(data_dir, "0011")
        label_file = os.path.join(speaker_dir, f"{split}.txt")

        if not os.path.exists(label_file):
            print(f"  警告: 找不到標籤文件 {label_file}")
            return samples

        # 讀取標籤文件
        transcripts = {}
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("|")
                    if len(parts) >= 3:
                        audio_id = parts[0]
                        text = parts[2]
                        transcripts[audio_id] = text

        # 遍歷音訊文件
        if os.path.exists(speaker_dir):
            for audio_file in os.listdir(speaker_dir):
                if not audio_file.endswith(".wav"):
                    continue

                audio_id = audio_file.replace(".wav", "")
                audio_path = os.path.join(speaker_dir, audio_file)

                if audio_id in transcripts:
                    # 正規化文字
                    text = self.text_processor.normalize_text(transcripts[audio_id])

                    sample = DataSample(
                        sample_id=audio_id,
                        mode=mode,
                        input_text=text,  # TTS: 文字輸入
                        output_audio_path=audio_path,  # TTS: 音訊輸出
                        speaker_id="0011",  # ESD 說話人 ID
                        dataset_source="ESD",
                    )
                    samples.append(sample)

        return samples

    def create_mixed_dataset(
        self,
        dataset_configs: List[Dict],
        output_file: str,
        mode_distribution: Dict[str, float] = None,
    ):
        """創建混合資料集"""
        if mode_distribution is None:
            mode_distribution = {"TITO": 0.3, "AITO": 0.3, "TIAO": 0.2, "AIAO": 0.2}

        all_samples = []

        # 處理每個資料集
        for config in dataset_configs:
            dataset_type = config["type"]
            data_dir = config["data_dir"]
            split = config.get("split", "train")

            if dataset_type == "AISHELL":
                samples = self.process_aishell(data_dir, split)
            elif dataset_type == "CommonVoice":
                language = config.get("language", "zh-TW")
                samples = self.process_common_voice(data_dir, split, language)
            elif dataset_type == "WenetSpeech":
                samples = self.process_wenetspeech(data_dir, split)
            elif dataset_type == "AISHELL-3":
                samples = self.process_aishell3(data_dir, split, "TIAO")  # TTS 模式
            elif dataset_type == "BZNSYP":
                samples = self.process_bznsyp(data_dir, split, "TIAO")  # TTS 模式
            elif dataset_type == "ESD":
                samples = self.process_esd(data_dir, split, "TIAO")  # TTS 模式
            else:
                print(f"不支援的資料集類型: {dataset_type}")
                continue

            all_samples.extend(samples)
            print(f"載入 {dataset_type} {split}: {len(samples)} 個樣本")

        # 隨機分配模式
        for sample in all_samples:
            # 根據分佈隨機選擇模式
            mode = np.random.choice(
                list(mode_distribution.keys()), p=list(mode_distribution.values())
            )
            sample.mode = mode

        # 打亂樣本順序
        random.shuffle(all_samples)

        # 保存到文件
        samples_dict = []
        for sample in all_samples:
            sample_dict = {
                "sample_id": sample.sample_id,
                "mode": sample.mode,
                "input_text": sample.input_text,
                "input_audio_path": sample.input_audio_path,
                "output_text": sample.output_text,
                "output_audio_path": sample.output_audio_path,
                "speaker_id": sample.speaker_id,
                "dataset_source": sample.dataset_source,
                "metadata": sample.metadata,
            }
            samples_dict.append(sample_dict)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples_dict, f, ensure_ascii=False, indent=2)

        print(f"混合資料集已保存到 {output_file}")
        print(f"總樣本數: {len(all_samples)}")

        # 統計模式分佈
        mode_counts = {}
        for sample in all_samples:
            mode_counts[sample.mode] = mode_counts.get(sample.mode, 0) + 1

        print("模式分佈:")
        for mode, count in mode_counts.items():
            percentage = count / len(all_samples) * 100
            print(f"  {mode}: {count} ({percentage:.1f}%)")

        return all_samples


def load_config(config_path: str) -> dict:
    """載入配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def download_aishell(data_dir: str):
    """下載 AISHELL-1 資料集"""
    print("AISHELL-1 資料集下載指南:")
    print("1. 前往 https://www.openslr.org/33/")
    print("2. 下載以下文件:")
    print("   - data_aishell.tgz (音訊資料)")
    print("   - resource_aishell.tgz (資源文件)")
    print(f"3. 解壓到 {data_dir}")
    print("4. 確保目錄結構如下:")
    print("   AISHELL-1/")
    print("   ├── wav/")
    print("   │   ├── train/")
    print("   │   ├── dev/")
    print("   │   └── test/")
    print("   └── transcript/")
    print("       └── aishell_transcript_v0.8.txt")


def download_common_voice(data_dir: str, language: str = "zh-TW"):
    """下載 Common Voice 繁體中文資料集"""
    print(f"Common Voice {language} 資料集下載指南:")
    print("1. 前往 https://commonvoice.mozilla.org/zh-TW/datasets")
    print("2. 註冊並同意授權條款")
    print(f"3. 下載 {language} 資料集")
    print(f"4. 解壓到 {data_dir}")
    print("5. 確保目錄結構如下:")
    print(f"   CommonVoice/{language}/")
    print("   ├── clips/")
    print("   ├── train.tsv")
    print("   ├── dev.tsv")
    print("   ├── test.tsv")
    print("   └── validated.tsv")


def download_aishell3(data_dir: str):
    """下載 AISHELL-3 TTS 資料集"""
    print("AISHELL-3 TTS 資料集下載指南:")
    print("1. 前往 https://www.openslr.org/93/")
    print("2. 下載以下文件:")
    print("   - AISHELL-3.tar.gz")
    print(f"3. 解壓到 {data_dir}")
    print("4. 確保目錄結構如下:")
    print("   AISHELL-3/")
    print("   ├── train/")
    print("   │   ├── SSB*/*.wav")
    print("   │   └── label_train-set.txt")
    print("   └── test/")
    print("       ├── SSB*/*.wav")
    print("       └── label_test-set.txt")


def download_esd(data_dir: str):
    """下載 ESD (Emotion Speech Dataset) 中文部分"""
    print("ESD (Emotion Speech Dataset) 中文部分下載指南:")
    print("1. 前往 https://github.com/HLTSingapore/Emotional-Speech-Data")
    print("2. 按照說明下載 ESD 資料集")
    print("3. 提取中文說話人 (0011) 的資料")
    print(f"4. 解壓到 {data_dir}")
    print("5. 確保目錄結構如下:")
    print("   ESD/")
    print("   └── 0011/")
    print("       ├── *.wav")
    print("       └── train.txt")


def prepare_dataset(
    dataset_type: str,
    data_dir: str,
    output_dir: str,
    splits: List[str] = ["train", "dev", "test"],
    language: str = "zh-TW",
):
    """準備單個資料集"""
    logger = logging.getLogger(__name__)
    logger.info(f"開始準備 {dataset_type} 資料集，資料目錄: {data_dir}")

    processor = DatasetProcessor()

    for split in splits:
        logger.info(f"處理 {dataset_type} 的 {split} 分割")

        try:
            if dataset_type == "AISHELL":
                samples = processor.process_aishell(data_dir, split)
            elif dataset_type == "CommonVoice":
                samples = processor.process_common_voice(data_dir, split, language)
            elif dataset_type == "AISHELL-3":
                samples = processor.process_aishell3(data_dir, split, "TIAO")
            elif dataset_type == "ESD":
                samples = processor.process_esd(data_dir, split, "TIAO")
            else:
                print(f"不支援的資料集類型: {dataset_type}")
                continue

            if samples:
                # 保存處理後的資料
                output_file = os.path.join(
                    output_dir, f"{dataset_type.lower()}_{split}.json"
                )

                samples_dict = []
                for sample in samples:
                    sample_dict = {
                        "sample_id": sample.sample_id,
                        "mode": sample.mode,
                        "input_text": sample.input_text,
                        "input_audio_path": sample.input_audio_path,
                        "output_text": sample.output_text,
                        "output_audio_path": sample.output_audio_path,
                        "speaker_id": sample.speaker_id,
                        "dataset_source": sample.dataset_source,
                        "metadata": sample.metadata,
                    }
                    samples_dict.append(sample_dict)

                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(samples_dict, f, ensure_ascii=False, indent=2)

                print(f"  {split}: {len(samples)} 個樣本 -> {output_file}")
            else:
                print(f"  {split}: 未找到樣本")

        except Exception as e:
            print(f"  處理 {split} 時出錯: {e}")


def create_mixed_dataset(config: dict, output_dir: str):
    """創建混合資料集"""
    print("\n創建混合資料集...")

    data_config = config["data"]
    datasets = data_config.get("chinese_datasets", [])
    mode_weights = data_config.get("mode_weights", {})

    processor = DatasetProcessor()

    # 從配置檔案中獲取檔案路徑對應
    file_mapping = {
        "train": data_config.get("train_data_file", "./datasets/train_chinese.json"),
        "dev": data_config.get("eval_data_file", "./datasets/dev_chinese.json"),
        "test": os.path.join(output_dir, "test_chinese.json")  # test 通常沒有在配置中指定
    }

    # 為每個分割創建混合資料集
    for split in ["train", "dev", "test"]:
        print(f"\n處理 {split} 分割...")

        # 準備資料集配置
        dataset_configs = []
        for dataset_config in datasets:
            if split in dataset_config.get("splits", []):
                config_copy = dataset_config.copy()
                config_copy["split"] = split
                dataset_configs.append(config_copy)

        if not dataset_configs:
            print(f"  跳過 {split}：沒有可用的資料集")
            continue

        # 使用配置檔案中指定的檔案路徑
        output_file = file_mapping[split]
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            mixed_samples = processor.create_mixed_dataset(
                dataset_configs=dataset_configs,
                output_file=output_file,
                mode_distribution=mode_weights,
            )
            print(f"  {split}: {len(mixed_samples)} 個樣本 -> {output_file}")

        except Exception as e:
            print(f"  處理 {split} 時出錯: {e}")


def validate_dataset(data_file: str):
    """驗證資料集"""
    print(f"\n驗證資料集: {data_file}")

    if not os.path.exists(data_file):
        print(f"  錯誤: 文件不存在")
        return False

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"  總樣本數: {len(data)}")

        # 統計模式分佈
        mode_counts = {}
        missing_files = 0

        for sample in data:
            mode = sample.get("mode", "unknown")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

            # 檢查文件是否存在
            for path_key in ["input_audio_path", "output_audio_path"]:
                path = sample.get(path_key)
                if path and not os.path.exists(path):
                    missing_files += 1

        print(f"  模式分佈:")
        for mode, count in mode_counts.items():
            percentage = count / len(data) * 100
            print(f"    {mode}: {count} ({percentage:.1f}%)")

        if missing_files > 0:
            print(f"  警告: {missing_files} 個音訊文件不存在")
        else:
            print(f"  所有音訊文件都存在")

        return True

    except Exception as e:
        print(f"  錯誤: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="SpeechLLM 資料集準備腳本")
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml", help="配置文件路徑"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["download", "prepare", "mixed", "validate", "all"],
        default="all",
        help="執行動作",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["AISHELL", "CommonVoice", "WenetSpeech", "AISHELL-3", "BZNSYP", "ESD"],
        help="指定資料集（僅用於 prepare 動作）",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="輸出目錄（若未指定則使用配置檔案中的路徑）")

    args = parser.parse_args()

    # 載入配置
    config = load_config(args.config)
    data_config = config["data"]
    
    # 從配置檔案中推導輸出目錄
    if args.output_dir is None:
        # 從配置檔案中的 train_data_file 路徑推導輸出目錄
        train_data_file = data_config.get("train_data_file", "./datasets/train_chinese.json")
        args.output_dir = os.path.dirname(train_data_file)
        if not args.output_dir:
            args.output_dir = "./datasets"
    
    # 初始化日誌系統
    logger = setup_logging(log_dir=os.path.join(args.output_dir, "logs"))
    logger.info(f"載入配置文件: {args.config}")
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"輸出目錄: {args.output_dir} (從配置檔案推導)")

    if args.action in ["download", "all"]:
        print("\n=== 資料集下載指南 ===")

        for dataset_config in data_config.get("chinese_datasets", []):
            dataset_type = dataset_config["type"]
            data_dir = dataset_config["data_dir"]

            if dataset_type == "AISHELL":
                download_aishell(data_dir)
            elif dataset_type == "CommonVoice":
                language = dataset_config.get("language", "zh-TW")
                download_common_voice(data_dir, language)
            elif dataset_type == "WenetSpeech":
                download_wenetspeech(data_dir)
            elif dataset_type == "AISHELL-3":
                download_aishell3(data_dir)
            elif dataset_type == "BZNSYP":
                download_bznsyp(data_dir)
            elif dataset_type == "ESD":
                download_esd(data_dir)

            print()

    if args.action in ["prepare", "all"]:
        print("\n=== 準備資料集 ===")

        datasets_to_prepare = []
        if args.dataset:
            # 準備指定資料集
            for dataset_config in data_config.get("chinese_datasets", []):
                if dataset_config["type"] == args.dataset:
                    datasets_to_prepare.append(dataset_config)
        else:
            # 準備所有資料集
            datasets_to_prepare = data_config.get("chinese_datasets", [])

        for dataset_config in datasets_to_prepare:
            dataset_type = dataset_config["type"]
            data_dir = dataset_config["data_dir"]
            splits = dataset_config.get("splits", ["train", "dev", "test"])
            language = dataset_config.get("language", "zh-TW")

            if os.path.exists(data_dir):
                prepare_dataset(
                    dataset_type, data_dir, args.output_dir, splits, language
                )
            else:
                print(f"跳過 {dataset_type}：資料目錄不存在 {data_dir}")

    if args.action in ["mixed", "all"]:
        print("\n=== 創建混合資料集 ===")
        create_mixed_dataset(config, args.output_dir)

    if args.action in ["validate", "all"]:
        print("\n=== 驗證資料集 ===")

        # 驗證混合資料集 - 使用配置檔案中指定的路徑
        validation_files = {
            "train": data_config.get("train_data_file", "./datasets/train_chinese.json"),
            "dev": data_config.get("eval_data_file", "./datasets/dev_chinese.json"),
            "test": os.path.join(args.output_dir, "test_chinese.json")
        }
        
        for split, data_file in validation_files.items():
            if os.path.exists(data_file):
                validate_dataset(data_file)
            else:
                print(f"  跳過 {split}：檔案不存在 {data_file}")

    print("\n資料集準備完成！")
    print("\n後續步驟:")
    print("1. 檢查生成的資料文件")
    print("2. 更新配置文件中的資料路徑")
    print("3. 開始訓練: python scripts/train.py")


if __name__ == "__main__":
    main()
