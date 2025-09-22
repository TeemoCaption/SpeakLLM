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
from pathlib import Path
from typing import Dict, List
import pandas as pd
from dataclasses import dataclass
import random
import numpy as np

# 添加專案根目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from speechllm.align.alignment import ChineseTextProcessor


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
    dataset_source: str = None  # AISHELL, WenetSpeech, CommonVoice, etc.
    metadata: Dict = None


class DatasetProcessor:
    """資料集處理器"""
    
    def __init__(self, text_processor=None):
        self.text_processor = text_processor or ChineseTextProcessor(convert_traditional=False)
    
    def process_aishell(self, data_dir: str, split: str = "train", mode: str = "AITO"):
        """處理 AISHELL 資料集"""
        samples = []
        
        # AISHELL 資料結構
        audio_dir = os.path.join(data_dir, "wav", split)
        transcript_file = os.path.join(data_dir, "transcript", f"aishell_transcript_v0.8.txt")
        
        # 讀取轉錄文件
        transcripts = {}
        if os.path.exists(transcript_file):
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        audio_id, text = parts
                        transcripts[audio_id] = text
        
        # 遍歷音訊文件
        if os.path.exists(audio_dir):
            for speaker_dir in os.listdir(audio_dir):
                speaker_path = os.path.join(audio_dir, speaker_dir)
                if not os.path.isdir(speaker_path):
                    continue
                    
                for audio_file in os.listdir(speaker_path):
                    if not audio_file.endswith('.wav'):
                        continue
                    
                    audio_id = audio_file.replace('.wav', '')
                    audio_path = os.path.join(speaker_path, audio_file)
                    
                    if audio_id in transcripts:
                        # 正規化文字
                        text = self.text_processor.normalize_text(transcripts[audio_id])
                        
                        sample = DataSample(
                            sample_id=audio_id,
                            mode=mode,
                            input_audio_path=audio_path if mode in ["AITO", "AIAO"] else None,
                            input_text=text if mode in ["TITO", "TIAO"] else None,
                            output_text=text if mode in ["AITO", "TITO"] else None,
                            output_audio_path=audio_path if mode in ["TIAO", "AIAO"] else None,
                            speaker_id=speaker_dir,
                            dataset_source="AISHELL",
                            metadata={"split": split}
                        )
                        samples.append(sample)
        
        return samples
    
    def process_common_voice(self, data_dir: str, split: str = "train", language: str = "zh-TW", mode: str = "AITO"):
        """處理 Common Voice 繁體中文資料集"""
        samples = []
        
        # Common Voice 資料結構
        tsv_file = os.path.join(data_dir, f"{split}.tsv")
        audio_dir = os.path.join(data_dir, "clips")
        
        if not os.path.exists(tsv_file):
            print(f"找不到 {tsv_file}")
            return samples
        
        # 讀取 TSV 文件
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            
            for idx, row in df.iterrows():
                audio_file = row['path']
                text = row['sentence']
                
                audio_path = os.path.join(audio_dir, audio_file)
                if not os.path.exists(audio_path):
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
                    speaker_id=row.get('client_id', 'unknown'),
                    dataset_source="CommonVoice",
                    metadata={
                        "split": split,
                        "language": language,
                        "age": row.get('age', ''),
                        "gender": row.get('gender', ''),
                        "accent": row.get('accent', '')
                    }
                )
                samples.append(sample)
        except Exception as e:
            print(f"處理 Common Voice 資料時出錯: {e}")
        
        return samples
    
    def process_wenetspeech(self, data_dir: str, split: str = "train", mode: str = "AITO"):
        """處理 WenetSpeech 資料集"""
        samples = []
        
        # WenetSpeech 資料結構
        manifest_file = os.path.join(data_dir, f"{split}.json")
        
        if not os.path.exists(manifest_file):
            print(f"找不到 {manifest_file}")
            return samples
        
        # 讀取 manifest 文件
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    audio_path = data['audio_filepath']
                    text = data['text']
                    
                    # 處理相對路徑
                    if not os.path.isabs(audio_path):
                        audio_path = os.path.join(data_dir, audio_path)
                    
                    if not os.path.exists(audio_path):
                        continue
                    
                    # 正規化文字
                    normalized_text = self.text_processor.normalize_text(text)
                    
                    sample = DataSample(
                        sample_id=f"wenetspeech_{split}_{line_idx}",
                        mode=mode,
                        input_audio_path=audio_path if mode in ["AITO", "AIAO"] else None,
                        input_text=normalized_text if mode in ["TITO", "TIAO"] else None,
                        output_text=normalized_text if mode in ["AITO", "TITO"] else None,
                        output_audio_path=audio_path if mode in ["TIAO", "AIAO"] else None,
                        speaker_id=data.get('speaker', 'unknown'),
                        dataset_source="WenetSpeech",
                        metadata={
                            "split": split,
                            "duration": data.get('duration', 0),
                            "domain": data.get('domain', ''),
                            "source": data.get('source', '')
                        }
                    )
                    samples.append(sample)
                    
                except json.JSONDecodeError:
                    continue
        
        return samples
    
    def create_mixed_dataset(self, dataset_configs: List[Dict], output_file: str, mode_distribution: Dict[str, float] = None):
        """創建混合資料集"""
        if mode_distribution is None:
            mode_distribution = {
                "TITO": 0.3,
                "AITO": 0.3,
                "TIAO": 0.2,
                "AIAO": 0.2
            }
        
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
            else:
                print(f"不支援的資料集類型: {dataset_type}")
                continue
            
            all_samples.extend(samples)
            print(f"載入 {dataset_type} {split}: {len(samples)} 個樣本")
        
        # 隨機分配模式
        for sample in all_samples:
            # 根據分佈隨機選擇模式
            mode = np.random.choice(
                list(mode_distribution.keys()),
                p=list(mode_distribution.values())
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
                "metadata": sample.metadata
            }
            samples_dict.append(sample_dict)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
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
    with open(config_path, 'r', encoding='utf-8') as f:
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


def download_wenetspeech(data_dir: str):
    """下載 WenetSpeech 資料集"""
    print("WenetSpeech 資料集下載指南:")
    print("1. 前往 https://github.com/wenet-e2e/WenetSpeech")
    print("2. 按照說明申請資料集存取權限")
    print("3. 使用提供的腳本下載資料")
    print(f"4. 解壓到 {data_dir}")
    print("5. 確保目錄結構如下:")
    print("   WenetSpeech/")
    print("   ├── audio/")
    print("   ├── train.json")
    print("   ├── dev.json")
    print("   └── test.json")


def prepare_dataset(
    dataset_type: str,
    data_dir: str,
    output_dir: str,
    splits: List[str] = ["train", "dev", "test"],
    language: str = "zh-TW"
):
    """準備單個資料集"""
    print(f"\n準備 {dataset_type} 資料集...")
    
    processor = DatasetProcessor()
    
    for split in splits:
        print(f"處理 {split} 分割...")
        
        try:
            if dataset_type == "AISHELL":
                samples = processor.process_aishell(data_dir, split)
            elif dataset_type == "CommonVoice":
                samples = processor.process_common_voice(data_dir, split, language)
            elif dataset_type == "WenetSpeech":
                samples = processor.process_wenetspeech(data_dir, split)
            else:
                print(f"不支援的資料集類型: {dataset_type}")
                continue
            
            if samples:
                # 保存處理後的資料
                output_file = os.path.join(output_dir, f"{dataset_type.lower()}_{split}.json")
                
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
                        "metadata": sample.metadata
                    }
                    samples_dict.append(sample_dict)
                
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(samples_dict, f, ensure_ascii=False, indent=2)
                
                print(f"  {split}: {len(samples)} 個樣本 -> {output_file}")
            else:
                print(f"  {split}: 未找到樣本")
                
        except Exception as e:
            print(f"  處理 {split} 時出錯: {e}")


def create_mixed_dataset(config: dict, output_dir: str):
    """創建混合資料集"""
    print("\n創建混合資料集...")
    
    data_config = config['data']
    datasets = data_config.get('chinese_datasets', [])
    mode_weights = data_config.get('mode_weights', {})
    
    processor = DatasetProcessor()
    
    # 為每個分割創建混合資料集
    for split in ["train", "dev", "test"]:
        print(f"\n處理 {split} 分割...")
        
        # 準備資料集配置
        dataset_configs = []
        for dataset_config in datasets:
            if split in dataset_config.get('splits', []):
                config_copy = dataset_config.copy()
                config_copy['split'] = split
                dataset_configs.append(config_copy)
        
        if not dataset_configs:
            print(f"  跳過 {split}：沒有可用的資料集")
            continue
        
        # 創建混合資料集
        output_file = os.path.join(output_dir, f"{split}_chinese.json")
        
        try:
            mixed_samples = processor.create_mixed_dataset(
                dataset_configs=dataset_configs,
                output_file=output_file,
                mode_distribution=mode_weights
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
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  總樣本數: {len(data)}")
        
        # 統計模式分佈
        mode_counts = {}
        missing_files = 0
        
        for sample in data:
            mode = sample.get('mode', 'unknown')
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            # 檢查文件是否存在
            for path_key in ['input_audio_path', 'output_audio_path']:
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
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="配置文件路徑"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["download", "prepare", "mixed", "validate", "all"],
        default="all",
        help="執行動作"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["AISHELL", "CommonVoice", "WenetSpeech"],
        help="指定資料集（僅用於 prepare 動作）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="輸出目錄"
    )
    
    args = parser.parse_args()
    
    # 載入配置
    print(f"載入配置文件: {args.config}")
    config = load_config(args.config)
    data_config = config['data']
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.action in ["download", "all"]:
        print("\n=== 資料集下載指南 ===")
        
        for dataset_config in data_config.get('chinese_datasets', []):
            dataset_type = dataset_config['type']
            data_dir = dataset_config['data_dir']
            
            if dataset_type == "AISHELL":
                download_aishell(data_dir)
            elif dataset_type == "CommonVoice":
                language = dataset_config.get('language', 'zh-TW')
                download_common_voice(data_dir, language)
            elif dataset_type == "WenetSpeech":
                download_wenetspeech(data_dir)
            
            print()
    
    if args.action in ["prepare", "all"]:
        print("\n=== 準備資料集 ===")
        
        datasets_to_prepare = []
        if args.dataset:
            # 準備指定資料集
            for dataset_config in data_config.get('chinese_datasets', []):
                if dataset_config['type'] == args.dataset:
                    datasets_to_prepare.append(dataset_config)
        else:
            # 準備所有資料集
            datasets_to_prepare = data_config.get('chinese_datasets', [])
        
        for dataset_config in datasets_to_prepare:
            dataset_type = dataset_config['type']
            data_dir = dataset_config['data_dir']
            splits = dataset_config.get('splits', ['train', 'dev', 'test'])
            language = dataset_config.get('language', 'zh-TW')
            
            if os.path.exists(data_dir):
                prepare_dataset(dataset_type, data_dir, args.output_dir, splits, language)
            else:
                print(f"跳過 {dataset_type}：資料目錄不存在 {data_dir}")
    
    if args.action in ["mixed", "all"]:
        print("\n=== 創建混合資料集 ===")
        create_mixed_dataset(config, args.output_dir)
    
    if args.action in ["validate", "all"]:
        print("\n=== 驗證資料集 ===")
        
        # 驗證混合資料集
        for split in ["train", "dev", "test"]:
            data_file = os.path.join(args.output_dir, f"{split}_chinese.json")
            if os.path.exists(data_file):
                validate_dataset(data_file)
    
    print("\n資料集準備完成！")
    print("\n後續步驟:")
    print("1. 檢查生成的資料文件")
    print("2. 更新配置文件中的資料路徑")
    print("3. 開始訓練: python scripts/train.py")


if __name__ == "__main__":
    main()
