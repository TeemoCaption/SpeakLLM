#!/usr/bin/env python3
"""
SpeechLLM 訓練腳本
執行三階段訓練流程
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from speechllm.models.speechllm import SpeechLLM, SpeechLLMConfig
from speechllm.data.dataset import SpeechLLMDataset
from speechllm.training.trainer import SpeechLLMTrainer, TrainingConfig
from speechllm.codecs.vocab_manager import VocabManager
from speechllm.codecs.audio_tokenizer import AudioTokenizer
from speechllm.align.interleaving import InterleavingGenerator


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """設置日誌配置"""
    # 創建日誌目錄
    os.makedirs(log_dir, exist_ok=True)

    # 生成日誌檔案名稱（包含時間戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    # 配置日誌格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 設置日誌配置
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),  # 檔案輸出
            logging.StreamHandler(sys.stdout),  # 控制台輸出
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"訓練日誌系統已初始化，日誌檔案：{log_file}")
    return logger


def set_seed(seed: int):
    """設置隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """載入配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_model_config(config: dict) -> SpeechLLMConfig:
    """創建模型配置"""
    model_config = config["model"]
    return SpeechLLMConfig(
        llm_model_name=model_config["llm_model_name"],
        freeze_llm=model_config["freeze_llm"],
        use_lora=model_config["use_lora"],
        lora_rank=model_config["lora_rank"],
        lora_alpha=model_config["lora_alpha"],
        whisper_model_name=model_config["whisper_model_name"],
        freeze_whisper=model_config["freeze_whisper"],
        num_query_tokens=model_config["num_query_tokens"],
        qformer_hidden_size=model_config["qformer_hidden_size"],
        qformer_num_layers=model_config["qformer_num_layers"],
        audio_transformer_layers=model_config["audio_transformer_layers"],
        audio_transformer_hidden_size=model_config["audio_transformer_hidden_size"],
        num_rvq_layers=model_config["num_rvq_layers"],
        codebook_size=model_config["codebook_size"],
        use_gradient_checkpointing=model_config["use_gradient_checkpointing"],
        mixed_precision=model_config["mixed_precision"],
    )


def create_training_config(config: dict) -> TrainingConfig:
    """創建中文優化的訓練配置"""
    training_config = config["training"]

    learning_rate_value = training_config["learning_rate"]
    if isinstance(learning_rate_value, str):
        learning_rate_value = float(learning_rate_value)

    weight_decay_value = training_config["weight_decay"]
    if isinstance(weight_decay_value, str):
        weight_decay_value = float(weight_decay_value)

    # 創建訓練配置，支援中文優化參數
    train_config = TrainingConfig(
        output_dir=training_config["output_dir"],
        run_name=training_config["run_name"],
        seed=training_config["seed"],
        num_epochs=training_config["num_epochs"],
        batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=learning_rate_value,
        weight_decay=weight_decay_value,
        warmup_steps=training_config["warmup_steps"],
        max_grad_norm=training_config["max_grad_norm"],
        stage_a_epochs=training_config["stage_a_epochs"],
        stage_b_epochs=training_config["stage_b_epochs"],
        stage_c_epochs=training_config["stage_c_epochs"],
        text_loss_weight=training_config["text_loss_weight"],
        rvq_loss_weight=training_config["rvq_loss_weight"],
        alignment_loss_weight=training_config["alignment_loss_weight"],
        kl_loss_weight=training_config["kl_loss_weight"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        logging_steps=training_config["logging_steps"],
        save_total_limit=training_config["save_total_limit"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        device=training_config.get("device"),
        dataloader_num_workers=training_config["dataloader_num_workers"],
        dataloader_pin_memory=training_config["dataloader_pin_memory"],
        use_wandb=training_config["use_wandb"],
        wandb_project=training_config["wandb_project"],
        wandb_entity=training_config.get("wandb_entity"),
    )

    # 設定中文優化的 RVQ 層權重
    if "rvq_layer_weights" in training_config:
        train_config.rvq_layer_weights = training_config["rvq_layer_weights"]

    return train_config


def main():
    parser = argparse.ArgumentParser(description="SpeechLLM 訓練腳本")
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml", help="配置文件路徑"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="從檢查點恢復訓練"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="分散式訓練的本地排名"
    )

    args = parser.parse_args()

    # 初始化日誌系統
    logger = setup_logging(log_dir="logs")

    # 載入配置
    logger.info(f"載入配置文件: {args.config}")
    config = load_config(args.config)

    # 設置環境變數
    if 'environment' in config:
        logger.info("設置環境變數")
        env_config = config['environment']
        if 'cuda_visible_devices' in env_config:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(env_config['cuda_visible_devices'])
            logger.info(f"CUDA_VISIBLE_DEVICES: {env_config['cuda_visible_devices']}")
        if 'torch_num_threads' in env_config:
            torch.set_num_threads(env_config['torch_num_threads'])
            logger.info(f"Torch threads: {env_config['torch_num_threads']}")
        if 'omp_num_threads' in env_config:
            os.environ['OMP_NUM_THREADS'] = str(env_config['omp_num_threads'])
            logger.info(f"OMP threads: {env_config['omp_num_threads']}")

    # 創建配置物件
    logger.info("創建模型和訓練配置")
    model_config = create_model_config(config)
    training_config = create_training_config(config)

    # 設置隨機種子
    logger.info(f"設置隨機種子: {training_config.seed}")
    set_seed(training_config.seed)

    logger.info("配置載入完成:")
    logger.info(f"  模型: {model_config.llm_model_name}")
    logger.info(f"  輸出目錄: {training_config.output_dir}")
    logger.info(f"  批次大小: {training_config.batch_size}")
    logger.info(f"  學習率: {training_config.learning_rate}")
    logger.info(f"  總 epochs: {training_config.num_epochs}")

    # 創建輸出目錄
    os.makedirs(training_config.output_dir, exist_ok=True)
    logger.info(f"創建輸出目錄: {training_config.output_dir}")

    # 保存配置
    config_save_path = os.path.join(training_config.output_dir, "config.yaml")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"配置已保存到: {config_save_path}")

    # 初始化模型
    logger.info("初始化模型...")
    model = SpeechLLM(model_config)
    logger.info(f"模型初始化完成，參數量: {model.get_trainable_parameters():,}")

    # 載入資料集
    logger.info("載入資料集...")
    data_config = config["data"]

    # 初始化組件
    logger.info("初始化詞彙管理器和音訊編碼器")
    vocab_manager = VocabManager(
        base_tokenizer_name=model_config.llm_model_name,
        num_rvq_layers=model_config.num_rvq_layers,
        codebook_size=model_config.codebook_size,
    )

    audio_tokenizer = AudioTokenizer(vocab_manager=vocab_manager)
    interleaving_generator = InterleavingGenerator(
        audio_tokenizer=audio_tokenizer, vocab_manager=vocab_manager
    )

    # 創建訓練資料集
    logger.info(f"創建訓練資料集，資料檔案: {data_config['train_data_file']}")
    train_dataset = SpeechLLMDataset(
        data_file=data_config["train_data_file"],
        audio_tokenizer=audio_tokenizer,
        vocab_manager=vocab_manager,
        interleaving_generator=interleaving_generator,
        max_text_length=data_config["max_text_length"],
        max_audio_length=data_config["max_audio_length"],
        sample_rate=data_config["sample_rate"],
        mode_weights=data_config["mode_weights"],
        cache_audio_tokens=data_config["cache_audio_tokens"],
        cache_dir=data_config.get("cache_dir"),
    )
    logger.info(f"訓練資料集載入完成: {len(train_dataset)} 個樣本")

    # 創建評估資料集（如果存在）
    eval_dataset = None
    if os.path.exists(data_config["eval_data_file"]):
        logger.info(f"創建評估資料集，資料檔案: {data_config['eval_data_file']}")
        eval_dataset = SpeechLLMDataset(
            data_file=data_config["eval_data_file"],
            audio_tokenizer=audio_tokenizer,
            vocab_manager=vocab_manager,
            interleaving_generator=interleaving_generator,
            max_text_length=data_config["max_text_length"],
            max_audio_length=data_config["max_audio_length"],
            sample_rate=data_config["sample_rate"],
            cache_audio_tokens=data_config["cache_audio_tokens"],
            cache_dir=data_config.get("cache_dir"),
        )
        logger.info(f"評估資料集載入完成: {len(eval_dataset)} 個樣本")
    else:
        logger.warning(f"評估資料檔案不存在: {data_config['eval_data_file']}")

    # 創建訓練器
    logger.info("創建訓練器...")
    trainer = SpeechLLMTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=training_config,
    )

    # 從檢查點恢復（如果指定）
    if args.resume_from_checkpoint:
        logger.info(f"從檢查點恢復訓練: {args.resume_from_checkpoint}")
        trainer.load_checkpoint(args.resume_from_checkpoint)

    # 開始訓練
    logger.info("開始訓練...")
    try:
        trainer.train()
        logger.info("訓練完成！")
    except KeyboardInterrupt:
        logger.warning("訓練被中斷")
        trainer.save_checkpoint()
        logger.info("檢查點已保存")
    except Exception as e:
        logger.error(f"訓練過程中出現錯誤: {e}")
        trainer.save_checkpoint()
        logger.info("檢查點已保存")
        raise


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
