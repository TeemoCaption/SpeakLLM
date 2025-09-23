"""
SpeechLLM 訓練器
實現三階段訓練策略：輸入對齊、語音輸出、多任務聯訓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import wandb
from tqdm import tqdm
import numpy as np

from ..models.speechllm import SpeechLLM, SpeechLLMConfig
from ..data.dataset import SpeechLLMDataset, create_dataloader
from .loss import SpeechLLMLoss
from .optimizer import create_optimizer_and_scheduler


@dataclass
class TrainingConfig:
    """訓練配置"""
    # 基本配置
    output_dir: str = "./outputs"
    run_name: str = "speechllm_training"
    seed: int = 42
    
    # 訓練參數
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # 中文三階段訓練
    stage_a_epochs: int = 4  # 中文輸入對齊階段（AISHELL/THCHS/Common Voice）
    stage_b_epochs: int = 4  # 中文語音輸出階段（AISHELL-3/BZNSYP/ESD）
    stage_c_epochs: int = 6  # 中文多任務聯訓階段（四種模式+多口音）
    
    # 中文優化損失權重
    text_loss_weight: float = 1.0
    rvq_loss_weight: float = 0.6  # 提高 RVQ 損失權重
    alignment_loss_weight: float = 0.4  # 提高對齊損失權重（中文重要）
    kl_loss_weight: float = 0.3  # 提高 KL 蒸餾權重（Stage A 重要）
    
    # 階層式 RVQ 損失權重（L1 語義層 > L2-L4 聲學層）
    rvq_layer_weights: List[float] = None  # 將在 __post_init__ 中設定
    
    # 評估和保存
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    
    # 硬體配置
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Wandb 配置
    use_wandb: bool = True
    wandb_project: str = "speechllm"
    wandb_entity: Optional[str] = None
    
    def __post_init__(self):
        """初始化後處理"""
        if self.rvq_layer_weights is None:
            # 設定階層式 RVQ 損失權重（L1 語義層 > L2-L4 聲學層）
            self.rvq_layer_weights = [3.0, 1.5, 1.0, 1.0]  # L1, L2, L3, L4


class SpeechLLMTrainer:
    """
    SpeechLLM 訓練器
    實現三階段訓練策略
    """
    
    def __init__(
        self,
        model: SpeechLLM,
        train_dataset: SpeechLLMDataset,
        eval_dataset: Optional[SpeechLLMDataset] = None,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainingConfig()
        
        # 設置日誌
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 設置設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用設備: {self.device}")
        self.model.to(self.device)
        
        # 創建輸出目錄
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 初始化損失函數
        self.loss_fn = SpeechLLMLoss(
            text_weight=self.config.text_loss_weight,
            rvq_weight=self.config.rvq_loss_weight,
            alignment_weight=self.config.alignment_loss_weight,
            kl_weight=self.config.kl_loss_weight
        )
        
        # 創建資料載入器
        self.train_dataloader = create_dataloader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers
        )
        
        if self.eval_dataset:
            self.eval_dataloader = create_dataloader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_num_workers
            )
        
        # 創建優化器和調度器
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            model=self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            total_steps=len(self.train_dataloader) * self.config.num_epochs
        )
        
        # 混合精度訓練
        self.scaler = None
        if self.config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 訓練狀態
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        
        # 初始化 Wandb
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_name,
                config=asdict(self.config)
            )
    
    def train(self):
        """執行完整的三階段訓練"""
        print("開始 SpeechLLM 三階段訓練")
        print(f"Stage A (輸入對齊): {self.config.stage_a_epochs} epochs")
        print(f"Stage B (語音輸出): {self.config.stage_b_epochs} epochs")
        print(f"Stage C (多任務聯訓): {self.config.stage_c_epochs} epochs")
        
        # Stage A: 輸入對齊/蒸餾
        print("\n=== Stage A: 輸入對齊/蒸餾 ===")
        self._train_stage_a()
        
        # Stage B: 語音輸出器
        print("\n=== Stage B: 語音輸出器 ===")
        self._train_stage_b()
        
        # Stage C: 多任務聯訓
        print("\n=== Stage C: 多任務聯訓 ===")
        self._train_stage_c()
        
        print("訓練完成！")
        
        # 保存最終模型
        self.save_model("final_model")
        
        if self.config.use_wandb:
            wandb.finish()
    
    def _train_stage_a(self):
        """Stage A: 中文輸入對齊/蒸餾階段"""
        print("Stage A: 中文語音理解訓練（AISHELL/THCHS/Common Voice）")
        
        # 只訓練 Whisper encoder 和 Q-Former
        self._freeze_components(["llm", "audio_transformer"])
        
        # 使用中文 ASR 資料：AITO 模式
        stage_a_modes = ["AITO"]
        
        for epoch in range(self.config.stage_a_epochs):
            self.current_epoch = epoch
            print(f"Stage A - Epoch {epoch + 1}/{self.config.stage_a_epochs}")
            print("  目標：讓模型聽懂中文語音")
            print("  資料：AISHELL/THCHS/Common Voice zh")
            print("  方法：輸入對齊 + KL 蒸餾")
            
            self._train_epoch(
                allowed_modes=stage_a_modes,
                stage="A"
            )
            
            if self.eval_dataset:
                self._evaluate(stage="A")
    
    def _train_stage_b(self):
        """Stage B: 中文語音生成階段"""
        print("Stage B: 中文語音生成訓練（AISHELL-3/BZNSYP/ESD）")
        
        # 解凍音訊 Transformer，凍結其他組件
        self._freeze_components(["llm", "whisper_encoder", "qformer"])
        self._unfreeze_components(["audio_transformer"])
        
        # 使用中文 TTS 資料：TIAO 和 AIAO 模式
        stage_b_modes = ["TIAO", "AIAO"]
        
        for epoch in range(self.config.stage_b_epochs):
            self.current_epoch = epoch + self.config.stage_a_epochs
            print(f"Stage B - Epoch {epoch + 1}/{self.config.stage_b_epochs}")
            print("  目標：讓模型會說中文")
            print("  資料：AISHELL-3/BZNSYP/ESD 等 TTS 資料")
            print("  方法：階層式 RVQ 生成（先語義 L1，再聲學 L2-L4）")
            
            self._train_epoch(
                allowed_modes=stage_b_modes,
                stage="B"
            )
            
            if self.eval_dataset:
                self._evaluate(stage="B")
    
    def _train_stage_c(self):
        """Stage C: 中文多任務聯訓階段"""
        print("Stage C: 中文多任務聯合訓練（混合四種模式）")
        
        # 解凍所有組件（或使用 LoRA）
        if not self.config.freeze_llm:
            self._unfreeze_components(["llm", "whisper_encoder", "qformer", "audio_transformer"])
        
        # 使用所有中文任務模式，混入多口音資料
        stage_c_modes = ["TITO", "AITO", "TIAO", "AIAO"]
        
        for epoch in range(self.config.stage_c_epochs):
            self.current_epoch = epoch + self.config.stage_a_epochs + self.config.stage_b_epochs
            print(f"Stage C - Epoch {epoch + 1}/{self.config.stage_c_epochs}")
            print("  目標：統一四種中文任務模式")
            print("  資料：混合所有中文資料 + 多口音（zh-TW/多地口音）")
            print("  方法：只對 <assistant> 段計算 loss + 全雙工打斷分類")
            
            self._train_epoch(
                allowed_modes=stage_c_modes,
                stage="C"
            )
            
            if self.eval_dataset:
                self._evaluate(stage="C")
    
    def _train_epoch(self, allowed_modes: List[str], stage: str):
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Training Stage {stage}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 過濾批次（只使用允許的模式）
            filtered_batch = self._filter_batch_by_modes(batch, allowed_modes)
            if not filtered_batch:
                continue
            
            # 移動到設備
            filtered_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in filtered_batch.items()}
            
            # 前向傳播
            loss = self._forward_step(filtered_batch, stage)
            
            # 反向傳播
            loss = loss / self.config.gradient_accumulation_steps
            
            if self.config.fp16 and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累積
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.fp16 and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # 更新進度條
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'step': self.global_step
            })
            
            # 記錄和保存
            if self.global_step % self.config.logging_steps == 0:
                self._log_metrics({
                    f"train/loss_stage_{stage}": loss.item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/global_step": self.global_step
                })
            
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Stage {stage} - Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _forward_step(self, batch: Dict, stage: str) -> torch.Tensor:
        """前向傳播步驟"""
        # 根據階段調整損失計算
        if stage == "A":
            # Stage A: 主要關注輸入對齊
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                mode=batch["modes"][0]  # 假設批次中所有樣本模式相同
            )
            
            # 只計算文字損失和對齊損失
            loss = outputs["loss"]
            
        elif stage == "B":
            # Stage B: 主要關注語音輸出
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                mode=batch["modes"][0]
            )
            
            # 重點關注 RVQ 損失
            loss = outputs.get("rvq_loss", outputs["loss"])
            
        else:  # Stage C
            # Stage C: 多任務聯合訓練
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                mode=batch["modes"][0]
            )
            
            # 使用完整損失
            loss = outputs["loss"]
        
        return loss
    
    def _filter_batch_by_modes(self, batch: Dict, allowed_modes: List[str]) -> Optional[Dict]:
        """根據允許的模式過濾批次"""
        # 找到批次中符合條件的樣本
        valid_indices = []
        for i, mode in enumerate(batch["modes"]):
            if mode in allowed_modes:
                valid_indices.append(i)
        
        if not valid_indices:
            return None
        
        # 創建過濾後的批次
        filtered_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                filtered_batch[key] = value[valid_indices]
            elif isinstance(value, list):
                filtered_batch[key] = [value[i] for i in valid_indices]
            else:
                filtered_batch[key] = value
        
        return filtered_batch
    
    def _freeze_components(self, components: List[str]):
        """凍結指定組件"""
        for component_name in components:
            if hasattr(self.model, component_name):
                component = getattr(self.model, component_name)
                for param in component.parameters():
                    param.requires_grad = False
                print(f"凍結組件: {component_name}")
    
    def _unfreeze_components(self, components: List[str]):
        """解凍指定組件"""
        for component_name in components:
            if hasattr(self.model, component_name):
                component = getattr(self.model, component_name)
                for param in component.parameters():
                    param.requires_grad = True
                print(f"解凍組件: {component_name}")
    
    def _evaluate(self, stage: str) -> Dict[str, float]:
        """評估模型"""
        if not self.eval_dataset:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc=f"Evaluating Stage {stage}"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    mode=batch["modes"][0]
                )
                
                loss = outputs["loss"]
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        metrics = {
            f"eval/loss_stage_{stage}": avg_loss,
            "eval/global_step": self.global_step
        }
        
        self._log_metrics(metrics)
        
        # 保存最佳模型
        if avg_loss < self.best_eval_loss:
            self.best_eval_loss = avg_loss
            self.save_model("best_model")
        
        print(f"Stage {stage} - Eval Loss: {avg_loss:.4f}")
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """記錄指標"""
        if self.config.use_wandb:
            wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self):
        """保存檢查點"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": asdict(self.config)
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        torch.save(checkpoint, os.path.join(checkpoint_path, "pytorch_model.bin"))
        
        # 保存配置
        with open(os.path.join(checkpoint_path, "training_config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        print(f"檢查點已保存到: {checkpoint_path}")
    
    def save_model(self, model_name: str):
        """保存模型"""
        model_path = os.path.join(self.config.output_dir, model_name)
        self.model.save_pretrained(model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """載入檢查點"""
        checkpoint = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), 
                               map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        print(f"檢查點已載入: {checkpoint_path}")


if __name__ == "__main__":
    # 測試訓練器
    print("測試 SpeechLLM 訓練器")
    
    # 創建模型配置
    model_config = SpeechLLMConfig(
        llm_model_name="Qwen/Qwen2.5-7B-Instruct",
        use_lora=True,
        freeze_llm=True
    )
    
    # 創建訓練配置
    training_config = TrainingConfig(
        output_dir="./test_outputs",
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=2,
        stage_a_epochs=1,
        stage_b_epochs=0,
        stage_c_epochs=0,
        use_wandb=False
    )
    
    print("配置創建完成")
    print(f"模型配置: {model_config}")
    print(f"訓練配置: {training_config}")
    
    # 注意：實際使用時需要真實的資料集
    print("訓練器測試完成（需要真實資料集才能運行完整測試）")
