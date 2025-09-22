"""
優化器和學習率調度器
為 SpeechLLM 提供優化策略
"""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LinearLR, 
    SequentialLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau
)
from typing import Dict, List, Optional, Tuple, Union
import math


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    optimizer_type: str = "adamw",
    scheduler_type: str = "cosine_with_warmup",
    **kwargs
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    創建優化器和學習率調度器
    
    Args:
        model: 模型
        learning_rate: 學習率
        weight_decay: 權重衰減
        warmup_steps: 預熱步數
        total_steps: 總步數
        optimizer_type: 優化器類型
        scheduler_type: 調度器類型
        
    Returns:
        optimizer: 優化器
        scheduler: 學習率調度器
    """
    # 創建優化器
    optimizer = create_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        **kwargs
    )
    
    # 創建調度器
    scheduler = create_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        scheduler_type=scheduler_type,
        **kwargs
    )
    
    return optimizer, scheduler


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    momentum: float = 0.9,
    **kwargs
) -> torch.optim.Optimizer:
    """
    創建優化器
    
    Args:
        model: 模型
        learning_rate: 學習率
        weight_decay: 權重衰減
        optimizer_type: 優化器類型 (adamw, adam, sgd)
        beta1: Adam beta1 參數
        beta2: Adam beta2 參數
        eps: Adam epsilon 參數
        momentum: SGD 動量參數
        
    Returns:
        optimizer: 優化器
    """
    # 獲取參數組（可以為不同組件設置不同學習率）
    param_groups = get_parameter_groups(model, weight_decay)
    
    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "adam":
        optimizer = Adam(
            param_groups,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = SGD(
            param_groups,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"不支援的優化器類型: {optimizer_type}")
    
    return optimizer


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    layerwise_lr_decay: Optional[float] = None
) -> List[Dict]:
    """
    獲取參數組，可以為不同組件設置不同的學習率和權重衰減
    
    Args:
        model: 模型
        weight_decay: 默認權重衰減
        layerwise_lr_decay: 層級學習率衰減（可選）
        
    Returns:
        param_groups: 參數組列表
    """
    param_groups = []
    
    # 不進行權重衰減的參數類型
    no_decay_params = ["bias", "LayerNorm", "layer_norm", "ln"]
    
    # 為不同組件設置不同學習率
    component_lr_scales = {
        "llm": 0.1,  # LLM 使用較小學習率
        "whisper_encoder": 0.5,  # Whisper encoder 中等學習率
        "qformer": 1.0,  # Q-Former 正常學習率
        "audio_transformer": 1.0,  # 音訊 Transformer 正常學習率
    }
    
    for component_name, component in model.named_children():
        lr_scale = component_lr_scales.get(component_name, 1.0)
        
        # 分離需要和不需要權重衰減的參數
        decay_params = []
        no_decay_params_list = []
        
        for name, param in component.named_parameters():
            if not param.requires_grad:
                continue
                
            if any(nd in name for nd in no_decay_params):
                no_decay_params_list.append(param)
            else:
                decay_params.append(param)
        
        # 添加參數組
        if decay_params:
            param_groups.append({
                "params": decay_params,
                "weight_decay": weight_decay,
                "lr_scale": lr_scale,
                "component": component_name
            })
        
        if no_decay_params_list:
            param_groups.append({
                "params": no_decay_params_list,
                "weight_decay": 0.0,
                "lr_scale": lr_scale,
                "component": component_name
            })
    
    return param_groups


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    scheduler_type: str = "cosine_with_warmup",
    min_lr_ratio: float = 0.1,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    創建學習率調度器
    
    Args:
        optimizer: 優化器
        warmup_steps: 預熱步數
        total_steps: 總步數
        scheduler_type: 調度器類型
        min_lr_ratio: 最小學習率比例
        
    Returns:
        scheduler: 學習率調度器
    """
    if scheduler_type == "cosine_with_warmup":
        # 帶預熱的餘弦退火
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr_ratio * optimizer.defaults['lr']
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
    elif scheduler_type == "linear_with_warmup":
        # 帶預熱的線性衰減
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        linear_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=total_steps - warmup_steps
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, linear_scheduler],
            milestones=[warmup_steps]
        )
        
    elif scheduler_type == "cosine_restarts":
        # 餘弦退火重啟
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_steps // 4,  # 第一次重啟週期
            T_mult=2,  # 週期倍增因子
            eta_min=min_lr_ratio * optimizer.defaults['lr']
        )
        
    elif scheduler_type == "plateau":
        # 基於驗證損失的調度器
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=min_lr_ratio * optimizer.defaults['lr']
        )
        
    elif scheduler_type == "constant":
        # 常數學習率（僅預熱）
        scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
    else:
        raise ValueError(f"不支援的調度器類型: {scheduler_type}")
    
    return scheduler


class LayerwiseLRDecayScheduler:
    """
    層級學習率衰減調度器
    為不同深度的層設置不同的學習率
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        decay_rate: float = 0.8,
        num_layers: int = 12
    ):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.num_layers = num_layers
        
        # 為每個參數組設置層級學習率
        self._setup_layerwise_lr()
    
    def _setup_layerwise_lr(self):
        """設置層級學習率"""
        for group in self.optimizer.param_groups:
            if 'layer_id' in group:
                layer_id = group['layer_id']
                # 越深的層學習率越小
                lr_scale = self.decay_rate ** (self.num_layers - layer_id - 1)
                group['lr'] *= lr_scale
    
    def step(self):
        """更新學習率（如果需要）"""
        pass


class AdaptiveLearningRateScheduler:
    """
    自適應學習率調度器
    根據損失變化動態調整學習率
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        patience: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-7,
        threshold: float = 1e-4
    ):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.last_epoch = 0
    
    def step(self, loss: float):
        """
        根據損失更新學習率
        
        Args:
            loss: 當前損失值
        """
        if loss < self.best_loss - self.threshold:
            self.best_loss = loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            # 降低學習率
            for group in self.optimizer.param_groups:
                old_lr = group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                group['lr'] = new_lr
                print(f"學習率從 {old_lr:.2e} 降低到 {new_lr:.2e}")
            
            self.num_bad_epochs = 0
        
        self.last_epoch += 1


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """獲取當前學習率"""
    return optimizer.param_groups[0]['lr']


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    """設置學習率"""
    for group in optimizer.param_groups:
        group['lr'] = lr


if __name__ == "__main__":
    # 測試優化器和調度器
    print("測試優化器和調度器")
    
    # 創建簡單模型
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    )
    
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試優化器創建
    optimizer = create_optimizer(
        model=model,
        learning_rate=1e-3,
        weight_decay=0.01,
        optimizer_type="adamw"
    )
    
    print(f"優化器類型: {type(optimizer).__name__}")
    print(f"參數組數量: {len(optimizer.param_groups)}")
    
    # 測試調度器創建
    scheduler = create_scheduler(
        optimizer=optimizer,
        warmup_steps=100,
        total_steps=1000,
        scheduler_type="cosine_with_warmup"
    )
    
    print(f"調度器類型: {type(scheduler).__name__}")
    
    # 測試學習率變化
    print("\n學習率變化測試:")
    initial_lr = get_learning_rate(optimizer)
    print(f"初始學習率: {initial_lr:.2e}")
    
    # 模擬訓練步驟
    lrs = []
    for step in range(200):
        scheduler.step()
        current_lr = get_learning_rate(optimizer)
        if step % 50 == 0:
            print(f"步驟 {step}: 學習率 = {current_lr:.2e}")
        lrs.append(current_lr)
    
    print(f"最終學習率: {lrs[-1]:.2e}")
    
    # 測試自適應調度器
    print("\n測試自適應調度器:")
    adaptive_scheduler = AdaptiveLearningRateScheduler(optimizer)
    
    # 模擬損失變化
    losses = [1.0, 0.8, 0.7, 0.69, 0.68, 0.67, 0.67, 0.67]  # 損失停滯
    
    for epoch, loss in enumerate(losses):
        print(f"Epoch {epoch}: Loss = {loss:.3f}, LR = {get_learning_rate(optimizer):.2e}")
        adaptive_scheduler.step(loss)
    
    print("\n優化器和調度器測試完成！")
