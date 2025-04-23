from typing import Dict, Tuple

from loguru import logger
import numpy as np
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from einops import rearrange

from c4a0_rust import N_COLS, N_ROWS  # type: ignore


class ModelConfig(BaseModel):
    """Configuration for ConnectFourNet."""
    n_residual_blocks: int
    conv_filter_size: int
    n_policy_layers: int
    n_value_layers: int
    lr_schedule: Dict[int, float]
    l2_reg: float
    label_smoothing: float = 0.1


class ConnectFourNet(pl.LightningModule):
    EPS = 1e-8

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        
        # Convolutional backbone
        self.conv = nn.Sequential(
            nn.Conv2d(2, config.conv_filter_size, 3, padding=1),
            nn.BatchNorm2d(config.conv_filter_size),
            nn.Mish(),
            *[ResidualBlock(config.conv_filter_size) 
              for _ in range(config.n_residual_blocks)]
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Policy head components
        self.policy_se = SqueezeExcitation(config.conv_filter_size)
        self.policy_head = nn.Sequential(
            *[ResidualFC(config.conv_filter_size) 
              for _ in range(config.n_policy_layers-1)],
            PolicyTemperatureScaling(),
            nn.Linear(config.conv_filter_size, N_COLS),
            nn.LogSoftmax(dim=1)
        )
        
        # Value head components
        self.value_head = DuelingHead(config.conv_filter_size)
        self.value_processor = nn.Sequential(
            nn.Linear(config.conv_filter_size*2, 128),
            nn.Mish(),
            nn.Linear(128, 2),
            nn.Tanh()
        )

        # Metrics
        self.policy_kl_div = torchmetrics.KLDivergence(log_prob=True)
        self.value_mse = torchmetrics.MeanSquaredError()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Feature extraction
        x = self.conv(x)
        pooled = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        # Policy calculation
        policy_features = self.policy_se(pooled)
        policy_logprobs = self.policy_head(policy_features)
        
        # Value calculation
        value_features = self.value_head(pooled)
        q_values = self.value_processor(value_features)
        
        return policy_logprobs, q_values[:, 0], q_values[:, 1]

    def forward_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            pos = torch.from_numpy(x).to(self.device)
            policy, q_penalty, q_no_penalty = self.forward(pos)
        return (
            policy.cpu().numpy(),
            q_penalty.cpu().numpy(),
            q_no_penalty.cpu().numpy(),
        )

    def configure_optimizers(self):
        gen_n: int = self.trainer.gen_n  # type: ignore
        schedule = sorted(self.hparams.lr_schedule.items())
        _, lr = next(iter(schedule))
        for threshold, rate in schedule:
            if gen_n >= threshold:
                lr = rate
        return torch.optim.AdamW(
            self.parameters(), 
            lr=lr,
            weight_decay=self.hparams.l2_reg
        )

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def _shared_step(self, batch, prefix):
        pos, policy_target, q_penalty_target, q_no_penalty_target = batch
        
        # Data augmentation
        if prefix == "train":
            flip_mask = torch.rand(pos.size(0)) < 0.5
            pos[flip_mask] = torch.flip(pos[flip_mask], dims=[3])
            policy_target[flip_mask] = torch.flip(policy_target[flip_mask], dims=[1])
        
        # Forward pass
        policy_logprob, q_penalty_pred, q_no_penalty_pred = self(pos)
        
        # Label smoothing
        policy_target = (1 - self.hparams.label_smoothing) * policy_target + \
                      self.hparams.label_smoothing / N_COLS
        
        # Loss calculation
        policy_loss = self.policy_kl_div(
            torch.log(policy_target + self.EPS), 
            policy_logprob
        ) * 2.0
        
        value_loss = self.value_mse(
            torch.cat([q_penalty_pred, q_no_penalty_pred], dim=0),
            torch.cat([q_penalty_target, q_no_penalty_target], dim=0)
        ) * 0.5
        
        total_loss = policy_loss + value_loss
        
        # Logging
        self.log_dict({
            f"{prefix}_loss": total_loss,
            f"{prefix}_policy_loss": policy_loss,
            f"{prefix}_value_loss": value_loss
        }, prog_bar=True)
        
        return total_loss


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.Mish(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.mish = nn.Mish()

    def forward(self, x):
        return self.mish(x + self.block(x))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.Mish(),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class ResidualFC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class PolicyTemperatureScaling(nn.Module):
    def __init__(self, init_temp=0.7):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor([init_temp]))

    def forward(self, logits):
        return logits / self.temp.clamp(min=0.1, max=2.0)


class DuelingHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.value_stream = nn.Linear(dim, dim)
        self.advantage_stream = nn.Linear(dim, dim)

    def forward(self, x):
        v = self.value_stream(x)
        a = self.advantage_stream(x)
        return torch.cat([v + (a - a.mean()), v - (a - a.mean())], dim=1)