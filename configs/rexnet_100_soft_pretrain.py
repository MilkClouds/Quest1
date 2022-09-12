from types import SimpleNamespace
import torch, torch.nn as nn
from schedulers import CosineAnnealingWarmUpRestarts
config = SimpleNamespace()
config.name = "rexnet_100_soft"
config.max_epoch = 20
config.batch_size = 64
config.lr = 1e-9
config.optimizer = torch.optim.AdamW
config.scheduler = CosineAnnealingWarmUpRestarts
config.T0 = 20
config.eta_max = 5e-3
config.weight_decay = 1e-4
config.smoothing = 0.1
config.alpha = 0.5
config.criterion = nn.CrossEntropyLoss(label_smoothing=config.smoothing)
config.mode = ['pretrain', 'finetune'][0]
config.load_from = ""