# ============= train.py ==================
# init command: torchrun --nproc_per_node=1 train.py
import logging, wandb, torch
import torch.distributed as dist, torch.nn.functional as F, torch.nn as nn
import pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader

from types import SimpleNamespace
from tqdm import tqdm
from pathlib import Path

from models import getModels
from losses import DistillationLoss
from datasets import load_dataset
from schedulers import CosineAnnealingWarmUpRestarts
from utils import count_parameters, AverageMeter, TqdmLoggingHandler

def run_epoch(model, optimizer, criterion, epoch, mode, **kwargs):
    loss_sum = acc = size = 0
    loader = kwargs['train_loader' if mode == 'train' else 'valid_loader']
    if len(loader) == 0:
        return
    log_interval = len(loader) // 8
    for data in tqdm(loader):
        x, y = data
        x = x.to(local_rank)
        y = y.to(local_rank)

        optimizer.zero_grad()
        out = model(x)
        # if config.mode == 'pretrain': # for NLP pretrain
        #     y = model.emb(y)
        loss = criterion(x, out, y)
        if mode == 'train':
            loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if config.mode == 'finetune':
            acc += torch.count_nonzero(out.argmax(axis=-1) == y).item()
        size += len(x)
        if local_rank == 0 and step.update() % log_interval == 0:
            wandb.log({'loss': loss}, step=step.count)
        if mode == 'valid' and step.count % 8 == 0:
            break
            
    loss_mean = loss_sum / len(loader)
    logging.info(f"Loss: {loss_mean}, {mode}_acc: {acc / size}")
    if local_rank == 0:
        wandb.log({f'{mode}_loss': loss_mean, f'{mode}_acc': acc / size,
            'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch}, step=step.count)

def train(config):
    Path(f"checkpoints/{config.name}").mkdir(exist_ok=True)
    logging.info(f"[ {config.mode} begin. ]")
    teacher_model, model = getModels(config.mode)
    teacher_model.to(local_rank)
    teacher_model.load_state_dict(torch.load('checkpoints/vit.pkl'))
    teacher_model.eval()
    model.load_state_dict(torch.load(config.load_from)) if config.load_from else None
    model.head.fc = nn.Linear(1280, 1000 if config.mode == 'pretrain' else 102)
    model.to(local_rank)
    if local_rank == 0:
        wandb.init(project="Deepest", name=f"{config.name}_{config.mode}")
        wandb.watch(model)
        wandb.config.update(config)
    optimizer = config.optimizer(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1)
    scheduler = config.scheduler(optimizer, config.T0, T_up = 1, eta_max=config.eta_max)
    # scheduler = config.scheduler(optimizer, config.step_size)
    criterion = config.criterion
    criterion = DistillationLoss(criterion, teacher_model, 'soft', config.alpha, 1.0)

    train_dataset, valid_dataset, test_dataset = load_dataset(config.mode)
    logging.info(f'Loading data is finished! train: {len(train_dataset)}, valid: {len(valid_dataset)}')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, shuffle=False, batch_size=config.batch_size // world_size, num_workers=2)
    valid_loader = DataLoader(dataset=valid_dataset, sampler=valid_sampler, shuffle=False, batch_size=config.batch_size // world_size, num_workers=2)
    for epoch in range(1, 1 + config.max_epoch):
        logging.info(f"Epoch {epoch}")
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)

        model.train()
        run_epoch(mode='train', **locals())

        model.eval()
        with torch.no_grad():
            run_epoch(mode='valid', **locals())
        
        scheduler.step()
        if local_rank == 0:
            torch.save(model.state_dict(), f'checkpoints/{config.name}/{config.mode}_{epoch}.pkl')

    if test_dataset is not None:
        logging.info(f'Test begin! test: {len(test_dataset)}')
        valid_dataset = test_dataset
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_loader = DataLoader(dataset=valid_dataset, sampler=valid_sampler, shuffle=False, batch_size=config.batch_size // world_size, num_workers=world_size << 2)
        model.eval()
        with torch.no_grad():
            run_epoch(mode='test', **locals())
    if local_rank == 0:
        torch.save(model.state_dict(), f'checkpoints/{config.name}/{config.mode}.pkl')
        wandb.finish()
    dist.barrier()

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank == 0 else logging.WARNING,
        handlers=[TqdmLoggingHandler()])
    logging.info(f"Training begin. world_size: {world_size}")

    config = SimpleNamespace()
    config.name = "rexnet_100_soft"
    # config.max_epoch = 100
    # config.batch_size = 64
    # config.lr = 1e-1
    # config.optimizer = torch.optim.SGD

    # config.step_size = 30
    # config.scheduler = torch.optim.lr_scheduler.StepLR
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
    step = AverageMeter()
    # train(config)
    
    # config.max_epoch = 100
    # config.batch_size = 64
    # config.lr = 1e-1
    # config.optimizer = torch.optim.SGD

    # config.step_size = 30
    # config.scheduler = torch.optim.lr_scheduler.StepLR
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
    config.mode = ['pretrain', 'finetune'][1]
    config.load_from = "checkpoints/rexnet_100/pretrain.pkl"
    config.load_from = ""
    step = AverageMeter()
    train(config)