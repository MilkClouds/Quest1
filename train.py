# ============= train.py ==================
# init command: torchrun --nproc_per_node=1 train.py
import logging, wandb, torch
from sched import scheduler
import secrets
import torch.distributed as dist, torch.nn.functional as F, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from types import SimpleNamespace
from tqdm import tqdm

from models import getModel
from datasets import load_dataset
from schedulers import CosineAnnealingWarmUpRestarts
from utils import count_parameters, AverageMeter, TqdmLoggingHandler

def run_epoch(model, optimizer, criterion, train_loader, valid_loader, epoch, mode, **kwargs):
    loss_sum = acc = size = 0
    loader = (train_loader if mode == 'train' else valid_loader)
    if len(loader) == 0:
        return
    log_interval = len(loader) // 8
    for data in tqdm(loader):
        x, y = data
        x = x.to(local_rank)
        y = y.to(local_rank)

        optimizer.zero_grad()
        out = model(x)
        if config.mode == 'pretrain':
            y = model.emb(y)
        loss = criterion(out, y)
        if mode == 'train':
            loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if config.mode == 'finetune':
            acc += torch.count_nonzero(out.argmax(axis=-1) == y).item()
        size += len(x)
        if local_rank == 0 and step.update() % log_interval == 0:
            wandb.log({'loss': loss}, step=step.count)
            
    loss_mean = loss_sum / len(loader)
    logging.info(f"Loss: {loss_mean}, {mode}_acc: {acc / size}")
    if local_rank == 0:
        wandb.log({f'{mode}_loss': loss_mean, f'{mode}_acc': acc / size,
            'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch}, step=step.count)

def train(config):
    logging.info(f"[ {config.mode} begin. ]")
    model = getModel()
    model.to(local_rank)
    if local_rank == 0:
        wandb.init(project="Deepest", name=config.mode)
        wandb.watch(model)
        wandb.config.update(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, config.T0, eta_max=config.eta_max)
    criterion = config.criterion
    train_dataset, valid_dataset, test_dataset = load_dataset(config.mode)
    logging.info(f'Loading data is finished! train: {len(train_dataset)}, valid: {len(valid_dataset)}')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, shuffle=False, batch_size=config.batch_size // world_size)
    valid_loader = DataLoader(dataset=valid_dataset, sampler=valid_sampler, shuffle=False, batch_size=config.batch_size // world_size)
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
            torch.save(model.state_dict(), f'checkpoints/{config.mode}_{epoch}.pkl')

    if test_dataset is not None:
        logging.info(f'Test begin! test: {len(test_dataset)}')
        valid_dataset = test_dataset
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_loader = DataLoader(dataset=valid_dataset, sampler=valid_sampler, shuffle=False, batch_size=config.batch_size // world_size)
        model.eval()
        with torch.no_grad():
            run_epoch(mode='valid', **locals())
    if local_rank == 0:
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
    config.max_epoch = 10
    config.batch_size = 64
    config.lr = 1e-5
    config.T0 = config.max_epoch
    config.eta_max = 5e-3
    config.weight_decay = 0
    # config.step_size = 3
    config.criterion = nn.CrossEntropyLoss()
    config.mode = ['pretrain', 'finetune'][1]
    step = AverageMeter()
    train(config)