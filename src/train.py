#!/usr/bin/env python
# coding: utf-8

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import argparse
import numpy as np
import pandas as pd
import importlib
import sys, gc, re
import random
from tqdm import tqdm
import argparse
from sklearn.metrics import log_loss, mean_squared_error
import yaml
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset 
from torch.utils.data import SequentialSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import tokenizers

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#########
os.environ["WANDB_PROJECT"] = "Feedback3"
os.environ["WANDB_ENTITY"] = "ime"
os.environ["WANDB_SILENT"] = "false"
os.environ["WANDB_MODE"] = "online"

print("Torch version", torch.__version__)
print("Transformers version", transformers.__version__)
print(f"tokenizers.__version__: {tokenizers.__version__}")
# print("PLight version", pl.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gc.enable()

## =========================
# from dataset import *
from dataset.dataset import collate
from utils.utils import *

## =========================

def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

# Turns a dictionary into a class
class dict2class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



def get_train_dataloader(train_ds, cfg):
    train_dataloader = DataLoader(
        train_ds,
        sampler=None,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
        collate_fn=cfg.CustomDataset.get_train_collate_fn(cfg),
        drop_last=True, #cfg.training.drop_last_batch,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}, bs: {cfg.training.batch_size}")
    return train_dataloader


def get_val_dataloader(val_ds, cfg):
    val_dataloader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=cfg.training.validation_batch_size, #1,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        collate_fn=cfg.CustomDataset.get_validation_collate_fn(cfg),
        worker_init_fn=worker_init_fn,
    )
    print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}, bs: {cfg.training.validation_batch_size}")
    return val_dataloader


def get_scheduler(cfg, optimizer, total_steps):
    if cfg.scheduler == "Linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.warmup_epochs * (total_steps // cfg.training.batch_size)
            ),
            num_training_steps=cfg.training.epochs
            * (total_steps // cfg.training.batch_size),
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.warmup_epochs * (total_steps // cfg.training.batch_size)
            ),
            num_training_steps=cfg.training.epochs
            * (total_steps // cfg.training.batch_size),
        )
    return scheduler


def load_checkpoint(cfg, model):
    d = torch.load(cfg.architecture.pretrained_weights, map_location="cpu")

    if "model" in d:
        model_weights = d["model"]
    else:
        model_weights = d

    if (
        model.backbone.embeddings.word_embeddings.weight.shape[0]
        < model_weights["backbone.embeddings.word_embeddings.weight"].shape[0]
    ):
        print("resizing pretrained embedding weights")
        model_weights["backbone.embeddings.word_embeddings.weight"] = model_weights[
            "backbone.embeddings.word_embeddings.weight"
        ][: model.backbone.embeddings.word_embeddings.weight.shape[0]]

    try:
        model.load_state_dict(model_weights, strict=True)
    except Exception as e:
        print("removing unused pretrained layers")
        for layer_name in re.findall("size mismatch for (.*?):", str(e)):
            model_weights.pop(layer_name, None)
        model.load_state_dict(model_weights, strict=False)

    print(f"Weights loaded from: {cfg.architecture.pretrained_weights}")



def get_model(cfg):
    Net = importlib.import_module(cfg.model_class).Net
    return Net(cfg)




def run_fold(cfg, fold=0):

    print("#" * 20)
    print(f"### Fold: {fold} training..")
    print("#" * 20)

    ## ========================================================
    ##### Dataset
    ## ========================================================

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if fold==0 or fold == -1: tokenizer.save_pretrained(OUTPUT+'/tokenizer/')
    cfg.tokenizer = tokenizer

    cfg.CustomDataset = importlib.import_module('dataset').TrainDataset #

    if cfg.train_dataframe.endswith(".pq"):
            train_df = pd.read_parquet(cfg.train_dataframe)
    else:
        train_df = pd.read_csv(cfg.train_dataframe)

    if "kfold" in train_df.columns:
        if cfg.fold == -1:
            val_df = train_df.sample(100).copy()
            #val_df = train_df[train_df.kfold == 0].copy()
        else:
            val_df = train_df[train_df.kfold == cfg.fold].copy()
            train_df = train_df[train_df.kfold != cfg.fold].copy()
    else:
        val_df = train_df.copy()

    train_dataset = cfg.CustomDataset(cfg, train_df)  # mode="train"
    val_dataset   = cfg.CustomDataset(cfg, val_df)    # mode="val

    cfg.train_dataset = train_dataset

    train_dataloader = get_train_dataloader(train_dataset, cfg)
    val_dataloader   = get_val_dataloader(val_dataset, cfg)

    print('input example:', train_dataset[0][0])
    print('label example:', train_dataset[0][1])

    valid_folds = val_df.copy()

    ## ========================================================
    #####  Model
    ## ========================================================

    model = get_model(cfg)
    model.to(device)

    if cfg.resume_pretrained_weights != "":
        try:
            load_checkpoint(cfg, model)
        except:
            print("WARNING: could not load checkpoint")
    ##-----------------------------

    ### loss 
    if cfg.criterion=="L1_smooth":
        criterion = nn.SmoothL1Loss(reduction='mean') # 
    elif cfg.criterion=="rmse":
        criterion = RMSELoss(reduction="mean")
    else: #cfg.criterion=="mse":
        criterion = nn.MSELoss(reduction="mean")
    #criterion.to(device)

    ## ========================================================
    #####  optim/scheduler
    ## ========================================================

    total_steps = len(train_dataset)
    params = model.parameters()
    no_decay = ["bias", "LayerNorm.weight"]
    differential_layers = cfg.training.differential_learning_rate_layers

    if cfg.training.use_layer_lr_decay:
        optim_group_params = [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": 0,
            }]
    
        # initialize lrs for every layer
        num_layers = model.config.num_hidden_layers
        # layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
        layers = [model.model.embeddings] + list(model.model.encoder.layer)
        layers.reverse()
        lr = cfg.training.lr
        for layer in layers:
            lr *= cfg.training.layerwise_learning_rate_decay
            optim_group_params += [
                {
                    "params": [
                        p for n, p in layer.named_parameters() 
                        if (not any(layer in n for layer in differential_layers))
                        and (not any(nd in n for nd in no_decay))
                        # if not any(nd in n for nd in no_decay)
                        ],
                    "weight_decay": cfg.training.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [
                        p for n, p in layer.named_parameters() 
                        if (not any(layer in n for layer in differential_layers))
                        and (any(nd in n for nd in no_decay))
                        # if any(nd in n for nd in no_decay)
                        ],
                    "weight_decay": 0.0,
                    "lr": lr,
                }]
    else:
        optim_group_params = [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.lr,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (not any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.lr,
                "weight_decay": 0,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (not any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if (any(layer in name for layer in differential_layers))
                    and (any(nd in name for nd in no_decay))
                ],
                "lr": cfg.training.differential_learning_rate,
                "weight_decay": 0,
            }]   
    # -------------------------------------------------------------------------
    if cfg.optim=="AdamP": # 
        pass
        # optimizer = AdamP(optim_group_params,
        #     lr=cfg.training.lr,
        #     weight_decay=cfg.training.weight_decay,
        # )
    else: ## AdamW
        optimizer = torch.optim.AdamW(
            optim_group_params,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

    scheduler = get_scheduler(cfg, optimizer, total_steps)

    ## ========================================================
    #####  Train loop
    ## ========================================================

    if cfg.mixed_precision:
        scaler = GradScaler()

    cfg.curr_step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()

    for epoch in range(cfg.training.epochs):
        set_seed(cfg.seed + epoch)

        cfg.curr_epoch = epoch
        print("EPOCH:", epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)
        losses = []
        gc.collect()
        model.train()

        # ==== TRAIN LOOP
        for itr in progress_bar:
            i += 1
            cfg.curr_step += cfg.training.batch_size
            
            data, labels = next(tr_it)  # data = next(tr_it)
            data = collate(data)
            batch = cfg.CustomDataset.batch_to_device(data, device)
            labels = cfg.CustomDataset.batch_to_device(labels, device)

            if cfg.mixed_precision:
                with autocast():
                    # output_dict = model(batch)
                    output = model(batch)
            else:
                # output_dict = model(batch)
                output = model(batch)

            # loss = output_dict["loss"]
            loss = criterion(output, labels)
            losses.append(loss.item())

            if cfg.training.grad_accumulation != 1:
                loss /= cfg.training.grad_accumulation

            if cfg.mixed_precision:
                scaler.scale(loss).backward()
                if i % cfg.training.grad_accumulation == 0:
                    if cfg.training.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.gradient_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if i % cfg.training.grad_accumulation == 0:
                    if cfg.training.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.training.gradient_clip
                        )
                    optimizer.step()
                    optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            if cfg.curr_step % cfg.training.batch_size == 0:
                progress_bar.set_description(
                    f"lr: {np.round(optimizer.param_groups[0]['lr'],7)}, loss: {np.mean(losses[-10:]):.4f}"
                )

        del data, batch, labels
        torch.cuda.empty_cache()
        _ = gc.collect()

        # ==== VALID LOOP
        progress_bar = tqdm(range(len(val_dataloader)))
        val_it = iter(val_dataloader)

        val_losses = []
        preds = []
        probabilities = []
        all_targets = []
        targets = []
        
        model.eval()
        for itr in progress_bar:
            data, labels = next(val_it) # data = next(val_it)
            data = collate(data)
            batch = cfg.CustomDataset.batch_to_device(data, device)
            labels = cfg.CustomDataset.batch_to_device(labels, device)

            with torch.no_grad():
                if cfg.mixed_precision:
                    with autocast():
                        output = model(batch)# output_dict = model(batch)
                else:
                    output = model(batch)   # output_dict = model(batch)

            loss = criterion(output, labels)
            val_losses.append(loss.item())

            # if cfg.dataset == "feedback_dataset":
            #     for logits, word_start_mask, target in zip(
            #         output_dict["logits"],
            #         output_dict["word_start_mask"],
            #         output_dict["target"],
            #     ):
            #         probs = (
            #             torch.softmax(logits[word_start_mask], dim=1)
            #             .detach()
            #             .cpu()
            #             .numpy()
            #         )
            #         targets = target[word_start_mask].detach().cpu().numpy()

            #         for p, t in zip(probs, targets):
            #             probabilities.append(p)
            #             if cfg.training.is_pseudo:
            #                 all_targets.append(np.argmax(t))
            #             else:
            #                 all_targets.append(t)
            # else:# output_dict["logits"].float().softmax(dim=1).detach().cpu().numpy()
            preds.append(output.float().detach().cpu().numpy())
            targets.append(labels.float().detach().cpu().numpy())
                
        val_loss = np.mean(val_losses)        

        # if cfg.dataset == "feedback_dataset":
        #     metric = log_loss(
        #         all_targets,
        #         np.vstack(probabilities),
        #         eps=1e-7,
        #         labels=list(range(cfg.dataset.num_classes)),
        #     )
        # else:
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        if cfg.exp_name != "pretrain-type-v9":
            metric, _ = get_score(targets, preds)
            # metric = log_loss(val_df[cfg.dataset.label_columns].values.argmax(axis=1), preds)
        else:
            metric = None
        print("Validation MCRMSE metric", metric)

        # curr_score = val_loss 
        curr_score = metric
        ### TODO metric MCRMSE

        if curr_score < best_val_loss:
            best_val_loss = curr_score
            print(f'Epoch {epoch+1} - Save Best Score: {best_val_loss:.4f} Model')
            torch.save({
                'model': model.state_dict(),
                'predictions': preds
                }, OUTPUT+f"/{cfg.model_name.replace('/', '-')}_fold{fold}_best.pth")

    ##### full fit save
    if cfg.training.epochs > 0:
        checkpoint = {"model": model.state_dict()}
        torch.save(checkpoint, OUTPUT+f"/pretrain_model_ep{epoch}.pth")

    ##########
    predictions = torch.load(OUTPUT+f"/{cfg.model_name.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[[f"p_{c}" for c in cfg.target_cols]] = predictions
    valid_folds.to_csv(OUTPUT+f"/oof_fold{fold}_{metric:.4f}.csv", index=False)

    torch.cuda.empty_cache()
    gc.collect()




## ========================================================
#####  Main
## ========================================================

## Config
parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
args, _ = parser.parse_known_args(sys.argv)

# cfg = yaml.safe_load(open(parser_args.config).read())
with open(args.config, 'rb') as fo:
    cfg = yaml.safe_load(fo)

# cfg = dict2class(cfg)
for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)
print(cfg)

OUTPUT = os.path.join(cfg.output, cfg.exp_name)
os.makedirs(OUTPUT, exist_ok=True)

if __name__ == "__main__":
    cfg.device = device
    if cfg.seed < 0: cfg.seed = np.random.randint(1_000_000)
    set_seed(cfg.seed)

    if cfg.exp_name.startswith('pretrain'):
        run_fold(cfg, -1)
    else:
        for fold in range(cfg.n_folds):
            if fold in cfg.trn_fold:
                run_fold(cfg, fold)




