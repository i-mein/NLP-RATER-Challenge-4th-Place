from transformers import get_cosine_schedule_with_warmup
from transformers import AdamW, Adafactor
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import SequentialSampler, DataLoader
import torch
from torch import optim
import importlib


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_model(cfg, ds):
    Net = importlib.import_module(cfg.model).Net
    net = Net(cfg)
    if cfg.pretrained_weights is not None:
        print(f"{cfg.local_rank}: loading weights from", cfg.pretrained_weights)
        state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
        if "model" in state_dict.keys():
            state_dict = state_dict["model"]
        state_dict = {key.replace("module.", ""): val for key, val in state_dict.items()}
        if cfg.pop_weights is not None:
            print(f"popping {cfg.pop_weights}")
            to_pop = []
            for key in state_dict:
                for item in cfg.pop_weights:
                    if item in key:
                        to_pop += [key]
            for key in to_pop:
                print(f"popping {key}")
                state_dict.pop(key)

        net.load_state_dict(state_dict, strict=cfg.pretrained_weights_strict)
        print(f"{cfg.local_rank}: weights loaded from", cfg.pretrained_weights)

    return net


def create_checkpoint(cfg, model, optimizer, epoch, scheduler=None, scaler=None):

    state_dict = model.state_dict()
    if cfg.save_weights_only:
        checkpoint = {"model": state_dict}
        return checkpoint

    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


def get_dataset(df, cfg, mode="train"):

    # modes train, val, index
    print(f"Loading {mode} dataset")

    if mode == "train":
        dataset = get_train_dataset(df, cfg)
    #     elif mode == 'train_val':
    #         dataset = get_val_dataset(df, cfg)
    elif mode == "val":
        dataset = get_val_dataset(df, cfg)
    elif mode == "test":
        dataset = get_test_dataset(df, cfg)
    else:
        pass
    return dataset


def get_dataloader(ds, cfg, mode="train"):

    if mode == "train":
        dl = get_train_dataloader(ds, cfg)
    elif mode == "val":
        dl = get_val_dataloader(ds, cfg)
    elif mode == "test":
        dl = get_test_dataloader(ds, cfg)
    return dl


def get_train_dataset(train_df, cfg):
    print("Loading train dataset")

    train_dataset = cfg.CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    if cfg.data_sample > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(cfg.data_sample))
    return train_dataset


def get_train_dataloader(train_ds, cfg):

    sampler = None

    train_dataloader = DataLoader(
        train_ds,
        sampler=sampler,
        shuffle=(sampler is None),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataset(val_df, cfg, allowed_targets=None):
    print("Loading val dataset")
    val_dataset = cfg.CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")
    return val_dataset


def get_val_dataloader(val_ds, cfg):

    sampler = SequentialSampler(val_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"valid: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_test_dataset(test_df, cfg):
    print("Loading test dataset")
    test_dataset = cfg.CustomDataset(test_df, cfg, aug=cfg.val_aug, mode="test")
    return test_dataset


def get_test_dataloader(test_ds, cfg):

    sampler = SequentialSampler(test_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    test_dataloader = DataLoader(
        test_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=cfg.val_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"test: dataset {len(test_ds)}, dataloader {len(test_dataloader)}")
    return test_dataloader


def get_optimizer(model, cfg):

    params = model.parameters()

    if cfg.optimizer == "Adam":
        optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "AdamW_mixed":
        params = [
            {
                "params": [
                    param for name, param in model.named_parameters() if "backbone" in name
                ],
                "lr": cfg.lr[0],
            },
            {
                "params": [
                    param for name, param in model.named_parameters() if not "backbone" in name
                ],
                "lr": cfg.lr[1],
            },
        ]
        optimizer = AdamW(params, lr=cfg.lr[1], weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "AdamW":
        optimizer = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "Adafactor":
        optimizer = Adafactor(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            scale_parameter=False,
            relative_step=False,
        )
    elif cfg.optimizer == "Adafactor_mixed":
        params = [
            {
                "params": [
                    param for name, param in model.named_parameters() if "backbone" in name
                ],
                "lr": cfg.lr[0],
            },
            {
                "params": [
                    param for name, param in model.named_parameters() if not "backbone" in name
                ],
                "lr": cfg.lr[1],
            },
        ]
        optimizer = Adafactor(
            params,
            lr=cfg.lr[1],
            weight_decay=cfg.weight_decay,
            scale_parameter=False,
            relative_step=False,
        )

    return optimizer


def get_scheduler(cfg, optimizer, total_steps):

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
        num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
    )

    return scheduler


def get_data(cfg):

    # setup dataset

    print(f"reading {cfg.train_df}")
    df = pd.read_csv(cfg.train_df)

    if cfg.test:
        test_df = pd.read_csv(cfg.test_df)
    else:
        test_df = None

    if cfg.fold == -1:
        val_df = df[df["fold"] == 0]
    else:
        val_df = df[df["fold"] == cfg.fold]

    train_df = df[df["fold"] != cfg.fold]

    return train_df, val_df, test_df
