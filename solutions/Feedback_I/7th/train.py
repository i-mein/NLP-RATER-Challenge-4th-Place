import numpy as np
import importlib
import sys
from tqdm import tqdm
import gc
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict

from utils import (
    set_seed,
    get_model,
    create_checkpoint,
    get_data,
    get_dataset,
    get_dataloader,
)
from utils import (
    get_optimizer,
    get_scheduler,
)

from metrics import calc_metric
from copy import copy
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


sys.path.append("configs")
sys.path.append("models")
sys.path.append("data")
sys.path.append("postprocess")


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help="config filename")
parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)

# overwrite params in config with additional args
if len(other_args) > 1:
    other_args = {k.replace("-", ""): v for k, v in zip(other_args[1::2], other_args[2::2])}

    for key in other_args:
        if key in cfg.__dict__:

            print(f"overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}")
            cfg_type = type(cfg.__dict__[key])
            if cfg_type == bool:
                cfg.__dict__[key] = other_args[key] == "True"
            elif cfg_type == type(None):
                cfg.__dict__[key] = other_args[key]
            else:
                cfg.__dict__[key] = cfg_type(other_args[key])


os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)

cfg.CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
cfg.tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
cfg.val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device

post_process_pipeline = importlib.import_module(cfg.post_process_pipeline).post_process_pipeline


def run_eval(model, val_dataloader, cfg, pre="val", curr_epoch=0):
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    val_data = defaultdict(list)

    for ind_, data in enumerate(tqdm(val_dataloader, disable=cfg.local_rank != 0)):

        batch = batch_to_device(data, device)

        if cfg.mixed_precision:
            with autocast():
                output = model(batch)
        else:
            output = model(batch)

        for key, val in output.items():
            val_data[key] += [output[key]]

    for key, val in output.items():
        value = val_data[key]
        if isinstance(value[0], list):
            val_data[key] = [item for sublist in value for item in sublist]

        else:
            if len(value[0].shape) == 0:
                val_data[key] = torch.stack(value)
            else:
                val_data[key] = torch.cat(value, dim=0)

    if cfg.local_rank == 0:
        if cfg.save_val_data:
            torch.save(val_data, f"{cfg.output_dir}/fold{cfg.fold}/{pre}_data_seed{cfg.seed}.pth")

    loss_names = [key for key in output if "loss" in key]
    for k in loss_names:
        if cfg.local_rank == 0 and k in val_data:
            losses = val_data[k].cpu().numpy()
            loss = np.mean(losses)

            print(f"Mean {pre}_{k}", loss)

    if (
        (cfg.local_rank == 0)
        and (cfg.calc_metric)
        and (((curr_epoch + 1) % cfg.calc_metric_epochs) == 0)
    ):
        val_df = val_dataloader.dataset.df
        pp_out = post_process_pipeline(cfg, val_data, val_df)
        calc_metric(cfg, pp_out, val_df, pre)

    print("EVAL FINISHED")

    return 0


if __name__ == "__main__":

    # set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)

    train_dataloader = None
    train_val_dataloader = None
    val_dataloader = None
    test_dataloader = None

    cfg.local_rank = 0
    cfg.rank = 0  # global rank

    device = "cuda:%d" % cfg.gpu
    cfg.device = device

    set_seed(cfg.seed)

    train_df, val_df, test_df = get_data(cfg)

    train_dataset = get_dataset(train_df, cfg, mode="train")
    train_dataloader = get_dataloader(train_dataset, cfg, mode="train")

    val_dataset = get_dataset(val_df, cfg, mode="val")
    val_dataloader = get_dataloader(val_dataset, cfg, mode="val")

    if cfg.test:
        test_dataset = get_dataset(test_df, cfg, mode="test")
        test_dataloader = get_dataloader(test_dataset, cfg, mode="test")

    if cfg.train_val:
        train_val_dataset = get_dataset(train_df, cfg, mode="val")
        train_val_dataloader = get_dataloader(train_val_dataset, cfg, "val")

    model = get_model(cfg, train_dataset)
    model.to(device)

    total_steps = len(train_dataset)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()

    for epoch in range(cfg.epochs):

        set_seed(cfg.seed + epoch + cfg.local_rank)

        cfg.curr_epoch = epoch
        if cfg.local_rank == 0:
            print("EPOCH:", epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1

                cfg.curr_step += cfg.batch_size

                try:
                    data = next(tr_it)
                except Exception as e:
                    print(e)
                    print("DATA FETCH ERROR")

                model.train()
                torch.set_grad_enabled(True)

                batch = batch_to_device(data, device)

                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict["loss"]

                losses.append(loss.item())

                if cfg.mixed_precision:
                    scaler.scale(loss).backward()

                    if i % cfg.grad_accumulation == 0:
                        if cfg.clip_grad > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if cfg.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                    if i % cfg.grad_accumulation == 0:

                        optimizer.step()
                        optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if cfg.local_rank == 0 and cfg.curr_step % cfg.batch_size == 0:

                    loss_names = [key for key in output_dict if "loss" in key]

                    progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

            print(f"Mean train_loss {np.mean(losses):.4f}")

        if cfg.val:
            if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
                if cfg.local_rank == 0:
                    val_loss = run_eval(model, val_dataloader, cfg, pre="val", curr_epoch=epoch)
            else:
                val_score = 0

        if (cfg.local_rank == 0) and (cfg.epochs > 0) and (cfg.save_checkpoint):
            if not cfg.save_only_last_ckpt:
                checkpoint = create_checkpoint(
                    cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler
                )

                torch.save(
                    checkpoint,
                    f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth",
                )

    if (cfg.local_rank == 0) and (cfg.epochs > 0) and (cfg.save_checkpoint):
        # print(f'SAVING LAST EPOCH: val_loss {val_loss:.5}')
        checkpoint = create_checkpoint(
            cfg, model, optimizer, epoch, scheduler=scheduler, scaler=scaler
        )

        torch.save(
            checkpoint, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth"
        )
