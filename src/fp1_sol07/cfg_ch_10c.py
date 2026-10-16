from default_config import basic_cfg
import os

cfg = basic_cfg

# paths
cfg.name = os.path.basename(__file__).split(".")[0]

# model
cfg.model = "mdl_ch_2d"
cfg.loss_weights = [1, 50]
cfg.backbone = "allenai/longformer-large-4096"

# DATASET
cfg.dataset = "ds_ch_4"
cfg.tokenizer = "allenai/longformer-large-4096"
cfg.padding = True
cfg.truncation = True
cfg.max_length = 1024

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 3
cfg.lr = 0.00005
cfg.optimizer = "Adam"
cfg.weight_decay = 1e-5
cfg.warmup = 1
cfg.batch_size = 8
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 2.0
cfg.num_workers = 8

# EVAL
cfg.calc_metric = True
cfg.post_process_pipeline = "pp_ch_1b"
