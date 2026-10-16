from default_config import basic_cfg
import os

cfg = basic_cfg

# paths
cfg.name = os.path.basename(__file__).split(".")[0]

# model
cfg.model = "mdl_ch_8_deberta"
cfg.sep_bias = -4.0
cfg.class_bias = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
cfg.class_loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01]
cfg.loss_weights = [1, 20, 0.1]  # class, seps, flows
cfg.backbone = "microsoft/deberta-v3-large"

# DATASET
cfg.dataset = "ds_ch_6"
cfg.tokenizer = "microsoft/deberta-v3-large"
cfg.transformer_config = "/mount/feedback/huggingface_models/deberta-v3-large"
cfg.padding = True
cfg.truncation = True
cfg.max_length = 1250
cfg.max_length_test = 1250
cfg.dropout_tokens = 0.2  # word level
cfg.dropout = 0.2  # after last hidden layer

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs = 5
cfg.lr = [0.00003, 0.0003]
cfg.optimizer = "Adafactor_mixed"
cfg.weight_decay = 1e-5
cfg.warmup = 1
cfg.batch_size = 16
cfg.mixed_precision = True
cfg.pin_memory = False
cfg.grad_accumulation = 1
cfg.num_workers = 8
cfg.gradient_checkpointing = True

# EVAL
cfg.calc_metric = True
cfg.post_process_pipeline = "pp_pp_9"
cfg.save_weights_only = True
