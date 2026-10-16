from types import SimpleNamespace
import os

cfg = SimpleNamespace(**{})

# stages
cfg.train = True
cfg.val = True
cfg.test = False
cfg.train_val = False

# dataset
cfg.dataset = "base_ds"
cfg.n_classes = 1
cfg.batch_size = 32
cfg.val_df = None
cfg.test_df = None
cfg.batch_size_val = None
cfg.train_aug = None
cfg.val_aug = None
cfg.test_augs = False
cfg.discourse_types = [
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Evidence",
    "Lead",
    "Position",
    "Rebuttal",
    "None",
    "N/A",
]
cfg.data_sample = -1
cfg.dropout_tokens = 0.0

# img model
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.pretrained = True
cfg.pretrained_weights = None
cfg.pop_weights = None
cfg.pretrained_weights_strict = True
cfg.offline_inference = False
cfg.eval_epochs = 1
cfg.eval_train_epochs = 5
cfg.drop_path_rate = None
cfg.drop_rate = None
cfg.dropout = 0.0
cfg.warmup = 0
cfg.label_smoothing = 0

# training
cfg.fold = 0
cfg.val_fold = -1
cfg.lr = 1e-4
cfg.schedule = "cosine"
cfg.weight_decay = 0
cfg.optimizer = "Adam"
cfg.epochs = 10
cfg.seed = -1
cfg.resume_training = False
cfg.simple_eval = False
cfg.do_test = True
cfg.do_seg = False
cfg.eval_ddp = True
cfg.clip_grad = 0
cfg.save_val_data = True
cfg.gradient_checkpointing = False

# eval
cfg.calc_metric = True
cfg.calc_metric_epochs = 1
cfg.post_process_pipeline = "base_pp"

# ressources
cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.gpu = 0
cfg.num_workers = 4
cfg.drop_last = True
cfg.save_checkpoint = True
cfg.save_only_last_ckpt = False
cfg.save_weights_only = False
cfg.clip_mode = "norm"
cfg.norm_eps = 1e-4
cfg.loss = "bce"

cfg.data_dir = "./input/feedback-prize-2021/"
cfg.data_folder = cfg.data_dir + "train/"
cfg.test_data_folder = cfg.data_dir + "test/"
cfg.train_df = cfg.data_dir + "train_5folded_v1.csv"
cfg.output_dir = f"./output/models/{os.path.basename(__file__).split('.')[0]}"

basic_cfg = cfg
