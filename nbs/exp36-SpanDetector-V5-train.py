# %%
# ## Setup
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fold', type=int, default=0 )
parser.add_argument('-gpu', '--device', type=int, default=0)
parser.add_argument('-v', '--verbose', action='store_true')  # on/off flag
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device) #'0'


# %%
import os, sys, gc 
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path 
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold 
from sklearn.preprocessing import OneHotEncoder
import argparse
import collections
import yaml
from types import SimpleNamespace
import re
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss



import transformers
from transformers import DebertaTokenizer, DebertaForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from functools import partial
import multiprocessing as mp
from scipy.special import softmax

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import (AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, 
                          TrainingArguments, 
                          Trainer,
                          TrainerCallback)
from transformers import DebertaModel, DebertaV2Model


import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


import wandb
import warnings 
warnings.filterwarnings('ignore')
torch.set_warn_always(False)

print('torch.__version__:', torch.__version__)
print('transformers.__version__:', transformers.__version__)
print('PLightning.__version__:', pl.__version__)


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







# # Config
class CFG:

    exp_name = 'exp36_base' 
    task = "ner"

    n_folds = 5 # 10
    trn_folds = [args.fold] #[0]  

    model_name = "microsoft/deberta-base"
    # model_name = "microsoft/deberta-large"
    
    pretrained_detector_path = '/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/checkpoints/tasckj' #'none' # 'taskj'

    max_length = 1536 #768  
    stride = 128
    min_tokens = 6
    # TRAINING HYPERPARAMS
    batch_size = 16 #32 #4       # 16
    epochs = 4 #6 #4 #1
    gradient_checkpointing = True 
    scheduler = "Linear"

    LR = 2e-5  #6e-5 #2e-5    # 4e-5    ## 3e-05
    WD = 0.01
    WARMUP = 0.0 #0.1
    use_custom_trainer = False # set True for custom training loop - TODO
    fp16 = True 
    seed = 42
    train = True
    infer = False
    debug = False # set True for debugging
    wandb = True # False
    wb_project = "NLP-Rater"
    output = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/"

    accumulate_grad_batches = 4
    GRAD_ACC = 4 #8 #1 #2 #4  #   8 # 4 ## 8
    gradient_clip_val = 1.0

    restart_epoch = 0
    early_stopping = False
    patience = 100 
    hidden_drop_prob = 0.0 
    p_drop = 0.0 
    warmup_ratio = 0.1

    rnn = 'none'
    p_aug = 0.0
    mixup_alpha=1.0
    aug_stop_epoch=999
    adv_sift = False 
    l2norm = False
    awp = False
    awp_lr=1.0 
    awp_eps=0.01
    awp_start_epoch=0
    scheduler='cosine'

    # ######
    # DATA_BASE_DIR = cfg["DATA_BASE_DIR"]
    # LR = cfg["LR"]
    # NUM_EPOCHS = cfg["NUM_EPOCHS"]
    NUM_CORES = 8 #cfg["NUM_CORES"]
    # BATCH_SIZE = cfg["BATCH_SIZE"]
    # USE_FP16 = cfg["USE_FP16"]
    # GRAD_ACCUM_STEPS = cfg["GRAD_ACCUM_STEPS"]
    # MAX_SEQ_LENGTH = cfg["MAX_SEQ_LENGTH"]
    # SPLIT_NUM = cargs.fold
    # PRETRAINED_MODEL = cargs.pretrained_model

    # TRAIN_CSV = os.path.join(DATA_BASE_DIR, 'feedback-prize-2021/train.csv')
    # TRAIN_DIR = os.path.join(DATA_BASE_DIR, 'feedback-prize-2021/train/')

    MIN_TOKENS = {
        "Lead": 6,
        "Position": 3,
        "Evidence": 20,
        "Claim": 1,
        "Concluding Statement": 3,
        "Counterclaim": 7,
        "Rebuttal": 6
    }



import yaml 
import importlib


# # config = yaml.safe_load(open(parser_args.config).read()) ## {CFG.exp_name}
# with open(f'/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/config/exp08.yaml', 'r') as f:
#     config = yaml.safe_load(f)

# for k, v in config.items():
#     if type(v) == dict:
#         config[k] = SimpleNamespace(**v)
# cfg = SimpleNamespace(**config)
# print(cfg)

# %%
os.chdir('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3')
sys.path.append('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/')
sys.path.append('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/src/')

from src.utils.utils import *
from src.fp1_sol01.utils import Log, get_f1_score, after_deal
from src.fp1_sol01.utils import get_feat, dataset, Collate


DIR = Path("/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/")
OUTPUT = Path(CFG.output + CFG.exp_name)
OUTPUT.mkdir(exist_ok=True, parents=True)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# %%
# seed everything
CFG.device = device
# set_seed(CFG.seed)
seed_everything(CFG.seed)

# %%
# if CFG.wandb:
#     api_key = os.getenv('wb_api_key_core')
#     wandb.login(key=api_key)
#     wandb.init(project=CFG.wb_project, entity="core") #
#     # anony = None


discourse_type = [
    'Claim', 'Evidence', 'Position', 'Concluding Statement', 'Lead', 'Counterclaim', 'Rebuttal'
    ]




ner_labels = {
    'O': 0,
    'B-Lead': 1,
    'I-Lead': 2,
    'B-Position': 3,
    'I-Position': 4,
    'B-Evidence': 5,
    'I-Evidence': 6,
    'B-Claim': 7,
    'I-Claim': 8,
    'B-Concluding Statement': 9,
    'I-Concluding Statement': 10,
    'B-Counterclaim': 11,
    'I-Counterclaim': 12,
    'B-Rebuttal': 13,
    'I-Rebuttal': 14
    }

# %%
# ner_labels_eff = {
#     'O': 0,
#     'Non-Effective': 1,
#     'Effective': 2,
#     }

ner_labels_eff = {
    'Non-Effective': 0,
    'Effective': 1,
    }

inverted_ner_labels_eff = dict((v,k) for k,v in ner_labels_eff.items())
inverted_ner_labels_eff[-100] = 'Special Token'


## RATER mapping 
discourse_type_to_int = {
    'Lead': 0,
    'Position': 1,
    'Claim': 2,
    'Evidence': 3, 
    'Counterclaim': 4, 
    'Rebuttal': 5, 
    'Concluding Statement': 6, 
}


CLASSES = list(discourse_type_to_int.keys())

# LABEL2TYPE = ('Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement')
LABELS_TYPE = [
    'Lead', #0
    'Position', #1 
    'Claim', #2
    'Evidence', #3
    'Counterclaim', #4
    'Rebuttal', #5
    'Concluding Statement' #6 
]



TYPE2LABEL     = {t:l for l, t in enumerate(LABELS_TYPE)}
LABEL2TYPE = {k:v for v,k in TYPE2LABEL.items()}


# # Utils

# %%
## FP1-06 YOLO span detector 
## from: https://www.kaggle.com/code/tascj0/a-text-span-detector

from torchvision.ops import roi_align, nms

def to_gpu(data):
    '''
    https://www.kaggle.com/code/tascj0/a-text-span-detector
    '''
    if isinstance(data, dict):
        return {k: to_gpu(v) for k, v in data.items() }
        # return {k: to_gpu(v) for k, v in data.items() if k not in ['text', 'text_id']}
    elif isinstance(data, list):
        return [to_gpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return data

def to_np(t):
    if isinstance(t, torch.Tensor):
        return t.data.cpu().numpy()
    else:
        return t


def aggregate_tokens_to_words(feat, word_boxes):
    feat = feat.permute(0, 2, 1).unsqueeze(2)
    output = roi_align(feat, [word_boxes], 1, aligned=True)
    return output.squeeze(-1).squeeze(-1)


def span_nms(start, end, score, nms_thr=0.5):
    boxes = torch.stack(
        [
            start,
            torch.zeros_like(start),
            end,
            torch.ones_like(start),
        ],
        dim=1,
    ).float()
    keep = nms(boxes, score, nms_thr)
    return keep


### INIT Helpers 
### from mcvv 

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(
        module,
        a=1,
        mode='fan_in',
        nonlinearity='leaky_relu',
        distribution='uniform')


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def get_hf_scheduler(cfg, optimizer, total_steps):
    if cfg.scheduler == "Linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.WARMUP * (total_steps // cfg.batch_size)
            ),
            num_training_steps=cfg.epochs
            * (total_steps // cfg.batch_size),
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.WARMUP * (total_steps // cfg.batch_size)
            ),
            num_training_steps=cfg.epochs
            * (total_steps // cfg.batch_size),
        )
    return scheduler



def get_cos_scheduler(args, optimizer):
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer = optimizer,
        first_cycle_steps = args.num_train_steps, 
        cycle_mult = 1,
        max_lr = args.max_lr, 
        min_lr = args.min_lr, 
        warmup_steps = args.num_train_steps * 0.2,
        gamma = 1.,
        last_epoch = -1
    )
    return scheduler


def get_data(fold=0):
    
    df = pd.read_csv(DIR / "train_sgkf_fold5.csv")
    df['essay_id'] = df['id'] = df['essay_id_comp']
    df['discourse_effectiveness_int'] = df['discourse_effectiveness'].map(ner_labels_eff)

    ## add whole essays 
    df_texts = pd.read_csv('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/train_fold10_sgkf.csv')
    df_texts = df_texts[['essay_id', 'discourse_id', 'essay_text']]
    df = df.merge(df_texts, on=['essay_id', 'discourse_id'])

    if "class" not in df.columns:
        df["class"] = df["discourse_type"]

    print(df.shape)
    # display(df.head())

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)


    return train_df, valid_df
        


# # Data
train_df, valid_df = get_data(fold=CFG.trn_folds[0])
print('train.shape:', train_df.shape,'\t','test.shape:', valid_df.shape)



relation_mapper = {
    'Lead': ['Position',''],
    'Position': ['Lead',''],
    'Claim': ['Position',''],
    'Counterclaim': ['Position',''],
    'Rebuttal': ['Counterclaim',''],
    'Evidence': ['Claim',''],
    'Concluding Statement': ['Claim', 'Evidence']
}


from transformers import AutoModel, AutoTokenizer
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

# tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
# if 'deberta-v' in CFG.model_name or 'bigbird' in CFG.model_name:
#     tokenizer.add_tokens('\n')


if 'deberta-v2' in CFG.model_name or 'deberta-v3' in CFG.model_name:    
    tokenizer = DebertaV2TokenizerFast.from_pretrained(CFG.model_name, trim_offsets=False)
    special_tokens_dict = {'additional_special_tokens': ['\n\n'] + [f'[{s.upper()}]' for s in CLASSES]}
    _ = tokenizer.add_special_tokens(special_tokens_dict)
else:
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, trim_offsets=False)
    special_tokens_dict = {'additional_special_tokens': [f'[{s.upper()}]' for s in CLASSES]} # list(relation_mapper.keys()) 
    _ = tokenizer.add_special_tokens(special_tokens_dict)


CFG.tokenizer = tokenizer

# # Dataset 
class CustomCollator(object):
    def __init__(self, tokenizer, model_name): #model
        self.pad_token_id = tokenizer.pad_token_id

        config_model = config = AutoConfig.from_pretrained(model_name)
        # if hasattr(model.config, 'attention_window'):
        if hasattr(config_model, 'attention_window'):
            # For longformer
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/longformer/modeling_longformer.py#L1548
            self.attention_window = (
                config_model.attention_window if isinstance(
                    config_model.attention_window, int)
                else max(config_model.attention_window)
                )
        else:
            self.attention_window = None

    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
        if self.attention_window is not None:
            attention_window = self.attention_window
            padded_length = (attention_window -
                             max_seq_length % attention_window
                             ) % attention_window + max_seq_length
        else:
            padded_length = max_seq_length

        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)
        # labels_eff = torch.zeros(input_shape, dtype=torch.long)

        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']
        # labels_eff[0, :seq_length] = sample['labels_eff'][:seq_length]

        text_id = sample['text_id']
        text = sample['text']
        word_boxes = sample['word_boxes']
        gt_spans = sample['gt_spans']
        
        ## ++
        span_list = [sample['span_list']]
        discourse_types = [sample['discourse_types']]
        # labels_effectiveness = sample['labels_eff']

        # o = {}
        # for k in sample.keys():
        #     if k in ['span_list', 'discourse_types']:
        #         o[k] = [sample[k] for sample in samples]


        return dict(
            text_id=text_id,
            text=text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_boxes=word_boxes,
            gt_spans=gt_spans,
            ## ++
            span_list=span_list,
            discourse_types=discourse_types,
            # labels_effectiveness=labels_effectiveness
            # labels_eff=labels_eff
            )
    


# accepts file path, returns tuple of (file_ID, txt, NER labels)
def generate_NER_labels_for_file(uuid, df, id_col='essay_id'):

    curr_id = uuid 
    curr_txt = df[df[id_col]==uuid]['essay_text'].values[0].rstrip()

    # Set all token labels initially to non-label
    curr_labels = [ner_labels['O']] * len(curr_txt)

    # curr_labels_eff = [ner_labels_eff['O']] * len(curr_txt)
    curr_labels_eff = [-100] * len(curr_txt)
    
    # Iterate through all labels associated w/ file and update labels
    curr_df = df[df[id_col]==curr_id]
    for curr_discourse in curr_df.itertuples():

        curr_discourse_label     = curr_discourse.discourse_type 
        curr_discourse_label_eff = curr_discourse.discourse_effectiveness 

        for curr_txt_idx in range(curr_discourse.discourse_start, 
                                  min(curr_discourse.discourse_end+1, len(curr_labels))):
            if curr_txt_idx == curr_discourse.discourse_start:
                iob_label     = ner_labels['B-' + curr_discourse_label]
                # iob_label_eff = ner_labels_eff['B-' + curr_discourse_label_eff]

            else:
                iob_label     = ner_labels['I-' + curr_discourse_label]
                # iob_label_eff = ner_labels_eff['I-' + curr_discourse_label_eff]

            iob_label_eff = ner_labels_eff[curr_discourse_label_eff]

            curr_labels[curr_txt_idx]     = iob_label
            curr_labels_eff[curr_txt_idx] = iob_label_eff

    assert curr_labels != [ner_labels['O']] * len(curr_txt)
    # assert curr_labels_eff != [ner_labels_eff['O']] * len(curr_txt)
    assert curr_labels_eff != [-100] * len(curr_txt)

    # return curr_id, curr_txt, curr_labels, curr_labels_eff
    return curr_labels_eff

# [tokenizer.convert_tokens_to_ids(f'[{discourse_type.upper()}]') for discourse_type in CLASSES]


### FP2 2nd sol dataset 
# https://github.com/Shujun-He/TeamSKT-Feedback-Prize---Predicting-Effective-Arguments-2nd-Place-solution/blob/main/Shujun_solution/src/Feedback_Prize/Dataset.py

class Feedback2Dataset_02(Dataset):
    def __init__(self, df, tokenizer, mask_prob=0, mask_ratio=0, aug='false'):
        self.df = df
        self.unique_ids = sorted(df['essay_id'].unique())
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.aug = aug
        self.discourse_type_token_ids_dict = {
            discourse_type : tokenizer.convert_tokens_to_ids(f'[{discourse_type.upper()}]')
            for discourse_type in LABEL2TYPE
        }

        self.inv_discourse_type_token_ids_dict = {v:k for k,v in self.discourse_type_token_ids_dict.items()}
        
    def __len__(self):
        return len(self.unique_ids)
    
    def __getitem__(self, idx):
        essay_id = self.unique_ids[idx]
        sample_df = self.df[self.df['essay_id']==essay_id].reset_index(drop=True)
        discourse_ids = sample_df['discourse_id'].values

        text = ''
        discourse_types = []
        labels = []
        for discourse_type, discourse_text, label in zip(
            sample_df['discourse_type'].values, sample_df['discourse_text'].values, sample_df['discourse_effectiveness_int'].values
        ):
            text += f' [{discourse_type.upper()}] {discourse_text}'
            discourse_types.append(discourse_type.upper())
            labels.append(label)

        tokens = self.tokenizer.encode_plus(text, add_special_tokens=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.ones(len(input_ids)).long()
        
        span_list = []
        for i_token, input_id in enumerate(input_ids):
            if input_id in self.discourse_type_token_ids_dict.values():
                span_list.append(i_token)
        
        # random masking
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids)-1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = np.array([inds for inds in all_inds[:n_mask] if inds not in span_list])
            input_ids[mask_inds] = self.tokenizer.mask_token_id
        
        return dict(
            essay_id = essay_id,
            discourse_ids = discourse_ids,
            text = text,
            input_ids = input_ids,
            attention_mask = attention_mask,
            discourse_types = discourse_types,
            label = labels,
            span_list = span_list,
        )
    



class Feedback2Collator_02(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, samples):
        output = dict()
        
        for k in samples[0].keys():
            output[k] = [sample[k] for sample in samples]
        
        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])
        
        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s.tolist() + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s.tolist() + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s.tolist() for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s.tolist() for s in output["attention_mask"]]
            
        output["input_ids"] = torch.LongTensor(output["input_ids"])
        output["attention_mask"] = torch.LongTensor(output["attention_mask"])
        
        if "label" in output.keys():
            output["label"] = torch.LongTensor(output["label"])
        if "label2" in output.keys():
            output["label2"] = torch.LongTensor(output["label2"])
        return output


if CFG.debug:

    ds = Feedback2Dataset_02(train_df, CFG.tokenizer)
    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=Feedback2Collator_02(CFG.tokenizer))

# ds[0].keys()
##dict_keys(['essay_id', 'discourse_ids', 'text', 'input_ids', 'attention_mask', 'discourse_types', 'label', 'span_list'])


class SpanDetectorDataset(Dataset):
    def __init__(self, df,
                 tokenizer,
                 mask_prob=0.0,
                 mask_ratio=0.0):
        
        self.df = df.copy() #pd.read_csv(csv_file)
        self.samples = list(self.df.groupby('id'))
        self.tokenizer = tokenizer
        print(f'Loaded {len(self)} samples.')

        assert 0 <= mask_prob <= 1
        assert 0 <= mask_ratio <= 1
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        ###
        self.discourse_type_token_ids_dict = {
            discourse_type : tokenizer.convert_tokens_to_ids(f'[{discourse_type.upper()}]')
            for discourse_type in CLASSES
        }
        self.inv_discourse_type_token_ids_dict = {v:k for k,v in self.discourse_type_token_ids_dict.items()}



    def __len__(self):
        return len(self.samples)



    def __getitem__(self, index):
        text_id, df = self.samples[index]
        
        # text_path = os.path.join(self.text_dir, f'{text_id}.txt')
        # with open(text_path) as f:
        #     text = f.read().rstrip()

        text = df['essay_text'].values[0].rstrip()

        discourse_types = [] ## TODO replace 
        ############## Tom's sol 
        # if 'discourse_type' in df.columns:
        #     text = ''
        #     discourse_types = []
        #     labels_eff = []
        #     for disc_type_, disc_text_, lbl_eff_ in zip(
        #         df['discourse_type'].values, df['discourse_text'].values, df['discourse_effectiveness_int'].values
        #     ):
        #         text += f' [{disc_type_.upper()}] {disc_text_}'
        #         discourse_types.append(disc_type_.upper())
        #         labels_eff.append(lbl_eff_)

        # else:
        #     text = df['essay_text'].values[0].rstrip()
        #     #  * len() 

        # tokens = self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
        ##################################
        
        tokens = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.ones(len(input_ids)).long()
        
        # attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
        offset_mapping = self.strip_offset_mapping(text, offset_mapping)
        num_tokens = len(input_ids)


        # token slices of words
        woff = self.get_word_offsets(text)
        toff = offset_mapping
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1 + 1e-12)
        assert (ious > 0).any(-1).all()

        word_boxes = []
        for row in ious:
            inds = row.nonzero()[0]
            word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
        word_boxes = torch.FloatTensor(word_boxes)

        # word slices of ground truth spans
        gt_spans = []
        for _, row in df.iterrows():
            winds = row['predictionstring'].split()
            start = int(winds[0])
            end = int(winds[-1])
            span_label = TYPE2LABEL[row['discourse_type']]
            # gt_spans.append([start, end + 1, span_label])
            eff_label = ner_labels_eff[row['discourse_effectiveness']]
            gt_spans.append([start, end + 1, span_label, eff_label])

        gt_spans = torch.LongTensor(gt_spans)

        # random mask augmentation
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id



        #### TODO replace with gt_spans[:,0] --> change to orginal 
        # span_list = []
        # for i_token, input_id in enumerate(input_ids):
        #     if input_id in self.discourse_type_token_ids_dict.values():
        #         span_list.append(i_token)

        span_list = np.array(gt_spans)[:,0].tolist()
        
        if len(discourse_types)==0:
            discourse_types = np.array(gt_spans)[:,2].tolist()
            
        # ### Effective NER labels (3 class)
        # labels_eff = generate_NER_labels_for_file(text_id, self.df) # df
        # labels_eff = torch.LongTensor(labels_eff)

        return dict(
            text=text,
            text_id=text_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_boxes=word_boxes,
            gt_spans=gt_spans,
            ### ++ Tom 
            span_list=span_list,
            discourse_types=discourse_types,
            # labels_eff=labels_eff
                    )

    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)

    def get_word_offsets(self, text):
        matches = re.finditer("\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)



train_ds = SpanDetectorDataset(train_df, tokenizer=CFG.tokenizer, mask_prob=0.8, mask_ratio=0.3)
valid_ds = SpanDetectorDataset(valid_df, tokenizer=CFG.tokenizer)
collate = CustomCollator(CFG.tokenizer, CFG.model_name) #detector.model

train_loader = DataLoader(train_ds,
                      batch_size=1,
                      shuffle=True,
                      num_workers=2,
                      collate_fn=collate)

valid_loader = DataLoader(valid_ds,
                      batch_size=1,
                      shuffle=True,
                      num_workers=2,
                      collate_fn=collate)


# train_ds[0].keys()

# s = next(iter(train_loader))
# # s = train_ds[20]

# for k,v in s.items():
#     if k not in ['text', 'text_id', 'span_list', 'discourse_types']:
#         print(k, v.shape)
#     else:
#         print(k, v)



CFG.batch_size = 1 #args.train_batch_size
CFG.num_train_steps = int(len(train_ds) * CFG.epochs / CFG.batch_size)

# CFG.num_train_steps


# # Metric
# from fb1_4th.src.comp_metric import score_feedback_comp
# from copy import deepcopy 
from src.utils.RATER_MetricCode import compute_scores 



### RATER metric 

def remove_overlaps(segments):
    """
    Removes overlaps in predicted segments.

    Args:
    segments (list of tuples): Each tuple represents a segment and contains two elements - the start and end of the segment.

    Returns:
    list of tuples: The processed segments with overlaps removed.
    """
    # Sort segments based on their starting position
    segments.sort(key=lambda x: x[0])

    used_words = set()
    processed_segments = []

    for segment in segments:
        start, end = segment

        # Check for overlap with used words
        overlap = set(range(start, end+1)) & used_words

        # If there's an overlap, remove the used words from the current segment
        if overlap:
            new_segment = set(range(start, end+1)) - overlap

            # If the resulting segment is valid, add it to the list of processed segments
            if len(new_segment) > 1:
                processed_segments.append((min(new_segment), max(new_segment)))

            # Add the new words to the set of used words
            used_words.update(new_segment)
        else:
            # If there's no overlap, add the segment as is to the list of processed segments
            processed_segments.append(segment)

            # Add the words in the current segment to the set of used words
            used_words.update(range(start, end+1))

    return processed_segments

def calc_overlap(row):
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    return inter/max(len_gt,len_pred)

from sklearn.metrics import log_loss, roc_auc_score


def get_f1_score_v1(pred_df, test_df, log=print, slient=True):
    
    test_pred = pred_df.copy()
    
    if not slient:
        log('test_pred.shape:',test_pred.shape,'\ttest_df.shape:',test_df.shape,)
        log('pred class:\n',test_pred['class'].value_counts())
        log('true class:\n',test_df['discourse_type'].value_counts())

    # if 'class' not in test_pred.columns:
    #     test_pred['class'] = test_pred['discourse_type']
    # test_pred['class'] = test_pred['discourse_type']

    f1s = []
    for c in sorted(test_pred['class'].unique()):
        pred_df = test_pred.loc[test_pred['class']==c].copy()
        gt_df = test_df.loc[test_df['discourse_type']==c].copy()

        gt_df = gt_df[['id','discourse_type','predictionstring']].reset_index(drop=True).copy()
        pred_df = pred_df[['id','class','predictionstring']].reset_index(drop=True).copy()
        pred_df['pred_id'] = pred_df.index
        gt_df['gt_id'] = gt_df.index
        # Step 1. all ground truths and predictions for a given class are compared.
        joined = pred_df.merge(gt_df,
                               left_on=['id','class'],
                               right_on=['id','discourse_type'],
                               how='outer',
                               suffixes=('_pred','_gt')
                              )
        joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
        joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

        # print(joined.head())
        joined['min_overlaps'] = joined.apply(calc_overlap, axis=1)
        joined['potential_TP'] = (joined['min_overlaps'] >= 0.5)

        matched_pred_ids = joined.query('potential_TP')['pred_id'].unique()
        fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in matched_pred_ids]

        matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
        fn_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

        # Get numbers of each type
        TP = len(matched_gt_ids)
        FP = len(fp_pred_ids)
        FN = len(fn_gt_ids)
        #calc microf1
        f1_score = TP / (TP + 0.5*(FP+FN))
        
        if not slient:
            print(f'{c:<20} f1 score:\t{f1_score}')
        f1s.append(f1_score)
    
    print('\nOverall f1 score \t',  np.mean(f1s))

    ### Calculate the precision, recall, and F1 score
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    # F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    if not slient:
        print(f"TP: {TP}, FP: {FP}, FN: {FN}")
        print(f"pre: {precision}, rec: {recall}")

    return np.mean(f1s)


# %%
def get_f1_score(pred_df, test_df, log=print, slient=True):
    
    test_pred = pred_df.copy()
    
    if not slient:
        log('test_pred.shape:',test_pred.shape,'\ttest_df.shape:',test_df.shape,)
        log('pred class:\n',test_pred['class'].value_counts())
        log('true class:\n',test_df['discourse_type'].value_counts())
    

    test_pred['class'] = test_pred['discourse_type']

    f1s = []
    aucs = []
    logloss_scores = []
    total_scores = []
    for c in sorted(test_pred['class'].unique()):
        pred_df = test_pred.loc[test_pred['class']==c].copy()
        gt_df = test_df.loc[test_df['discourse_type']==c].copy()

        gt_df = gt_df[['id','discourse_type','predictionstring','discourse_effectiveness_int']].reset_index(drop=True).copy()
        pred_df = pred_df[['id','class','predictionstring', 'score_eff']].reset_index(drop=True).copy()
        
        pred_df['pred_id'] = pred_df.index
        gt_df['gt_id'] = gt_df.index
        
        # Step 1. all ground truths and predictions for a given class are compared.
        joined = pred_df.merge(gt_df,
                               left_on=['id','class'],
                               right_on=['id','discourse_type'],
                               how='outer',
                               suffixes=('_pred','_gt')
                              )
        joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
        joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

        # print(joined.head())
        joined['min_overlaps'] = joined.apply(calc_overlap, axis=1)
        joined['potential_TP'] = (joined['min_overlaps'] >= 0.5)

        matched_pred_ids = joined.query('potential_TP')['pred_id'].unique()
        fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in matched_pred_ids]

        matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
        fn_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

        # Get numbers of each type
        TP = len(matched_gt_ids)
        FP = len(fp_pred_ids)
        FN = len(fn_gt_ids)
        #calc microf1
        f1_score = TP / (TP + 0.5*(FP+FN))
        

        ###########
        ll_score = log_loss(
            joined[joined.potential_TP==True]['discourse_effectiveness_int'],
            joined[joined.potential_TP==True]['score_eff'],
            labels=[0,1]
        )
        logloss_scores.append(ll_score)

        auc_sc = roc_auc_score(
            joined[joined.potential_TP==True]['discourse_effectiveness_int'],
            joined[joined.potential_TP==True]['score_eff']
        )
        aucs.append(auc_sc)

        # #### try this
        # joined3 = joined[['potential_TP', 'discourse_effectiveness_int', 'score_eff']].dropna()
        # auc_sc2 = roc_auc_score(joined3['discourse_effectiveness_int'], joined3['score_eff'],)

        tot_score = 0.5*f1_score + 0.5*ll_score
        total_scores.append(tot_score)

        if not slient:
            print(f'{c:<20} f1 score:\t{f1_score}')
            print(f'{c:<20} logloss (TP) score:{ll_score:.4f} \t auc (TP) score: {auc_sc:.4f}')
            print(f'{c:<20} total score:\t{tot_score}')
            print()
        f1s.append(f1_score)

        ### Calculate the precision, recall, and F1 score
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        # F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        if not slient:
            print(f"TP: {TP}, FP: {FP}, FN: {FN}")
            print(f"pre: {precision}, rec: {recall}")
    
    print('\nOverall f1 score',  np.mean(f1s))
    print('Overall total score:\t', np.mean(total_scores))
    print()
    print('Overall logloss (TP) score:\t', np.mean(logloss_scores))
    print('Overall auc (TP) score:\t', np.mean(aucs))
    return np.mean(f1s)






### YOLO Span Detector 

class TextSpanDetector(nn.Module): #BaseModule --> mmcv 
    def __init__(self,
                 model_name,
                 num_classes=7,
                 num_classes2=3, 
                 dynamic_positive=False,
                 with_cp=False,
                 local_files_only=True,
                 init_cfg=None):
        super().__init__() #init_cfg

        self.num_classes = num_classes+1+2
        self.num_classes2 = num_classes2
        self.dynamic_positive = dynamic_positive
        self.loss_weights = [1.0, 1.0]
        
        # self.model = AutoModelForTokenClassification.from_pretrained(
        #     model_name,
        #     num_labels=1 + 2 + num_classes,
        #     local_files_only=local_files_only)
        
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name) # 768
        
        if with_cp:
            self.model.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        self.head_eff   = nn.Linear(config.hidden_size, self.num_classes2)

        # TODO init bias
        # self.model.classifier.bias.data[0].fill_(bias_init_with_prob(0.02))
        # self.model.classifier.bias.data[3:].fill_(bias_init_with_prob(1 / self.num_classes))

        self.loss_fct = nn.CrossEntropyLoss()
        # self.loss_fct2 = nn.CrossEntropyLoss()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.tokenizer = CFG.tokenizer


    def forward(self, batch):
        batch_size = batch['input_ids'].size(0)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'
        
        obj_pred, reg_pred, cls_pred, eff_pred = self.forward_logits(batch)
        
        obj_target, reg_target, cls_target, pos_loc = self.build_target(
            batch['gt_spans'], obj_pred, reg_pred, cls_pred)

        obj_loss, reg_loss, cls_loss = self.get_losses(obj_pred, reg_pred,
                                                       cls_pred, obj_target,
                                                       reg_target, cls_target,
                                                       pos_loc)
        loss1 = obj_loss + reg_loss + cls_loss
        
        # #### add NER loss (effectiveness)
        # # loss2 = self.loss_fct2(eff_pred.reshape(-1, self.num_classes2), batch['labels_eff'].reshape(-1))
        # loss2 = self.get_ner_loss(eff_pred, data['labels_eff'], batch['attention_mask'])

        # ### total loss 
        # loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2
        loss = loss1

        out = {}
        out['loss'] = loss
        out['loss1'] = loss1
        # out['loss2'] = loss2

        return out 



    def forward_logits(self, data):
        batch_size = data['input_ids'].size(0)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'
        
        ##### original 
        # outputs = self.model(input_ids=data['input_ids'],attention_mask=data['attention_mask'])
        # logits = outputs['logits']
        
        #### modify for AutoModel
        pooled_out = self.model(data['input_ids'], data['attention_mask'])[0]

        logits = self.classifier(self.dropout(pooled_out))
         
        logits = aggregate_tokens_to_words(logits, data['word_boxes'])
        assert logits.size(0) == data['text'].split().__len__()

        logits_eff = self.head_eff(self.dropout(pooled_out)) # effectiveness
        
        # logits_eff = aggregate_tokens_to_words(logits_eff, data['word_boxes'])
        # assert logits_eff.size(0) == data['text'].split().__len__()

        obj_pred = logits[..., 0]
        reg_pred = logits[..., 1:3]
        cls_pred = logits[..., 3:]
        return obj_pred, reg_pred, cls_pred, logits_eff


    def predict(self, data, test_score_thr):
        data = to_gpu(data)
        obj_pred, reg_pred, cls_pred = self.forward_logits(data)
        obj_pred = obj_pred.sigmoid()
        reg_pred = reg_pred.exp()
        cls_pred = cls_pred.sigmoid()

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        pr_scores = (obj_scores * cls_scores)**0.5
        pos_inds = pr_scores > test_score_thr

        if pos_inds.sum() == 0:
            return None

        pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]

        return dict(text_id=data['text_id'],
                    start=to_np(start),
                    end=to_np(end),
                    score=to_np(pr_score),
                    label=to_np(pr_label))

    
    # def train_step(self, data, optimizer, **kwargs):
    #     data = to_gpu(data)
        
    #     # obj_pred, reg_pred, cls_pred = self.forward_logits(data)
    #     obj_pred, reg_pred, cls_pred, eff_pred = self.forward_logits(data)


    #     obj_target, reg_target, cls_target, pos_loc = self.build_target(
    #         data['gt_spans'], obj_pred, reg_pred, cls_pred)

    #     obj_loss, reg_loss, cls_loss = self.get_losses(obj_pred, reg_pred,
    #                                                    cls_pred, obj_target,
    #                                                    reg_target, cls_target,
    #                                                    pos_loc)
    #     loss1 = obj_loss + reg_loss + cls_loss
        
    #     # #### add NER loss (effectiveness)
    #     # # loss2 = self.loss_fct2(eff_pred.reshape(-1, self.num_classes2), data['labels_eff'].reshape(-1))
    #     # loss2 = self.get_ner_loss(eff_pred, data['labels_eff'], data['attention_mask'])

    #     # ### total loss 
    #     # loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2

    #     loss = loss1

    #     log_vars = dict(
    #         obj_loss=obj_loss.item(),
    #         reg_loss=reg_loss.item(),
    #         cls_loss=cls_loss.item(),
    #         loss=loss.item(),
    #         # loss1=loss1.item(),
    #         # loss2=loss2.item(),
    #     )
    #     outputs = dict(loss=loss, log_vars=log_vars, num_samples=1)

    #     return outputs

    def get_losses(self, obj_pred, reg_pred, cls_pred, obj_target, reg_target,
                   cls_target, pos_loc):
        num_total_samples = pos_loc.numel()
        assert num_total_samples > 0
        reg_pred = reg_pred[pos_loc].exp()
        reg_target = reg_target[pos_loc]
        px1 = pos_loc - reg_pred[:, 0]
        px2 = pos_loc + reg_pred[:, 1]
        gx1 = reg_target[:, 0]
        gx2 = reg_target[:, 1]
        ix1 = torch.max(px1, gx1)
        ix2 = torch.min(px2, gx2)
        ux1 = torch.min(px1, gx1)
        ux2 = torch.max(px2, gx2)
        inter = (ix2 - ix1).clamp(min=0)
        union = (ux2 - ux1).clamp(min=0) + 1e-12
        iou = inter / union

        reg_loss = -iou.log().sum() / num_total_samples
        
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_pred[pos_loc],
            cls_target[pos_loc] * iou.detach().reshape(-1, 1),
            reduction='sum') / num_total_samples
        
        obj_loss = F.binary_cross_entropy_with_logits(
            obj_pred, obj_target, reduction='sum') / num_total_samples
        
        return obj_loss, reg_loss, cls_loss
    
    def get_ner_loss(self, outputs, targets, attention_mask):
        # loss_fct = nn.CrossEntropyLoss()
        active_loss = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = self.loss_fct(active_logits, true_labels)
        return loss   

    @torch.no_grad()
    def build_target(self, gt_spans, obj_pred, reg_pred, cls_pred):
        obj_target = torch.zeros_like(obj_pred)
        reg_target = torch.zeros_like(reg_pred)
        cls_target = torch.zeros_like(cls_pred)
        
        # first token as positive
        pos_loc = gt_spans[:, 0]
        obj_target[pos_loc] = 1
        reg_target[pos_loc, 0] = gt_spans[:, 0].float()
        reg_target[pos_loc, 1] = gt_spans[:, 1].float()
        cls_target[pos_loc, gt_spans[:, 2]] = 1
        
        # dynamically assign one more positive
        if self.dynamic_positive:
            cls_prob = (obj_pred.sigmoid().unsqueeze(1) *
                        cls_pred.sigmoid()).sqrt()
            for start, end, label in gt_spans:
                _cls_prob = cls_prob[start:end]
                _cls_gt = _cls_prob.new_full((_cls_prob.size(0), ),
                                             label,
                                             dtype=torch.long)
                _cls_gt = F.one_hot(
                    _cls_gt, num_classes=_cls_prob.size(1)).type_as(_cls_prob)
                cls_cost = F.binary_cross_entropy(_cls_prob,
                                                  _cls_gt,
                                                  reduction='none').sum(-1)
                _reg_pred = reg_pred[start:end].exp()
                _reg_loc = torch.arange(_reg_pred.size(0),
                                        device=_reg_pred.device)
                px1 = _reg_loc - _reg_pred[:, 0]
                px2 = _reg_loc + _reg_pred[:, 1]
                ix1 = torch.max(px1, _reg_loc[0])
                ix2 = torch.min(px2, _reg_loc[-1])
                ux1 = torch.min(px1, _reg_loc[0])
                ux2 = torch.max(px2, _reg_loc[-1])
                inter = (ix2 - ix1).clamp(min=0)
                union = (ux2 - ux1).clamp(min=0) + 1e-12
                iou = inter / union
                iou_cost = -torch.log(iou + 1e-12)
                cost = cls_cost + iou_cost

                pos_ind = start + cost.argmin()
                obj_target[pos_ind] = 1
                reg_target[pos_ind, 0] = start
                reg_target[pos_ind, 1] = end
                cls_target[pos_ind, label] = 1
            pos_loc = (obj_target == 1).nonzero().flatten()
        
        return obj_target, reg_target, cls_target, pos_loc



def check_model():

    # inp = torch.randint(0, 100, (2, CFG.max_length)) #.to(device)
    # lbl = torch.randint(0, 6, (2, CFG.max_length))   #.to(device)

    # dl = get_train_dataloader(train_ds, cfg)
    dl = train_loader
    batch = next(iter(dl))

    # mdl = RaterNet(cfg, vb=True)
    # mdl = Net(cfg)
    # mdl = TextModel(CFG.model_name, num_labels=len(args.labels_to_ids))
    # mdl = RaterModelMulti(CFG.model_name, num_labels=len(args.labels_to_ids))
    # mdl = RaterModel(CFG.model_name)
    mdl = TextSpanDetector(CFG.model_name)

    with torch.no_grad():
        out = mdl(batch)
        # out = mdl(batch, calculate_loss=True)
        # out = mdl(batch, calculate_loss=False)
        # out = mdl(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], labels2=batch['labels_eff'])
    
    print(out)
    print(out['loss'])
    # print(out['logits'].shape)
    # print('loss', out['loss'].item() )
    # print('loss_reg', out['loss_reg'].item() )
    # print('loss_aux', out['loss_aux'].item() )


    del mdl, out
    gc.collect()

    

# check_model()



# sd = torch.load("/kaggle/input/feedback-checkpoints/deberta_base_fold0.pth")['state_dict']
# new_sd = OrderedDict()
# for k, v in sd.items():
#     kk = k.replace("model.", "")
#     new_sd[kk] = v 

# self.model = AutoModelForTokenClassification.from_pretrained(
#     arch,
#     num_labels=1 + 2 + num_classes,
#     local_files_only=local_files_only)
# self.model.load_state_dict(new_sd)




# proba_thresh = { 
#     "Lead": 0.5*mult,
#     "Position": 0.4*mult,
#     "Evidence": 0.45*mult,
#     "Claim": 0.39*mult,
#     "Concluding Statement": 0.5*mult,
#     "Counterclaim": 0.35*mult,
#     "Rebuttal": 0.3*mult,

# %% [markdown]
# ## WoSpanDetector




# # Lightning Trainer

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from collections import OrderedDict
from src.losses.jaccard import *

from transformers import (AutoConfig, AutoModel, AutoTokenizer, AdamW,
                          get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_linear_schedule_with_warmup)


class LSTMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, p_drop=0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels,
                             hidden_size=out_channels,
                             num_layers=num_layers,
                             dropout=p_drop,
                             batch_first=True, 
                             bidirectional=True)
    def forward(self, x): #(bs,num_tokens,hidden_size)
        x,_ = self.lstm(x)
        return x
    
class GRUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, p_drop=0):
        super().__init__()
        self.lstm = nn.GRU(input_size=in_channels,
                           hidden_size=out_channels,
                           num_layers=num_layers,
                           dropout=p_drop,
                           batch_first=True, 
                           bidirectional=True)
    def forward(self, x): #(bs,num_tokens,hidden_size)
        x,_ = self.lstm(x)
        return x
    
    
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_layers=1, nhead=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=in_channels,nhead=nhead),
                                                 num_layers=num_layers)
    def forward(self, x):
        x = self.transformer(x)
        return x



class TextSpanDetectorPL(pl.LightningModule):
    def __init__(self,
            model_name,
            num_classes=7,
            num_classes2=2, #3, 
            dynamic_positive=True, #False,
            local_files_only=True,
            init_cfg=None, 
            cfg=None,
            train_df = None,
            valid_df = None,
            predict_df = None,
            # with_cp=False,
        ):
        super().__init__() #init_cfg

        self.cfg = cfg
        self.num_classes  = num_classes+1+2 ## 7 disc types + obj + reg
        self.num_classes2 = num_classes2
        self.dynamic_positive = dynamic_positive
        
        self.train_df = train_df
        self.valid_df = valid_df
        self.predict_df = predict_df
        self.loss_weights = [1.0, 1.0]

        self.test_score_thr = 0.15 #0.3 #0.5
        self.skip_validation = -1
        # self.scorer = scorer_fn
        # self.validation_step_outputs = []
        self.val_outs = []
        self.best_score = 0.0
        
        
        # self.backbone = AutoModelForTokenClassification.from_pretrained(model_name,
        #     num_labels=1 + 2 + num_classes,
        #     local_files_only=local_files_only)
        
        config_hf = AutoConfig.from_pretrained(model_name)
        config_hf.update({"max_position_embeddings": self.cfg.max_length})

        if 'deberta-v' in model_name:      
            self.backbone = DebertaV2Model(config_hf)
        elif 'deberta-' in model_name:   
            self.backbone = DebertaModel(config_hf) 
        else:
            self.backbone = AutoModel.from_pretrained(model_name) # 768

        
        if self.cfg.gradient_checkpointing: #with_cp:
            self.backbone.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config_hf.hidden_size, self.num_classes)
        self.head_eff   = nn.Linear(config_hf.hidden_size, self.num_classes2)

        # TODO init bias
        self.classifier.bias.data[0].fill_(bias_init_with_prob(0.02))
        self.classifier.bias.data[3:].fill_(bias_init_with_prob(1 / self.num_classes))

        self.loss_fct = nn.CrossEntropyLoss()
        # self.loss_fct2 = nn.CrossEntropyLoss()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.tokenizer = cfg.tokenizer


    def forward(self, batch):
        # x = self.backbone(**inputs).last_hidden_state
        # x = self.dropout(x)
        # x = self.fc(x)
        ################
        batch_size = batch['input_ids'].size(0)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'
        
        obj_pred, reg_pred, cls_pred, logits_eff_word, outputs = self.forward_logits(batch)
        
        obj_target, reg_target, cls_target, cls_eff_target, pos_loc = self.build_target(
            batch['gt_spans'], obj_pred, reg_pred, cls_pred, logits_eff_word)

        obj_loss, reg_loss, cls_loss, cls_eff_loss = self.get_losses(
            obj_pred,   reg_pred,   cls_pred,   logits_eff_word, 
            obj_target, reg_target, cls_target, cls_eff_target, 
            pos_loc
            )
        
        loss1 = obj_loss + reg_loss + cls_loss
        loss2 = cls_eff_loss

        # #### add NER loss (effectiveness)
        # # loss2 = self.loss_fct2(eff_pred.reshape(-1, self.num_classes2), batch['labels_eff'].reshape(-1))
        # loss2 = self.get_ner_loss(logits_eff_raw, batch['labels_eff'], batch['attention_mask']) #logits_eff_word TODO try
        # loss2 = self.get_ner_loss(logits_eff_word, batch['labels_eff'], batch['attention_mask'])  

        # ### total loss 
        #loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2
        loss = obj_loss + reg_loss + cls_loss + cls_eff_loss

        out = {}
        out['loss'] = loss
        out['loss1'] = loss1
        out['loss2'] = loss2
        # out['logits'] = logits_raw
        # out['logits_eff'] = logits_eff_raw
        out['logits_eff_word'] = logits_eff_word
        return out
    

    def forward_logits(self, data):
        ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
        bs = data['input_ids'].size(0)
        assert bs == 1, f'Only batch_size=1 supported, got batch_size={bs}.'
        
        ##### original 
        # outputs = self.model(input_ids=data['input_ids'],attention_mask=data['attention_mask'])
        # logits = outputs['logits']
        
        #### modify for AutoModel
        outputs = self.backbone(data['input_ids'], data['attention_mask']) 
        pooled_out = outputs[0]

        logits_raw = self.classifier(self.dropout(pooled_out))
         
        logits = aggregate_tokens_to_words(logits_raw, data['word_boxes'])
        assert logits.size(0) == data['text'].split().__len__()

        logits_eff_raw = self.head_eff(self.dropout(pooled_out)) # effectiveness
        
        logits_eff = aggregate_tokens_to_words(logits_eff_raw, data['word_boxes'])
        assert logits_eff.size(0) == data['text'].split().__len__()

        obj_pred = logits[..., 0]
        reg_pred = logits[..., 1:3]
        cls_pred = logits[..., 3:]

        return obj_pred, reg_pred, cls_pred, logits_eff, outputs
    
    ###### Aux functions 
    def get_losses(
            self, obj_pred, reg_pred, cls_pred, logits_eff_word, 
            obj_target, reg_target, cls_target, cls_eff_target, pos_loc):
        
        num_total_samples = pos_loc.numel()
        assert num_total_samples > 0
        reg_pred = reg_pred[pos_loc].exp()
        reg_target = reg_target[pos_loc]
        px1 = pos_loc - reg_pred[:, 0]
        px2 = pos_loc + reg_pred[:, 1]
        gx1 = reg_target[:, 0]
        gx2 = reg_target[:, 1]
        ix1 = torch.max(px1, gx1)
        ix2 = torch.min(px2, gx2)
        ux1 = torch.min(px1, gx1)
        ux2 = torch.max(px2, gx2)
        inter = (ix2 - ix1).clamp(min=0)
        union = (ux2 - ux1).clamp(min=0) + 1e-12
        iou = inter / union

        reg_loss = -iou.log().sum() / num_total_samples
        
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_pred[pos_loc],
            cls_target[pos_loc] * iou.detach().reshape(-1, 1),
            reduction='sum') / num_total_samples
        
        obj_loss = F.binary_cross_entropy_with_logits(
            obj_pred, obj_target, reduction='sum') / num_total_samples
        
        ###### ++
        cls_eff_loss = F.binary_cross_entropy_with_logits(
            logits_eff_word[pos_loc],
            cls_eff_target[pos_loc] * iou.detach().reshape(-1, 1),
            reduction='sum') / num_total_samples

        return obj_loss, reg_loss, cls_loss, cls_eff_loss

    def get_ner_loss(self, outputs, targets, attention_mask):
        # loss_fct = nn.CrossEntropyLoss()
        active_loss = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = self.loss_fct(active_logits, true_labels)
        return loss  
    
    
    @torch.no_grad()
    def build_target(self, gt_spans, obj_pred, reg_pred, cls_pred, cls_pred_eff):
        obj_target = torch.zeros_like(obj_pred)
        reg_target = torch.zeros_like(reg_pred)
        cls_target = torch.zeros_like(cls_pred)
        cls_eff_target = torch.zeros_like(cls_pred_eff)
        
        # first token as positive
        pos_loc = gt_spans[:, 0]
        obj_target[pos_loc] = 1
        reg_target[pos_loc, 0] = gt_spans[:, 0].half() if self.cfg.fp16 else gt_spans[:, 0].float()
        reg_target[pos_loc, 1] = gt_spans[:, 1].half() if self.cfg.fp16 else gt_spans[:, 0].float()
        cls_target[pos_loc, gt_spans[:, 2]] = 1
        ### ++
        cls_eff_target[pos_loc, gt_spans[:, 3]] = 1

        # dynamically assign one more positive
        if self.dynamic_positive:
            
            # cls_prob = (obj_pred.sigmoid().unsqueeze(1) * cls_pred.sigmoid()).sqrt()
            cls_prob = (obj_pred.unsqueeze(1) * cls_pred).sqrt() ## for BCEwithlogits

            for start, end, label, label_eff in gt_spans:
                _cls_prob = cls_prob[start:end]
                _cls_gt = _cls_prob.new_full((_cls_prob.size(0), ),
                                             label,
                                             dtype=torch.long)
                _cls_gt = F.one_hot(
                    _cls_gt, num_classes=_cls_prob.size(1)).type_as(_cls_prob)
                
                # cls_cost = F.binary_cross_entropy(_cls_prob, _cls_gt, reduction='none').sum(-1)
                ## Try with BCELogits
                cls_cost = F.binary_cross_entropy_with_logits(_cls_prob,
                                                  _cls_gt,
                                                  reduction='none').sum(-1)


                _reg_pred = reg_pred[start:end].exp()
                _reg_loc = torch.arange(_reg_pred.size(0),
                                        device=_reg_pred.device)
                px1 = _reg_loc - _reg_pred[:, 0]
                px2 = _reg_loc + _reg_pred[:, 1]
                ix1 = torch.max(px1, _reg_loc[0])
                ix2 = torch.min(px2, _reg_loc[-1])
                ux1 = torch.min(px1, _reg_loc[0])
                ux2 = torch.max(px2, _reg_loc[-1])
                inter = (ix2 - ix1).clamp(min=0)
                union = (ux2 - ux1).clamp(min=0) + 1e-12
                iou = inter / union
                iou_cost = -torch.log(iou + 1e-12)
                cost = cls_cost + iou_cost

                pos_ind = start + cost.argmin()
                obj_target[pos_ind] = 1
                reg_target[pos_ind, 0] = start
                reg_target[pos_ind, 1] = end
                cls_target[pos_ind, label] = 1
                ## ++ TODO try next run 
                # cls_eff_target[pos_ind, label_eff] = 1
            pos_loc = (obj_target == 1).nonzero().flatten()
        
        return obj_target, reg_target, cls_target, cls_eff_target, pos_loc # 



    ##############
    def shared_step(self, batch):
        # text, span_labels, label_ids, label_counts = batch
        # output = self(text)
        # preds, new_span_labels, loss_divs = preds_to_span_preds(output, label_ids, span_labels, label_counts)
        # loss = self.criterion(preds, new_span_labels.cuda())
        # preds = output.detach().cpu()
        ######################################################
        # batch = to_gpu(batch)
        
        # obj_pred, reg_pred, cls_pred = self.forward_logits(batch)
        obj_pred, reg_pred, cls_pred, logits_eff_word, outputs = self.forward_logits(batch)

        obj_target, reg_target, cls_target, cls_eff_target, pos_loc = self.build_target(
            batch['gt_spans'], obj_pred, reg_pred, cls_pred, logits_eff_word)

        obj_loss, reg_loss, cls_loss, cls_eff_loss = self.get_losses(
            obj_pred, reg_pred, cls_pred, logits_eff_word,
            obj_target, reg_target, cls_target, cls_eff_target, 
            pos_loc
            )
        loss1 = obj_loss + reg_loss + cls_loss
        
        # #### add NER loss (effectiveness)
        # # loss2 = self.loss_fct2(eff_pred.reshape(-1, self.num_classes2), data['labels_eff'].reshape(-1))
        # loss2 = self.get_ner_loss(logits_eff_raw, batch['labels_eff'], batch['attention_mask'])
        loss2 = cls_eff_loss

        ### total loss 
        loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2

        return {
            'loss': loss, 
            'obj_loss': obj_loss,
            'reg_loss': reg_loss,
            'cls_loss': cls_loss,
            'eff_loss': loss2,
            #####
            'cls_preds': cls_pred,
            # 'logits': logits_raw.softmax(-1),
            # 'logits_eff_raw': logits_eff_raw.softmax(-1),
            'logits_eff_word': logits_eff_word.softmax(-1),
            }
    

    def training_step(self, batch, batch_idx):

        out = self.shared_step(batch)

        self.log("train_loss", out['loss'], on_step=True, logger=True, prog_bar=True)
        self.log("train_obj_loss", out['obj_loss'], on_step=True, logger=True, prog_bar=True)
        self.log("train_reg_loss", out['reg_loss'], on_step=True, logger=True, prog_bar=True)
        self.log("train_cls_loss", out['cls_loss'], on_step=True, logger=True, prog_bar=True)
        self.log("train_eff_loss", out['eff_loss'], on_step=True, logger=True, prog_bar=True)
        # self.log("train_f1", f1, on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return out
    
    def train_epoch_end(self,outputs):
        tr_avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tr_obj_loss = torch.stack([x['obj_loss'] for x in outputs]).mean()
        tr_reg_loss = torch.stack([x['reg_loss'] for x in outputs]).mean()
        tr_cls_loss = torch.stack([x['cls_loss'] for x in outputs]).mean()

        print(f'epoch {trainer.current_epoch} training losses: {tr_avg_loss:.4f}, obj:{tr_obj_loss:.4f}, reg:{tr_reg_loss:.4f}, cls:{tr_cls_loss:.4f}')
        return {'train_loss_epoch': tr_avg_loss} 


    def validation_step(self, batch, batch_idx):
        # if self.skip_validation > 0:
        #     return {}

        out = self.shared_step(batch)

        self.log("val_loss", out['loss'], on_step=True, logger=True, prog_bar=True)
        self.log("val_obj_loss", out['obj_loss'], on_step=True, logger=True, prog_bar=True)
        self.log("val_reg_loss", out['reg_loss'], on_step=True, logger=True, prog_bar=True)
        self.log("val_cls_loss", out['cls_loss'], on_step=True, logger=True, prog_bar=True)
        self.log("val_eff_loss", out['eff_loss'], on_step=True, logger=True, prog_bar=True)
        # self.log("train_f1", f1, on_step=False, on_epoch=True, logger=True, prog_bar=False)

        self.val_outs.append({
            "val_loss": out['loss'], 
            "val_obj_loss": out['obj_loss'],
            "val_reg_loss": out['reg_loss'],
            "val_cls_loss": out['cls_loss'],
            "val_eff_loss": out['eff_loss'],
            ###
            # "logits": out['logits'], 
            # "logits_eff": out['logits_eff_raw'], 
            "cls_preds": out['cls_preds'],
            # "targets": target
            #### 
            'input_ids': batch['input_ids'], 
            'attention_mask': batch['attention_mask'], 
            'word_boxes': batch['word_boxes'], 
            'text': batch['text'], 
            'text_id': batch['text_id'], 
            })
        return out 


    def on_validation_epoch_start(self):
        print('\n')


    def on_validation_epoch_end(self):

        outputs = self.val_outs
        # print('len outputs', len(outputs))
        
        val_loss = 0
        preds = []
        for i, o in enumerate(outputs):
            inp_ids = o['input_ids']
            att_mask = o['attention_mask']
            word_boxes = o['word_boxes']
            text = o['text']
            txt_id = o['text_id']
            if o['val_loss'] is not None:
                val_loss += o['val_loss'].detach().cpu().item()
            batch = {'input_ids': inp_ids, 'attention_mask': att_mask, 'word_boxes': word_boxes, 'text_id': txt_id, 'text': text}

            ######################################################
            # out = self.predict_step(batch, i)
            ######################################################
            ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
            obj_pred, reg_pred, cls_pred, logits_eff_word, outputs = self.forward_logits(batch)
            obj_pred = obj_pred.sigmoid()
            reg_pred = reg_pred.exp()
            cls_pred = cls_pred.sigmoid()

            obj_scores = obj_pred
            cls_scores, cls_labels = cls_pred.max(-1)
            pr_scores = (obj_scores * cls_scores)**0.5
            pos_inds = pr_scores > self.test_score_thr ## TODO try thres per class 

            if pos_inds.sum() == 0:
                return None

            pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
            pos_loc = pos_inds.nonzero().flatten()
            start = pos_loc - reg_pred[pos_inds, 0]
            end = pos_loc + reg_pred[pos_inds, 1]

            min_idx, max_idx = 0, obj_pred.numel() - 1
            start = start.clamp(min=min_idx, max=max_idx).round().long()
            end = end.clamp(min=min_idx, max=max_idx).round().long()

            # nms
            keep = span_nms(start, end, pr_score)
            start = start[keep]
            end = end[keep]
            pr_score = pr_score[keep]
            pr_label = pr_label[keep]

            ### incl effectiveness preds
            logits_eff_word = torch.softmax(logits_eff_word, -1)

            eff_proba, eff_cls = logits_eff_word.max(-1) 

            eff_proba = eff_proba[keep]
            eff_cls   = eff_cls[keep]

            # # eff_proba, eff_cls = logits_eff_word[start:end].max(-1) 
            # eff_proba = []
            # eff_cls = []
            # for st_, en_ in zip(start, end):
            #     ## apply first mean then max
            #     active_logits_eff_word = logits_eff_word[st_:en_].mean()
            #     eff_proba_, eff_cls_ = active_logits_eff_word.max(-1) 
            #     ## TODO try first max then mean

            #     eff_proba.append(eff_proba_.detach().cpu().numpy())
            #     eff_cls.append(eff_cls_.detach().cpu().numpy())

            out = dict(
                text_id=batch['text_id'],
                start=to_np(start),
                end=to_np(end),
                score=to_np(pr_score),
                label=to_np(pr_label),
                ##### ++ 
                score_eff=to_np(eff_proba),
                label_eff=to_np(eff_cls),
                )
            ###########################
            test_ps = self.result_to_ps(out) # print('test_ps', test_ps)      
            # test_ps = format_result(out) #     
            preds.append(test_ps)  

         
        pred_df = pd.concat(preds)
        # print('pred_df', pred_df)

        metrics = get_f1_score(pred_df, self.valid_df, slient=False)
        # metrics = self.compute_metrics(logits , targets, attention_mask=mask)["f1"]
        # metrics = self.scorer.score(preds)
        avg_f1 = metrics#['val_f1_avg']

        ### reset placeholder 
        self.val_outs = []    

        if avg_f1>self.best_score:
            self.best_score = avg_f1 
        

        val_loss = val_loss / len(outputs)

        self.log(f'val_loss_epoch', val_loss, on_epoch=True)
        self.log(f'val_f1_avg', avg_f1, on_epoch=True)

        print(f'epoch {trainer.current_epoch} validation loss: {val_loss:.4f}')
        print(f'epoch {trainer.current_epoch} validation f1: {avg_f1} (best f1: {self.best_score:.4f})')
        # return metrics
        return {'val_loss_epoch': val_loss, 'val_f1': avg_f1}
    

    ### TODO check predict loop
    def predict_step(self, batch, batch_idx):
        ######################################################
        ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
        obj_pred, reg_pred, cls_pred, logits_eff_word, _ = self.forward_logits(batch)
        obj_pred = obj_pred.sigmoid()
        reg_pred = reg_pred.exp()
        cls_pred = cls_pred.sigmoid()

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        pr_scores = (obj_scores * cls_scores)**0.5
        pos_inds = pr_scores > self.test_score_thr ## TODO try thres per class 

        if pos_inds.sum() == 0:
            return None

        pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]

        ### incl effectiveness preds
        # logits_eff_word = torch.softmax(logits_eff_word, -1)
        logits_eff_word = logits_eff_word.sigmoid() ## TODO try with 

        eff_proba, eff_cls = logits_eff_word.max(-1) 

        eff_proba = eff_proba[keep]
        eff_cls   = eff_cls[keep]

        ###########################################################
        # # eff_proba, eff_cls = logits_eff_word[start:end].max(-1) 
        # eff_proba = []
        # eff_cls = []
        # for st_, en_ in zip(start, end):
        #     ## apply first mean then max
        #     active_logits_eff_word = logits_eff_word[st_:en_].mean()
        #     eff_proba_, eff_cls_ = active_logits_eff_word.max(-1) 

        #     eff_proba.append(eff_proba_.detach().cpu().numpy())
        #     eff_cls.append(eff_cls_.detach().cpu().numpy())


        return dict(
            text_id=batch['text_id'],
            start=to_np(start),
            end=to_np(end),
            score=to_np(pr_score),
            label=to_np(pr_label),
            ##### ++ 
            score_eff=to_np(eff_proba),
            label_eff=to_np(eff_cls),
            )


    def configure_optimizers(self):

        weight_decay = self.cfg.WD # self.cfg.training.weight_decay #self.cfg['optimizer']['weight_decay']
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                'params': [p for n, p in param_optimizer if 'backbone' in n and not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': self.cfg.LR, # self.cfg['optimizer']['params']['lr'],
            },
            {
                'params': [p for n, p in param_optimizer if 'backbone' in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.cfg.LR, #self.cfg['optimizer']['params']['lr'],
            },
            {
                'params': [p for n, p in param_optimizer if not 'backbone' in n and not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                # 'lr': self.cfg['optimizer']['params']['lr'] * self.cfg['optimizer']['head_lr_factor'],
                'lr': self.cfg.LR * 5.0,
            },
            {
                'params': [p for n, p in param_optimizer if not 'backbone' in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                # 'lr': self.cfg['optimizer']['params']['lr'] * self.cfg['optimizer']['head_lr_factor'],
                'lr': self.cfg.LR * 5.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.cfg.LR, eps = 1.0e-06, betas = [0.9, 0.999]) 

        # scheduler = get_scheduler_cos(self.cfg, optimizer) 
        scheduler = get_hf_scheduler(self.cfg, optimizer, self.cfg.num_train_steps)  
        
        lr_scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step', 
                'frequency': 1,
            }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}


    def result_to_ps(self, out):

        start_list = out['start']
        end_list = out['end']
        scores_list = out['score']
        labels_list = out['label']
        txt_id = out['text_id']

        scores_list2 = out['score_eff'] if 'score_eff' in out.keys() else np.zeros(len(start_list))
        labels_list2 = out['label_eff'] if 'label_eff' in out.keys() else np.zeros(len(start_list))

        scores_list2_0 = out['score_eff_0'] if 'score_eff_0' in out.keys() else np.zeros(len(start_list))
        scores_list2_1 = out['score_eff_1'] if 'score_eff_1' in out.keys() else np.zeros(len(start_list))

        res = []
        for start, end, score, label, score2, label2, score2_0, score2_1 in zip(start_list, end_list, scores_list, labels_list, scores_list2, labels_list2, scores_list2_0, scores_list2_1):
            res.append({
                'id': txt_id,
                'start': start, 
                'end': end, 
                'predictionstring': " ".join([str(i) for i in range(start, end+1)]),
                'score': score,
                'class': label,
                ###
                'class_eff': label2,
                'score_eff': score2,
                'score_eff_0': score2_0,
                'score_eff_1': score2_1,
            })
        res = pd.DataFrame(res)
        res['discourse_type'] = res['class']        # int 
        res['class'] = res['class'].map(LABEL2TYPE) # str 
        ### NEW
        res['score_discourse_effectiveness_0'] = res['score_eff_0'].copy()
        res['score_discourse_effectiveness_1'] = res['score_eff_1'].copy()
        return res
    

    def train_dataloader(self):
        return self._train_dataloader 
    

    def validation_dataloader(self):
        return self._valid_dataloader


def format_result(out):

        start_list = out['start']
        end_list = out['end']
        scores_list = out['score']
        labels_list = out['label']
        txt_id = out['text_id']

        scores_list2 = out['score_eff'] if 'score_eff' in out.keys() else np.zeros(len(start_list))
        labels_list2 = out['label_eff'] if 'label_eff' in out.keys() else np.zeros(len(start_list))

        scores_list2_0 = out['score_eff_0'] if 'score_eff_0' in out.keys() else np.zeros(len(start_list))
        scores_list2_1 = out['score_eff_1'] if 'score_eff_1' in out.keys() else np.zeros(len(start_list))

        res = []
        for start, end, score, label, score2, label2, score2_0, score2_1 in zip(start_list, end_list, scores_list, labels_list, scores_list2, labels_list2, scores_list2_0, scores_list2_1):
            res.append({
                'id': txt_id,
                'start': start, 
                'end': end, 
                'predictionstring': " ".join([str(i) for i in range(start, end+1)]),
                'score': score,
                'class': label,
                ###
                'class_eff': label2,
                'score_eff': score2,
                'score_eff_0': score2_0,
                'score_eff_1': score2_1,
            })
        res = pd.DataFrame(res)
        res['discourse_type'] = res['class']        # int 
        res['class'] = res['class'].map(LABEL2TYPE) # str 

        ### NEW
        res['score_discourse_effectiveness_0'] = res['score_eff_0'].copy()
        res['score_discourse_effectiveness_1'] = res['score_eff_1'].copy()
        
        return res



class TextSpanDetectorPL_FP1(pl.LightningModule):
    def __init__(self,
            model_name,
            num_classes=7,
            num_classes2=2, #3, 
            dynamic_positive=True, #False,
            local_files_only=True,
            init_cfg=None, 
            cfg=None,
            train_df = None,
            valid_df = None,
            predict_df = None,
            # pretrained_path="",
            pretrained_path="/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/checkpoints/", ## None
            # with_cp=False,
            use_sigmoid=False,
            test_score_thr = 0.2, # 0.4,
            nms_thr=0.5,
            warmup=0.0,
            vb=False
        ):
        super().__init__() #init_cfg

        self.cfg = cfg
        self.num_classes  = num_classes+1+2 ## 7 disc types + obj + reg
        self.num_classes2 = num_classes2
        self.dynamic_positive = dynamic_positive
        self.warmup_ratio = warmup #0.1
        self.train_df = train_df
        self.valid_df = valid_df
        self.predict_df = predict_df
        # self.loss_weights = [1.0, 4.0]    #[1.0, 1.0]     ## run 1
        self.loss_weights = [1.0, 1.0]  ## run 2 

        self.use_sigmoid = True if num_classes2==1 else use_sigmoid
        self.test_score_thr=test_score_thr #0.5
        self.nms_thr=nms_thr
        self.skip_validation = -1
        self.val_outs = []
        self.best_score = 0.0
        self.vb=vb

        if pretrained_path != "":

            if "deberta-large" in model_name:
                sd = torch.load(pretrained_path + "tasckj/deberta_large_fold0.pth")['state_dict']
            elif "deberta-base" in model_name:
                sd = torch.load(pretrained_path + "tasckj/deberta_base_fold0.pth")['state_dict']
            elif "longformer" in model_name:
                sd = torch.load(pretrained_path + "tasckj/longformer_large_fold0.pth")['state_dict']
            else:
                print("Check weights path..")
            new_sd = OrderedDict()
            for k, v in sd.items():
                kk = k.replace("model.", "")
                new_sd[kk] = v 

            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=1 + 2 + num_classes,
                local_files_only=local_files_only)
            self.model.load_state_dict(new_sd)

            # self.backbone = AutoModelForTokenClassification.from_pretrained(model_name,
            #     num_labels=1 + 2 + num_classes,
            #     local_files_only=local_files_only)

            config_hf = AutoConfig.from_pretrained(model_name)
            

        else:
            config_hf = AutoConfig.from_pretrained(model_name)
            # config_hf.update({"max_position_embeddings": self.cfg.max_length})

            if 'deberta-v' in model_name:      
                self.backbone = DebertaV2Model(config_hf)
            elif 'deberta-' in model_name:   
                self.backbone = DebertaModel(config_hf) 
            else:
                self.backbone = AutoModel.from_pretrained(model_name) # 768

        if self.cfg.gradient_checkpointing: #with_cp:
            self.model.gradient_checkpointing_enable()
            # self.backbone.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(config_hf.hidden_size, self.num_classes)
        self.head_eff   = nn.Linear(config_hf.hidden_size, self.num_classes2)

        ## init bias
        # self.classifier.bias.data[0].fill_(bias_init_with_prob(0.02))
        # self.classifier.bias.data[3:].fill_(bias_init_with_prob(1 / self.num_classes))

        self.model.classifier.bias.data[0].fill_(bias_init_with_prob(0.02))
        self.model.classifier.bias.data[3:].fill_(bias_init_with_prob(1 / self.num_classes))

        self.head_eff.bias.data[0].fill_(bias_init_with_prob(0.5))

        # self.loss_fct = nn.CrossEntropyLoss()
        # self.loss_fct2 = nn.CrossEntropyLoss()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.tokenizer = cfg.tokenizer


    def forward(self, batch):
        batch_size = batch['input_ids'].size(0)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'
        
        ## obj_pred, reg_pred, cls_pred, logits_eff_word, logits_raw_types, logits_raw_eff
        obj_pred, reg_pred, cls_pred, logits_eff_word, _, _ = self.forward_logits(batch)
        
        obj_target, reg_target, cls_target, cls_eff_target, pos_loc = self.build_target(
            batch['gt_spans'], obj_pred, reg_pred, cls_pred, logits_eff_word)

        obj_loss, reg_loss, cls_loss, cls_eff_loss = self.get_losses(
            obj_pred,   reg_pred,   cls_pred,   logits_eff_word, 
            obj_target, reg_target, cls_target, cls_eff_target, 
            pos_loc
            )
        
        loss1 = obj_loss + reg_loss + cls_loss
        loss2 = cls_eff_loss

        # #### add NER loss (effectiveness)
        # # loss2 = self.loss_fct2(eff_pred.reshape(-1, self.num_classes2), batch['labels_eff'].reshape(-1))
        # loss2 = self.get_ner_loss(logits_eff_raw, batch['labels_eff'], batch['attention_mask']) #logits_eff_word TODO try
        # loss2 = self.get_ner_loss(logits_eff_word, batch['labels_eff'], batch['attention_mask'])  

        # ### total loss 
        #loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2
        # loss = obj_loss + reg_loss + cls_loss + cls_eff_loss
        loss = obj_loss + reg_loss + cls_loss + self.loss_weights[1]*cls_eff_loss

        out = {}
        out['loss'] = loss
        out['loss1'] = loss1
        out['loss2'] = loss2
        # out['logits'] = logits_raw
        # out['logits_eff'] = logits_eff_raw
        out['logits_eff_word'] = logits_eff_word
        return out
    

    def forward_logits(self, data):
        ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
        bs = data['input_ids'].size(0)
        assert bs == 1, f'Only batch_size=1 supported, got batch_size={bs}.'
        
        ##### original 
        # outputs = self.model(input_ids=data['input_ids'],attention_mask=data['attention_mask'])
        # logits = outputs['logits']
        
        #### modify for AutoModel
        # outputs = self.model(data['input_ids'], data['attention_mask']) 
        outputs = self.model.deberta(data['input_ids'], data['attention_mask']) 
        # outputs = self.backbone(data['input_ids'], data['attention_mask']) 
        # pooled_out = outputs[0]
        x = outputs.last_hidden_state
        # x = self.dropout(x)

        # logits_type_raw = self.classifier(x)
        logits_type_raw = self.model.classifier(x)
         
        logits_type_word = aggregate_tokens_to_words(logits_type_raw, data['word_boxes'])
        assert logits_type_word.size(0) == data['text'].split().__len__()

        logits_eff_raw = self.head_eff(x) # effectiveness
        
        logits_eff_word = aggregate_tokens_to_words(logits_eff_raw, data['word_boxes'])
        assert logits_eff_word.size(0) == data['text'].split().__len__()

        obj_pred = logits_type_word[..., 0]
        reg_pred = logits_type_word[..., 1:3]
        cls_pred = logits_type_word[..., 3:]

        return dict(
            obj_pred=obj_pred, 
            reg_pred=reg_pred, 
            cls_pred=cls_pred, 
            logits_type_raw=logits_type_raw, 
            logits_eff_raw=logits_eff_raw, 
            logits_eff_word=logits_eff_word,
            hidden_state=x,
            )
    
    ## Loss functions 
    def get_losses(
            self, obj_pred, reg_pred, cls_pred, logits_eff_word, 
            obj_target, reg_target, cls_target, cls_eff_target, pos_loc, logits_eff_disc, cls_eff_target_disc):
        
        num_total_samples = pos_loc.numel()
        assert num_total_samples > 0
        
        reg_pred   = reg_pred[pos_loc].exp()
        reg_target = reg_target[pos_loc]
        
        px1 = pos_loc - reg_pred[:, 0]
        px2 = pos_loc + reg_pred[:, 1]
        gx1 = reg_target[:, 0]
        gx2 = reg_target[:, 1]
        ix1 = torch.max(px1, gx1)
        ix2 = torch.min(px2, gx2)
        ux1 = torch.min(px1, gx1)
        ux2 = torch.max(px2, gx2)
        inter = (ix2 - ix1).clamp(min=0)
        union = (ux2 - ux1).clamp(min=0) + 1e-12
        iou = inter / union

        # reg_loss = -iou.log().sum() / num_total_samples
        reg_loss = torch.sum(-torch.log(iou + 1e-12)) / num_total_samples
        # reg_loss = torch.nansum(-torch.log(iou + 1e-12)) / num_total_samples
        
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_pred[pos_loc],
            cls_target[pos_loc] * iou.detach().reshape(-1, 1),
            reduction='sum') / num_total_samples
        
        obj_loss = F.binary_cross_entropy_with_logits(
            obj_pred, obj_target, reduction='sum') / num_total_samples
        
        # cls_loss = cls_loss_bce

        ##### ++ Jaccard loss 
        # loss_fct_j = JaccardLoss()
        # cls_loss_j = loss_fct_j(cls_pred[pos_loc], cls_target[pos_loc] * iou.detach().reshape(-1, 1)) / num_total_samples

        # ### FOCAL Loss
        # gamma = 2.0
        # alpha = 1.0
        # loss_fct1 = nn.BCEWithLogitsLoss(reduction='none')
        # cls_bce_l1 = loss_fct1(cls_pred[pos_loc], cls_target[pos_loc] * iou.detach().reshape(-1, 1))
        # probas = torch.sigmoid(cls_pred[pos_loc])
        # cls_loss_j = torch.where(cls_target[pos_loc] >= 0.5, alpha * (1. - probas)**gamma * cls_bce_l1, probas**gamma * cls_bce_l1)
        # cls_loss_j = cls_loss_j.mean()
        # cls_loss = cls_loss_bce + cls_loss_j

        ###### ## TODO check run2
        cls_eff_loss_bce = F.binary_cross_entropy_with_logits(
            logits_eff_word[pos_loc],  
            cls_eff_target[pos_loc] * iou.detach().reshape(-1, 1), 
            reduction='sum') / num_total_samples

        # loss_fct = nn.CrossEntropyLoss(reduction='sum')
        # cls_eff_loss_ce = loss_fct(logits_eff_word[pos_loc].softmax(-1), cls_eff_target[pos_loc]) / num_total_samples
        # cls_eff_loss = F.cross_entropy(logits_eff_word[pos_loc].softmax(-1), cls_eff_target[pos_loc], reduction='sum') / num_total_samples

        # loss_fct_bce = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor([3.0]).cuda())
        # cls_eff_loss_bce = loss_fct_bce(
        #     logits_eff_word[pos_loc], 
        #     cls_eff_target[pos_loc] 
        #     ) / num_total_samples
        

        loss_fct = nn.CrossEntropyLoss(reduction='mean') # 'sum'
        # cls_eff_loss_ce = loss_fct(logits_eff_disc.softmax(-1), cls_eff_target) / num_total_samples
        cls_eff_loss_ce = loss_fct(logits_eff_disc.softmax(-1), cls_eff_target_disc)  
        # cls_eff_loss = cls_eff_loss_ce

        ### add BCE loss on pos + CE Loss on pooled outs
        cls_eff_loss = cls_eff_loss_ce + cls_eff_loss_bce

        # loss_fct2 = JaccardLoss()
        # cls_eff_loss_j = loss_fct2(logits_eff_word[pos_loc], cls_eff_target[pos_loc]) / num_total_samples
        
        # ### FOCAL Loss 
        # gamma = 2.0 
        # alpha = 1.0 
        # loss_fct2 = nn.BCEWithLogitsLoss(reduction='none')
        # bce_l1 = loss_fct2(logits_eff_word[pos_loc], cls_eff_target[pos_loc])
        # probas = torch.sigmoid(logits_eff_word[pos_loc])
        # cls_eff_loss_focal = torch.where(cls_eff_target[pos_loc] >= 0.5, alpha * (1. - probas)**gamma * bce_l1, probas**gamma * bce_l1)
        # cls_eff_loss_focal = cls_eff_loss_focal.mean()
        
        # cls_eff_loss = cls_eff_loss_bce + cls_eff_loss_focal

        return obj_loss, reg_loss, cls_loss, cls_eff_loss  

    def get_ner_loss(self, outputs, targets, attention_mask):
        # loss_fct = nn.CrossEntropyLoss()
        active_loss = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = self.loss_fct(active_logits, true_labels)
        return loss  
    
    
    @torch.no_grad()
    def build_target(self, gt_spans, obj_pred, reg_pred, cls_pred, cls_pred_eff):
        obj_target = torch.zeros_like(obj_pred)
        reg_target = torch.zeros_like(reg_pred)
        cls_target = torch.zeros_like(cls_pred)
        cls_eff_target = torch.zeros_like(cls_pred_eff)
        
        # first token as positive
        pos_loc = gt_spans[:, 0]
        obj_target[pos_loc] = 1
        reg_target[pos_loc, 0] = gt_spans[:, 0].half() if self.cfg.fp16 else gt_spans[:, 0].float()
        reg_target[pos_loc, 1] = gt_spans[:, 1].half() if self.cfg.fp16 else gt_spans[:, 0].float()
        cls_target[pos_loc, gt_spans[:, 2]] = 1
        
        ### ++
        if self.num_classes2==1:
            cls_eff_target[pos_loc, 0] = gt_spans[:, 3].half() if self.cfg.fp16 else gt_spans[:, 3].float()
        else:
            cls_eff_target[pos_loc, gt_spans[:, 3]] = 1

        # dynamically assign one more positive
        if self.dynamic_positive:
            
            # cls_prob = (obj_pred.sigmoid().unsqueeze(1) * cls_pred.sigmoid()).sqrt()
            cls_prob = (obj_pred.unsqueeze(1) * cls_pred).sqrt() ## for BCEwithlogits

            for start, end, label, label_eff in gt_spans:
                _cls_prob = cls_prob[start:end]
                _cls_gt = _cls_prob.new_full((_cls_prob.size(0), ),
                                             label,
                                             dtype=torch.long)
                _cls_gt = F.one_hot(
                    _cls_gt, num_classes=_cls_prob.size(1)).type_as(_cls_prob)
                
                # cls_cost = F.binary_cross_entropy(_cls_prob, _cls_gt, reduction='none').sum(-1)
                ## Try with BCELogits
                cls_cost = F.binary_cross_entropy_with_logits(_cls_prob,
                                                  _cls_gt,
                                                  reduction='none').sum(-1)


                _reg_pred = reg_pred[start:end].exp()
                _reg_loc = torch.arange(_reg_pred.size(0),
                                        device=_reg_pred.device)
                px1 = _reg_loc - _reg_pred[:, 0]
                px2 = _reg_loc + _reg_pred[:, 1]
                ix1 = torch.max(px1, _reg_loc[0])
                ix2 = torch.min(px2, _reg_loc[-1])
                ux1 = torch.min(px1, _reg_loc[0])
                ux2 = torch.max(px2, _reg_loc[-1])
                inter = (ix2 - ix1).clamp(min=0)
                union = (ux2 - ux1).clamp(min=0) + 1e-12
                iou = inter / union
                iou_cost = -torch.log(iou + 1e-12)
                cost = cls_cost + iou_cost

                pos_ind = start + cost.argmin()
                obj_target[pos_ind] = 1
                reg_target[pos_ind, 0] = start
                reg_target[pos_ind, 1] = end
                cls_target[pos_ind, label] = 1
            
                ## TODO try run2 
                cls_eff_target[pos_ind, label_eff] = 1
            
            pos_loc = (obj_target == 1).nonzero().flatten()
        
        return obj_target, reg_target, cls_target, cls_eff_target, pos_loc # 



    

    def training_step(self, batch, batch_idx):

        # out = self.shared_step(batch)
        res1 = self.forward_logits(batch)

        obj_target, reg_target, cls_target, cls_eff_target, pos_loc = self.build_target(
            batch['gt_spans'], res1['obj_pred'], res1['reg_pred'], res1['cls_pred'], res1['logits_eff_word'])
        
        # res2 = self.predict_text_span(batch, batch_idx) # start, end, score, pred_class
        # # if res2 is not None: print('[TRAIN] start/end', res2['start'], res2['end'])


        h = res1['hidden_state'].squeeze(0)  # (seq_len, hidden_size)
        if self.vb: print('[TRAIN] h', h.shape)

        # start_, end_ = res2['start'], res2['end']
        start_, end_ = batch['gt_spans'][:,0], batch['gt_spans'][:,1] 

        # logits_eff = []
        pooled_outs = []
        for st_, en_ in zip(start_, end_):
            # active_logits_eff_word = res1['logits_eff_word'][st_:en_].mean()
            # active_logits_eff_word = self.head_eff(h[st_:en_])                   # len_span x 2 
            # active_logits_eff_word = active_logits_eff_word.mean(0).unsqueeze(0) # 1x2

            pooled_outs.append(h[st_:en_].mean(0).unsqueeze(0))
            # logits_eff.append(active_logits_eff_word)
        
        if self.vb: 
            print('[TRAIN] len pooled_outs', len(pooled_outs))

        # logits_eff = torch.cat(logits_eff) 
        pooled_outs = torch.cat(pooled_outs)
        logits_eff = self.head_eff(pooled_outs)  
        if self.vb: print('[TRAIN] logits_eff', logits_eff.shape)

        eff_target = torch.tensor(batch['gt_spans'][:,-1])
        if self.vb: print('[TRAIN] eff_target', eff_target.shape)
        # print('[TRAIN] cls_eff_target word-level', cls_eff_target.shape)


        obj_loss, reg_loss, cls_loss, cls_eff_loss = self.get_losses(
            obj_pred=res1['obj_pred'],
            reg_pred=res1['reg_pred'],
            cls_pred=res1['cls_pred'],
            obj_target=obj_target,
            reg_target=reg_target,
            cls_target=cls_target,
            pos_loc=pos_loc,
            ###
            logits_eff_word=res1['logits_eff_word'], ## change: num words x num_classes2 (Nd x 2) 
            cls_eff_target=cls_eff_target,           ## change: num words x 1 
            ##
            logits_eff_disc=logits_eff,    ##  num_disc x num_classes2 (Nd x 2) 
            cls_eff_target_disc=eff_target, 
            )

        loss1 = obj_loss + reg_loss + cls_loss
        loss2 = cls_eff_loss
        loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2


        self.log("train_loss", loss, on_step=True, logger=True, prog_bar=True)
        self.log("train_obj_loss", obj_loss, on_step=True, logger=True, prog_bar=True)
        self.log("train_reg_loss", reg_loss, on_step=True, logger=True, prog_bar=True)
        self.log("train_cls_loss", cls_loss, on_step=True, logger=True, prog_bar=True)
        self.log("train_eff_loss", cls_eff_loss, on_step=True, logger=True, prog_bar=True)
        self.log("train_loss1", loss1, on_step=True, logger=True, prog_bar=True)
        # self.log("train_f1", f1, on_step=False, on_epoch=True, logger=True, prog_bar=False)
        
        return {
            'loss': loss, 
            'obj_loss': obj_loss,
            'reg_loss': reg_loss,
            'cls_loss': cls_loss,
            'seg_loss': loss1,
            'eff_loss': cls_eff_loss }
    

    def train_epoch_end(self,outputs):
        tr_avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tr_obj_loss = torch.stack([x['obj_loss'] for x in outputs]).mean()
        tr_reg_loss = torch.stack([x['reg_loss'] for x in outputs]).mean()
        tr_cls_loss = torch.stack([x['cls_loss'] for x in outputs]).mean()

        print(f'epoch {trainer.current_epoch} training losses: {tr_avg_loss:.4f}, obj:{tr_obj_loss:.4f}, reg:{tr_reg_loss:.4f}, cls:{tr_cls_loss:.4f}')
        return {'train_loss_epoch': tr_avg_loss} 


    def validation_step(self, batch, batch_idx):
        # if self.skip_validation > 0: return {}

        # out = self.shared_step(batch)
        res1 = self.forward_logits(batch)

        obj_target, reg_target, cls_target, cls_eff_target, pos_loc = self.build_target(
            batch['gt_spans'], res1['obj_pred'], res1['reg_pred'], res1['cls_pred'], res1['logits_eff_word'])
        
        # res2 = self.predict_text_span(batch, batch_idx) # start, end, score, pred_class
        # # if res2 is not None: print('[TRAIN] start/end', res2['start'], res2['end'])

        h = res1['hidden_state'].squeeze(0)  # (seq_len, hidden_size)
        if self.vb: print('[VAL] h', h.shape)

        # start_, end_ = res2['start'], res2['end']
        start_, end_ = batch['gt_spans'][:,0], batch['gt_spans'][:,1] 

        # logits_eff = []
        # for st_, en_ in zip(start_, end_):
        #     active_logits_eff_word = self.head_eff(h[st_:en_])                   # len_span x 2 
        #     active_logits_eff_word = active_logits_eff_word.mean(0).unsqueeze(0) # 1x2
        #     logits_eff.append(active_logits_eff_word)
        # logits_eff = torch.cat(logits_eff) 
        # logits_eff = torch.stack(logits_eff)
        
        pooled_outs = []
        for st_, en_ in zip(start_, end_):
            pooled_outs.append(h[st_:en_].mean(0).unsqueeze(0))        
        if self.vb: print('[VAL] len pooled_outs', len(pooled_outs))

        pooled_outs = torch.cat(pooled_outs)
        if self.vb: print('[VAL] pooled_outs', pooled_outs.shape)

        logits_eff = self.head_eff(pooled_outs)  

        eff_target = torch.tensor(batch['gt_spans'][:,-1])

        if self.vb: print('[VAL] logits_eff', logits_eff.shape)
        if self.vb: print('[VAL] eff_target', eff_target.shape)
        # print('[TRAIN] cls_eff_target word-level', cls_eff_target.shape)

        obj_loss, reg_loss, cls_loss, cls_eff_loss = self.get_losses(
            obj_pred=res1['obj_pred'],
            reg_pred=res1['reg_pred'],
            cls_pred=res1['cls_pred'],
            obj_target=obj_target,
            reg_target=reg_target,
            cls_target=cls_target,
            pos_loc=pos_loc,
            ###
            logits_eff_word=res1['logits_eff_word'], ## change: num words x num_classes2 (Nd x 2) 
            cls_eff_target=cls_eff_target,           ## change: num words x 1 
            ##
            logits_eff_disc=logits_eff,    ##  num_disc x num_classes2 (Nd x 2) 
            cls_eff_target_disc=eff_target, 
            )

        loss1 = obj_loss + reg_loss + cls_loss
        loss2 = cls_eff_loss
        loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2

        self.log("val_loss", loss, on_step=True, logger=True, prog_bar=True)
        self.log("val_obj_loss", obj_loss, on_step=True, logger=True, prog_bar=True)
        self.log("val_reg_loss", reg_loss, on_step=True, logger=True, prog_bar=True)
        self.log("val_cls_loss", cls_loss, on_step=True, logger=True, prog_bar=True)
        self.log("val_eff_loss", cls_eff_loss, on_step=True, logger=True, prog_bar=True)

        self.val_outs.append({
            "val_loss": loss, 
            "val_obj_loss": obj_loss,
            "val_reg_loss": reg_loss,
            "val_cls_loss": cls_loss,
            "val_eff_loss": cls_eff_loss,
            ###
            # "logits": res1['logits'], 
            # "logits_eff": res1['logits_eff_raw'], 
            'hidden_state': res1['hidden_state'].squeeze(0), 
            'obj_pred': res1['obj_pred'],
            'reg_pred': res1['reg_pred'],
            'cls_pred': res1['cls_pred'],
            ###
            'input_ids': batch['input_ids'], 
            'attention_mask': batch['attention_mask'], 
            'word_boxes': batch['word_boxes'], 
            'text': batch['text'], 
            'text_id': batch['text_id'], 
            })
        
        return {
            'loss': loss, 
            'obj_loss': obj_loss,
            'reg_loss': reg_loss,
            'cls_loss': cls_loss,
            'seg_loss': loss1,
            'eff_loss': cls_eff_loss }


    def on_validation_epoch_start(self):
        print('\n')


    def on_validation_epoch_end(self):

        outputs = self.val_outs
        # print('len outputs', len(outputs))
        
        ######################################################
            # out = self.predict_step(batch, i)
        ######################################################
        # val_loss = 0
        preds = []
        for i, o in enumerate(outputs):
            ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
            # batch = {'input_ids': o['input_ids'], 'attention_mask': o['attention_mask'], 'word_boxes': o['word_boxes'], 'text_id': o['text_id'], 'text': o['text']}
            # obj_pred, reg_pred, cls_pred, logits_eff_word, outputs = self.forward_logits(batch)
            # res1 = self.forward_logits(batch)

            obj_pred = o['obj_pred'].sigmoid() # res1['obj_pred'].sigmoid()
            reg_pred = o['reg_pred'].exp()
            cls_pred = o['cls_pred'].sigmoid()

            obj_scores = obj_pred
            cls_scores, cls_labels = cls_pred.max(-1)
            pr_scores = (obj_scores * cls_scores)**0.5
            
            pos_inds = pr_scores > self.test_score_thr
            
            if pos_inds.sum() == 0:
                return None

            pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
            pos_loc = pos_inds.nonzero().flatten()
            start = pos_loc - reg_pred[pos_inds, 0]
            end = pos_loc + reg_pred[pos_inds, 1]

            min_idx, max_idx = 0, obj_pred.numel() - 1
            start = start.clamp(min=min_idx, max=max_idx).round().long()
            end = end.clamp(min=min_idx, max=max_idx).round().long()

            # nms
            keep = span_nms(start, end, pr_score)
            start = start[keep]
            end = end[keep]
            pr_score = pr_score[keep]
            pr_label = pr_label[keep]

            # ### incl effectiveness preds --> TODO try both 
            # if self.use_sigmoid:
            #     logits_eff_word = logits_eff_word.sigmoid()         ## run 1 (2 classes)
            #     eff_proba, eff_cls = logits_eff_word.max(-1) 

            #     # eff_proba = eff_proba[keep]
            #     eff_proba = logits_eff_word[keep]
            #     eff_cls   = eff_cls[keep]

            #     # eff_proba_0 = eff_proba_1 = eff_proba
            #     eff_proba_1 = eff_proba
            #     eff_proba_0 = 1.0 - eff_proba_1

            # else:
            #     ### use softmax 
            #     logits_eff_word = torch.softmax(logits_eff_word, -1)  ## run 2 (2 classes)

            #     _, eff_cls = logits_eff_word.max(-1) 
            #     eff_proba_1 = logits_eff_word[keep, 1] ## positive class
            #     eff_proba_0 = logits_eff_word[keep, 0] ## negative class
            #     # eff_proba = eff_proba[keep]
            #     eff_proba = eff_proba_1
            #     eff_cls   = eff_cls[keep]

            # # eff_proba, eff_cls = logits_eff_word[start:end].max(-1) 

            h = o['hidden_state'].squeeze(0)  # (seq_len, hidden_size)

            # logits_eff = []
            # for st_, en_ in zip(start, end):
            #     # active_logits_eff_word = res1['logits_eff_word'][st_:en_].mean()
            #     active_logits_eff_word = self.head_eff(h[st_:en_])                   # len_span x 2 
            #     active_logits_eff_word = active_logits_eff_word.mean(0).unsqueeze(0) # 1x2
            #     logits_eff.append(active_logits_eff_word)
            # logits_eff = torch.cat(logits_eff) 

            pooled_outs = []
            for st_, en_ in zip(start, end):
                pooled_outs.append(h[st_:en_].mean(0).unsqueeze(0))        
            if self.vb: 
                print('[VAL] len pooled_outs list', len(pooled_outs))
                for j in range(len()):
                    print(f'[VAL] pooled_out list {j}-th', pooled_outs[j].shape )

            pooled_outs = torch.cat(pooled_outs)
            if self.vb: 
                print('[VAL] pooled_outs before head', pooled_outs.shape)
            logits_eff = self.head_eff(pooled_outs)  

            logits_eff_pred = logits_eff.softmax(-1)
            _, eff_cls = logits_eff_pred.max(-1) 
            eff_proba_0 = logits_eff_pred[:, 0] ## negative class
            eff_proba_1 = logits_eff_pred[:, 1] ## positive class

            # ## TODO in train try calibrate pred by obj 
            # eff_proba_0 = (obj_scores[pos_inds][keep] * logits_eff_pred[:, 0])**0.5
            # eff_proba_1 = (obj_scores[pos_inds][keep] * logits_eff_pred[:, 1])**0.5

            del h 
            torch.cuda.empty_cache()
            gc.collect()

            out = dict(
                text_id=o['text_id'],
                start=to_np(start),
                end=to_np(end),
                score=to_np(pr_score),
                label=to_np(pr_label),
                ##### ++ 
                label_eff=to_np(eff_cls),
                score_eff_0=to_np(eff_proba_0),
                score_eff_1=to_np(eff_proba_1),
                # score_eff=to_np(eff_proba_1),
                )
            ###########################
            # test_ps = self.result_to_ps(out) # print('test_ps', test_ps)       
            test_ps = format_result(out)        
            preds.append(test_ps)  

        ### pred_df
        preds = pd.concat(preds)
        # print('pred_df', preds)

        valid_df_ = self.valid_df.copy()
        del valid_df_['discourse_effectiveness']
        valid_df_['discourse_effectiveness'] = valid_df_['discourse_effectiveness_int']
        if "Lead" in valid_df_['discourse_type'].unique():
            valid_df_['discourse_type'] = valid_df_['discourse_type'].map(TYPE2LABEL)

        # avg_f1 = get_f1_score(preds, self.valid_df, slient=False)
        # avg_f1 = get_f1_score_v1(preds, self.valid_df, slient=False)
        scores_dict = compute_scores(preds, valid_df_, ids_selected=list(valid_df_.id.unique()), threshold=0.5, return_segments=False, return_scores=False)
        avg_f1 = scores_dict['F1_double_w']
        
        ### reset placeholder 
        self.val_outs = []   
        gc.collect() 

        if avg_f1>self.best_score:
            self.best_score = avg_f1 
        
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # val_loss = val_loss / len(outputs)

        
        self.log(f'val_f1_avg', avg_f1, on_epoch=True, logger=True, prog_bar=True)
        self.log(f'val_f1_sw', scores_dict['F1_single_w'], on_epoch=True, logger=True, prog_bar=True)
        self.log(f'val_f1_fp1', scores_dict['F1_FB1'], on_epoch=True, logger=True, prog_bar=True)
        # self.log(f'val_loss_epoch', val_loss, on_epoch=True)

        print(f'epoch {trainer.current_epoch} validation f1: {avg_f1} (best f1: {self.best_score:.4f})')
        print(f'epoch {trainer.current_epoch} validation f1_sw: {scores_dict["F1_single_w"]}')
        print(f'epoch {trainer.current_epoch} validation f1_fp1: {scores_dict["F1_FB1"]}')
        print(f'epoch {trainer.current_epoch} validation loss: {val_loss:.4f}')
        
        return {'val_loss_epoch': val_loss, 'val_f1_avg': avg_f1}
    

    ###  predict loop
    def predict_text_span(self, batch, batch_idx):
        ######################################################
        ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
        # obj_pred, reg_pred, cls_pred, logits_eff_word, _ = self.forward_logits(batch)
        res1 = self.forward_logits(batch)
        
        obj_pred = res1['obj_pred'].sigmoid()
        reg_pred = res1['reg_pred'].exp()
        cls_pred = res1['cls_pred'].sigmoid()

        # print('[PREDICT] obj_pred', obj_pred.shape) # Num words 

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        pr_scores = (obj_scores * cls_scores)**0.5
        
        pos_inds = pr_scores > self.test_score_thr ## TODO try thres per class 
         

        if pos_inds.sum() == 0:
            return None

        pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score, nms_thr=self.nms_thr)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]

        return dict(
            text_id=batch['text_id'],
            start=to_np(start),
            end=to_np(end),
            score=to_np(pr_score),
            label=to_np(pr_label),
            )


    ###  predict loop 
    def predict_step(self, batch, batch_idx):

        ### ------------------------
        # res2 = self.predict_text_span(batch, batch_idx)
        ### -------------------------
        ######################################################
        ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
        # obj_pred, reg_pred, cls_pred, logits_eff_word, _ = self.forward_logits(batch)
        res1 = self.forward_logits(batch)
        
        obj_pred = res1['obj_pred'].sigmoid()
        reg_pred = res1['reg_pred'].exp()
        cls_pred = res1['cls_pred'].sigmoid()

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        pr_scores = (obj_scores * cls_scores)**0.5
        
        pos_inds = pr_scores > self.test_score_thr ## TODO try thres per class 

        if pos_inds.sum() == 0:
            return None

        pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score, nms_thr=self.nms_thr)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]

        # ### incl effectiveness preds (method A) ##  
        # if self.use_sigmoid:
        #     logits_eff_word = logits_eff_word.sigmoid()         ## run 1 (2 classes)
        #     eff_proba, eff_cls = logits_eff_word.max(-1) 

        #     eff_proba = eff_proba[keep]
        #     eff_cls   = eff_cls[keep]

        #     # eff_proba_0 = eff_proba_1 = eff_proba
        #     eff_proba_0 = logits_eff_word[keep, 0] ## negative class
        #     eff_proba_1 = logits_eff_word[keep, 1] ## positive class

        # else:
        #     ### use softmax 
        #     logits_eff_word = torch.softmax(logits_eff_word, -1)  ## run 2 (2 classes)

        #     _, eff_cls = logits_eff_word.max(-1) 
        #     eff_proba_1 = logits_eff_word[keep, 1] ## positive class
        #     eff_proba_0 = logits_eff_word[keep, 0] ## negative class
        #     # eff_proba = eff_proba[keep]
        #     eff_proba = eff_proba_1
        #     eff_cls   = eff_cls[keep]
        ###########################################################
        # (method B) 
        # # eff_proba, eff_cls = logits_eff_word[start:end].max(-1) 
        # eff_proba = []
        # eff_cls = []
        # for st_, en_ in zip(start, end):
        #     ## apply first mean then max
        #     active_logits_eff_word = logits_eff_word[st_:en_].mean()
        #     eff_proba_, eff_cls_ = active_logits_eff_word.max(-1) 

        #     eff_proba.append(eff_proba_.detach().cpu().numpy())
        #     eff_cls.append(eff_cls_.detach().cpu().numpy())
        ############################################################
        # (method C) NEW

        h = res1['hidden_state'].squeeze(0)  # (seq_len, hidden_size)

        pooled_outs = []
        for st_, en_ in zip(start, end):
            pooled_outs.append(h[st_:en_].mean(0).unsqueeze(0))        

        pooled_outs = torch.cat(pooled_outs)
        logits_eff = self.head_eff(pooled_outs)  

        logits_eff_pred = logits_eff.softmax(-1)
        _, eff_cls = logits_eff_pred.max(-1) 
        
        eff_proba_0 = logits_eff_pred[:, 0] ## negative class
        eff_proba_1 = logits_eff_pred[:, 1] ## positive class

        # eff_proba_0 = (obj_scores[pos_inds][keep] * logits_eff_pred[:, 0])**0.5
        # eff_proba_1 = (obj_scores[pos_inds][keep] * logits_eff_pred[:, 1])**0.5


        del h 
        torch.cuda.empty_cache()
        gc.collect()


        return dict(
            text_id=batch['text_id'],
            start=to_np(start),
            end=to_np(end),
            score=to_np(pr_score),
            label=to_np(pr_label),
            # ###
            label_eff=to_np(eff_cls),
            score_eff_0=to_np(eff_proba_0),
            score_eff_1=to_np(eff_proba_1),
            score_eff=to_np(eff_proba_1),
            )


    def configure_optimizers(self):

        weight_decay = self.cfg.WD # self.cfg.training.weight_decay #self.cfg['optimizer']['weight_decay']
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
                # 'params': [p for n, p in param_optimizer if 'backbone' in n and not any(nd in n for nd in no_decay)],
                'params': [p for n, p in param_optimizer if 'deberta' in n and not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': self.cfg.LR, # self.cfg['optimizer']['params']['lr'],
            },
            {
                # 'params': [p for n, p in param_optimizer if 'backbone' in n and any(nd in n for nd in no_decay)],
                'params': [p for n, p in param_optimizer if 'deberta' in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.cfg.LR, #self.cfg['optimizer']['params']['lr'],
            },
            {
                # 'params': [p for n, p in param_optimizer if not 'backbone' in n and not any(nd in n for nd in no_decay)],
                'params': [p for n, p in param_optimizer if not 'deberta' in n and not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                # 'lr': self.cfg['optimizer']['params']['lr'] * self.cfg['optimizer']['head_lr_factor'],
                'lr': self.cfg.LR * 5.0,
            },
            {
                # 'params': [p for n, p in param_optimizer if not 'backbone' in n and any(nd in n for nd in no_decay)],
                'params': [p for n, p in param_optimizer if not 'deberta' in n and any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                # 'lr': self.cfg['optimizer']['params']['lr'] * self.cfg['optimizer']['head_lr_factor'],
                'lr': self.cfg.LR * 5.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.cfg.LR, eps = 1.0e-06, betas = [0.9, 0.999]) 

        # scheduler = get_scheduler_cos(self.cfg, optimizer) 
        # scheduler = get_hf_scheduler(self.cfg, optimizer, self.cfg.num_train_steps)  
        scheduler = self.fetch_scheduler(optimizer)
   
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
        return [optimizer], [scheduler]
    

    def fetch_scheduler(self, optimizer):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_ratio * self.cfg.num_train_steps),
            num_training_steps=self.cfg.num_train_steps,
            num_cycles=0.5,
            last_epoch=-1,
        )
        return scheduler


    def result_to_ps(self, out):

        start_list = out['start']
        end_list = out['end']
        scores_list = out['score']
        labels_list = out['label']
        txt_id = out['text_id']

        scores_list2 = out['score_eff'] if 'score_eff' in out.keys() else np.zeros(len(start_list))
        labels_list2 = out['label_eff'] if 'label_eff' in out.keys() else np.zeros(len(start_list))

        scores_list2_0 = out['score_eff_0'] if 'score_eff_0' in out.keys() else np.zeros(len(start_list))
        scores_list2_1 = out['score_eff_1'] if 'score_eff_1' in out.keys() else np.zeros(len(start_list))

        res = []
        for start, end, score, label, score2, label2, score2_0, score2_1 in zip(start_list, end_list, scores_list, labels_list, scores_list2, labels_list2, scores_list2_0, scores_list2_1):
            res.append({
                'id': txt_id,
                'start': start, 
                'end': end, 
                'predictionstring': " ".join([str(i) for i in range(start, end+1)]),
                'score': score,
                'class': label,
                ###
                'class_eff': label2,
                'score_eff': score2,
                'score_eff_0': score2_0,
                'score_eff_1': score2_1,
            })
        res = pd.DataFrame(res)
        res['discourse_type'] = res['class']        # int 
        res['class'] = res['class'].map(LABEL2TYPE) # str 

        ### NEW
        res['score_discourse_effectiveness_0'] = res['score_eff_0'].copy()
        res['score_discourse_effectiveness_1'] = res['score_eff_1'].copy()
        return res
    

    def train_dataloader(self):
        return self._train_dataloader 
    

    def validation_dataloader(self):
        return self._valid_dataloader





# sample_ds = SpanDetectorDataset(valid_df.iloc[:100], tokenizer=CFG.tokenizer)
# collate = CustomCollator(CFG.tokenizer, CFG.model_name) #detector.model

# sample_loader = DataLoader(sample_ds,
#                       batch_size=1,
#                       shuffle=True,
#                       num_workers=2,
#                       collate_fn=collate)

# # bs = next(iter(sample_loader))

# args2 = CFG
# args2.fp16 = False  

# model = TextSpanDetectorPL_FP1(CFG.model_name, cfg=args2, train_df=train_df, valid_df=valid_df, predict_df=valid_df, vb=True)



# with torch.no_grad():
#     # o = model(batch)
#     obj_pred, reg_pred, cls_pred, logits_eff_word, _ , _= model.forward_logits(bs)

#     obj_target, reg_target, cls_target, cls_eff_target, pos_loc = model.build_target(
#             bs['gt_spans'], obj_pred, reg_pred, cls_pred, logits_eff_word)

#     obj_loss, reg_loss, cls_loss, cls_eff_loss = model.get_losses(
#             obj_pred, reg_pred, cls_pred, logits_eff_word,
#             obj_target, reg_target, cls_target, cls_eff_target, 
#             pos_loc
#             )
    
# print(obj_loss, reg_loss, cls_loss, cls_eff_loss)
# # # (tensor(1.3768), tensor(0.2222), tensor(0.5043), tensor(1.5836))
# # #, jacc_loss
# # # o['logits'].shape, o['logits_eff'].shape 

# # # with torch.cuda.amp.autocast():
# # #     o = model(batch)


# batch = next(iter(train_loader))
# out = model.training_step(batch, 1)

# batch = next(iter(valid_loader))
# out = model.validation_step(batch, 1)
 
# val_out = model.on_validation_epoch_end()




def format_result_test(out):

        start_list = out['start']
        end_list = out['end']
        scores_list = out['score']
        labels_list = out['label']
        txt_id = out['text_id']

        scores_list2 = out['score_eff'] if 'score_eff' in out.keys() else np.zeros(len(start_list))
        labels_list2 = out['label_eff'] if 'label_eff' in out.keys() else np.zeros(len(start_list))

        scores_list2_0 = out['score_eff_0'] if 'score_eff_0' in out.keys() else np.zeros(len(start_list))
        scores_list2_1 = out['score_eff_1'] if 'score_eff_1' in out.keys() else np.zeros(len(start_list))

        res = []
        for start, end, score, label, score2, label2, score2_0, score2_1 in zip(start_list, end_list, scores_list, labels_list, scores_list2, labels_list2, scores_list2_0, scores_list2_1):
            res.append({
                'id': txt_id,
                'start': start, 
                'end': end, 
                'predictionstring': " ".join([str(i) for i in range(start, end+1)]),
                'score': score,
                'class': label,
                ###
                'class_eff': label2,
                'score_eff': score2,
                'score_eff_0': score2_0,
                'score_eff_1': score2_1,
            })
        res = pd.DataFrame(res)
        res['discourse_type'] = res['class']        # int 
        res['class'] = res['class'].map(LABEL2TYPE) # str 

        ### NEW
        res['score_discourse_effectiveness_0'] = res['score_eff_0'].copy()
        res['score_discourse_effectiveness_1'] = res['score_eff_1'].copy()
        
        return res


 


 
# CFG.debug = True 
# CFG.debug = False 
CFG.fp16 = False 
CFG.LR = 2e-5

 
# wandb.finish()
 

 
for fold_ in CFG.trn_folds:

    train_df, valid_df = get_data(fold=fold_)
    # log('train.shape:', train_df.shape,'\t','test.shape:', valid_df.shape)
    valid_df['discourse_effectiveness_int'] = valid_df.discourse_effectiveness.map(ner_labels_eff)


    train_ds = SpanDetectorDataset(train_df, tokenizer=CFG.tokenizer, mask_prob=0.5, mask_ratio=0.2)
    valid_ds = SpanDetectorDataset(valid_df, tokenizer=CFG.tokenizer, mask_prob=0.0, mask_ratio=0.0)

    train_loader = DataLoader(train_ds,
                        batch_size=1,
                        shuffle=True,
                        num_workers=2,
                        collate_fn=collate)

    valid_loader = DataLoader(valid_ds,
                        batch_size=1,
                        shuffle=True,
                        num_workers=2,
                        collate_fn=collate)
    

    # model = TextSpanDetectorPL(CFG.model_name, cfg=CFG, train_df=train_df, valid_df=valid_df, predict_df=valid_df)
    model = TextSpanDetectorPL_FP1(
        CFG.model_name, 
        cfg=CFG, 
        train_df=train_df, valid_df=valid_df, predict_df=valid_df,
        test_score_thr=0.5,
        )


    # es = EarlyStopping(
    #         # monitor = 'score',
    #         patience = 2,  
    #         mode = 'min',
    #     )

    ckp = callbacks.ModelCheckpoint(
        dirpath=OUTPUT / f'weights_f{fold_}',
        filename='checkpoint' + '-{epoch}-{val_loss:.4f}',
        # filename='checkpoint' + '-{epoch}-{val_f1_avg:.4f}',
        save_weights_only=True,
        save_top_k=10, #2, #1,
        save_last=False,
        mode='min', monitor='val_loss_epoch',
        # mode='max', monitor='val_f1_avg',
    )
    # callback_list.append(ckp)
    callback_list = [ckp] # [es]

    logger_list = []
    tb_logger = TensorBoardLogger(save_dir=OUTPUT / "logs/")
    # logger_list = [tb_logger]
    if CFG.wandb: 
        logger_list.append(WandbLogger(project=CFG.wb_project))
    

    trainer = pl.Trainer(
        max_epochs=5, # if CFG.debug else CFG.epochs,
        callbacks=callback_list,
        logger=logger_list,
        accelerator='gpu',
        accumulate_grad_batches = 4, #CFG.GRAD_ACC, 
        precision = 16 if CFG.fp16 else 32,
        val_check_interval = 0.5, #200 if CFG.debug else 0.5, #1.0 
        fast_dev_run = False,
        num_sanity_val_steps = 1, # if CFG.debug else 0,
        enable_checkpointing=True, # if CFG.gradient_checkpointing else None,
        gradient_clip_val = 1.0, #10, 
        # resume_from_checkpoint = null,
    )
    
    trainer.fit(model, train_loader, valid_loader)



# %%
### FP1 6th logs 
# loss: 3.0703, obj_loss: 1.5883, reg_loss: 0.4492, cls_loss: 1.0328, grad_norm: 13.0539

# %%
# ### RESUME TRAIN for 5 epochs 

# trainer = pl.Trainer(
#     max_epochs=5, # if CFG.debug else CFG.epochs,
#     callbacks=callback_list,
#     logger=logger_list,
#     accelerator='gpu',
#     accumulate_grad_batches = 1,  #CFG.GRAD_ACC 
#     precision = 16 if CFG.fp16 else 32,
#     val_check_interval = 0.5, #200 if CFG.debug else 0.5, #1.0 
#     fast_dev_run = False,
#     num_sanity_val_steps = 1 if CFG.debug else 0,
#     enable_checkpointing=True if CFG.gradient_checkpointing else None,
#     # gradient_clip_val = 10, 
#     # resume = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp31/weights/checkpoint_f0-epoch=0-val_loss=4.8259.ckpt",
#     # strategy=""
#     # devices=[0], #'auto',
# )

# model = TextSpanDetectorPL.load_from_checkpoint(
#     "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp32/weights/checkpoint_f0-epoch=0-val_loss=5.1559.ckpt",
#     model_name=CFG.model_name,
#     cfg=CFG, train_df=train_df, valid_df=valid_df, predict_df=valid_df)

# model.test_score_thr = 0.3 

# trainer.fit(model, train_loader, valid_loader)

# %%
# MDL_PATH = [
#     "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/weights_f0_run1/checkpoint-epoch=1-val_loss=8.4533.ckpt",
#     "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/weights_f0_run1/checkpoint-epoch=0-val_loss=6.8787.ckpt",
#     "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/weights_f0_run1/checkpoint-epoch=0-val_loss=6.9579.ckpt",
# ]

# MDL_PATH = [
#     "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/weights_f0/checkpoint-epoch=0-val_loss=5.9979.ckpt",
#     "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/weights_f0/checkpoint-epoch=1-val_loss=6.3044.ckpt",
#     "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/weights_f0/checkpoint-epoch=1-val_loss=6.3102.ckpt",
#     "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/weights_f0/checkpoint-epoch=2-val_loss=6.3758.ckpt"
# ]

# # %%
# trainer = pl.Trainer(
#     max_epochs=1, # if CFG.debug else CFG.epochs,
#     accelerator='gpu', #'cpu', #'gpu',
#     accumulate_grad_batches = CFG.GRAD_ACC ,   
#     precision = 16, #16 if CFG.fp16 else 32,
#     enable_checkpointing=True if CFG.gradient_checkpointing else None,
# )


# model = TextSpanDetectorPL_FP1.load_from_checkpoint(
#     MDL_PATH[0],
#     model_name=CFG.model_name,
#     cfg=CFG, train_df=train_df, valid_df=valid_df, predict_df=valid_df,
#     use_sigmoid=False, test_score_thr = 0.5  
#     )

# # # model.test_score_thr = 0.3 
# # # model.test_score_thr = 0.4 
# # model.test_score_thr = 0.5 

# # %%
# # res = trainer.validate(model, valid_loader,)

# # %%
# print('thres', model.test_score_thr)

# res = trainer.predict(model, valid_loader, return_predictions=True)  

# # %%
# res[0]

# # %%
# pred_df = pd.concat([format_result(res[i]) for i in range(len(res)) if res[i] is not None])

# # %%
# pred_df

# # %%
# metrics = get_f1_score(pred_df, valid_df, slient=False)
# # metrics = get_f1_score_v1(pred_df, valid_df, slient=False)

# # %%
# pred_df.id.nunique()

# # %%
# MDL_PATH[0]

# # %%
# # weights_f0/checkpoint-epoch=0-val_loss=5.9979.ckpt

# # Overall f1 score 0.7481760485625549
# # Overall total score:	 0.6886354020790534

# # Overall logloss (TP) score:	 0.6290947555955518
# # Overall auc (TP) score:	 0.7643433085673751

# # %%
# OUTPUT

# # %%
# # "exp36_base/weights_f0/checkpoint-epoch=1-val_loss=8.4533.ckpt",
# # Claim                f1 score:	0.6083878263434781
# # Concluding Statement f1 score:	0.818746470920384
# # Counterclaim         f1 score:	0.4059040590405904
# # Evidence             f1 score:	0.7141497284282434
# # Lead                 f1 score:	0.7695724597445863
# # Position             f1 score:	0.638349885408709
# # Rebuttal             f1 score:	0.3418803418803419

# # Overall f1 score 	 0.6138558245380475
# # TP: 240, FP: 292, FN: 632
# # pre: 0.45112781954887216, rec: 0.27522935779816515


# # "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/weights_f0/checkpoint-epoch=0-val_loss=6.8787.ckpt",
# # TP: 279, FP: 180, FN: 593
# # pre: 0.6078431372549019, rec: 0.31995412844036697

# # Overall f1 score 0.691698265313035


# pred_df.to_csv(OUTPUT / 'val_pred_fold2_th05_7774.csv', index=False)


# # %%
# import matplotlib.pyplot as plt 
# import seaborn as sns 

# # create a histogram plot of 'score' per discourse_type
# plt.figure(figsize=(12, 6))
# sns.histplot(data=pred_df, x='score', hue='discourse_type', bins=50, kde=True)
# plt.title('Scores per Discourse Type')

# # same as above in subplot per discourse_type
# plt.figure(figsize=(14, 6))
# for i, dt in enumerate(pred_df['discourse_type'].unique()):
#     plt.subplot(2, 4, i+1)
#     sns.histplot(data=pred_df[pred_df['discourse_type']==dt], x='score', bins=50, kde=True)
#     plt.title(f'{dt}')


# # %%
# import seaborn as sns 

# sns.histplot(data=pred_df, x='score_eff_0', bins=50, kde=True)
# sns.histplot(data=pred_df, x='score_eff_1', bins=50, kde=True)


# %% [markdown]
# # Infer Test

class SpanDetectorDatasetTest(Dataset):
    def __init__(self, df, tokenizer,):
        
        self.df = df.copy() #pd.read_csv(csv_file)
        self.samples = list(self.df.groupby('id'))
        self.tokenizer = tokenizer
        print(f'Loaded {len(self)} samples.')


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text_id, df = self.samples[index]
        
        # text_path = os.path.join(self.text_dir, f'{text_id}.txt')
        # with open(text_path) as f:
        #     text = f.read().rstrip()
        text = df['essay_text'].values[0].rstrip()

        tokens = self.tokenizer(text, return_offsets_mapping=True)
        
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
        offset_mapping = self.strip_offset_mapping(text, offset_mapping)
        num_tokens = len(input_ids)

        # token slices of words
        woff = self.get_word_offsets(text)
        toff = offset_mapping
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1 + 1e-12)
        assert (ious > 0).any(-1).all()

        word_boxes = []
        for row in ious:
            inds = row.nonzero()[0]
            word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
        word_boxes = torch.FloatTensor(word_boxes)


        return dict(
            text=text,
            text_id=text_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_boxes=word_boxes)
    

    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)

    def get_word_offsets(self, text):
        matches = re.finditer("\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)
    


class CustomCollatorTest(object):
    def __init__(self, tokenizer, model_name): #model
        self.pad_token_id = tokenizer.pad_token_id

        config_model = config = AutoConfig.from_pretrained(model_name)
        # if hasattr(model.config, 'attention_window'):
        if hasattr(config_model, 'attention_window'):
            # For longformer
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/longformer/modeling_longformer.py#L1548
            self.attention_window = (
                config_model.attention_window if isinstance(
                    config_model.attention_window, int)
                else max(config_model.attention_window)
                )
        else:
            self.attention_window = None

    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
        if self.attention_window is not None:
            attention_window = self.attention_window
            padded_length = (attention_window -
                             max_seq_length % attention_window
                             ) % attention_window + max_seq_length
        else:
            padded_length = max_seq_length

        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)
 
        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']
 
        text_id = sample['text_id']
        text = sample['text']
        word_boxes = sample['word_boxes']
        # gt_spans = sample['gt_spans']

        return dict(
            text_id=text_id,
            text=text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_boxes=word_boxes,
            )
    


def generate_submission(df_sub, save_name=None, out_dir=OUTPUT, verbose=False):

    pred_df = df_sub.copy()
    print(pred_df.shape)
    print(pred_df.id.nunique())
    display(pred_df)

    # ## convert effective scores to RATER format 
    # pred_df['score_discourse_effectiveness_0'] = -100
    # pred_df['score_discourse_effectiveness_1'] = -100
    # for i, row in pred_df.iterrows():
    #     if row['class_eff'] == 2: #'Effective':  
    #         pred_df.loc[i, 'score_discourse_effectiveness_1'] = row['score_eff']
    #         pred_df.loc[i, 'score_discourse_effectiveness_0'] = 1 - row['score_eff']
    #     else:
    #         pred_df.loc[i, 'score_discourse_effectiveness_0'] = row['score_eff']
    #         pred_df.loc[i, 'score_discourse_effectiveness_1'] = 1 - row['score_eff']


    sample_sub = pd.read_csv(DIR / 'sample_submission.csv')

    sub = pred_df.copy()
    sub = sub.rename(columns={'id': 'essay_id_comp'})

    sub['score_discourse_effectiveness_0'] = pred_df['score_eff_0']
    sub['score_discourse_effectiveness_1'] = pred_df['score_eff_1']


    sub["discourse_type"] = sub["class"].map(discourse_type_to_int)
    # sub["discourse_type"] = sub["discourse_type"].map(discourse_type_to_int)

    sub = sub[sample_sub.columns]
    # if sub.essay_id_comp.nunique() == 10402:
    #     sub.to_csv(f'{out_dir}/{save_name}.csv', index=False)
    # else:
    #     print("check ids", sub.essay_id_comp.nunique())

    display(sub.head())
    print(sub.essay_id_comp.nunique() == 10402)
    
    ###
    if verbose:
        print(pred_df['class'].value_counts())
        print()
        print(pred_df['class_eff'].value_counts())
        print(pred_df['class_eff'].value_counts(normalize=True))
        print(sub.dtypes)

        import seaborn as sns 
        import matplotlib.pyplot as plt

        sns.histplot(sub.score_discourse_effectiveness_0)
        sns.histplot(sub.score_discourse_effectiveness_1)
        plt.show()

    return sub


# CFG.infer = True


if CFG.infer:

    test = pd.read_csv('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/test.csv')
    test.columns = ['id', 'essay_text']
    test['essay_id'] = test['id'] 

    # print(test.shape)
    # display(test.head())

    test_ds = SpanDetectorDatasetTest(test, tokenizer=CFG.tokenizer)
    print(test_ds[0].keys())
    
    test_loader = DataLoader(test_ds,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=CustomCollatorTest(CFG.tokenizer, CFG.model_name)
                            )


# model.test_score_thr = 0.5
# model.test_score_thr = 0.4


if CFG.infer:

    print('test th:', model.test_score_thr)

    res = trainer.predict(model, test_loader, return_predictions=True)  
    pred_df = pd.concat([format_result(res[i]) for i in range(len(res)) if res[i] is not None])

    ####
    # sub = generate_submission(pred_df, test)



# pred_df.rename(columns={
#     "score_eff_0": "score_discourse_effectiveness_0",
#     "score_eff_1": "score_discourse_effectiveness_1",
# }).to_csv(OUTPUT / f"test_preds_fold2_th05.csv", index=False)



# sub = generate_submission(pred_df)

 
# ### Check missing ids 
# missing_ids = list(set(test.id) - set(sub.essay_id_comp))
# print(missing_ids)

# if len(missing_ids)>0:

#     dummy_sub = pd.read_csv('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/submissions/sub_wbf_exp18_exp22_6ckp.csv')
#     dummy_sub = dummy_sub[dummy_sub.essay_id_comp.isin(missing_ids)]
#     dummy_sub = dummy_sub.reset_index(drop=True)
#     display(dummy_sub)

#     sub = pd.concat([sub, dummy_sub], ignore_index=True)

#     print(sub.essay_id_comp.nunique() == 10402)

#     # ['DEC69C9D9F72', '93E547BF9654', 'CCD8783CF958', '261DCD26AE0C', 'E8C25EDACD20', '178BED4E1487', '7572D2FA0521', '7EEF18A458A3']

# # %%
# import seaborn as sns 

# sns.histplot(data=sub, x='score_discourse_effectiveness_0', bins=50, kde=True)
# sns.histplot(data=sub, x='score_discourse_effectiveness_1', bins=50, kde=True)

# # %%
# sub['discourse_type'].value_counts()

# # %%
# sub.isnull().sum()

# # %%
# OUTPUT

# # %%
# # # sub.to_csv(OUTPUT / "sub_fold2_th04_2class_soft_f1_7715.csv", index=False)
# # # sub.to_csv(OUTPUT / "sub_fold2_th05_2class_soft_f1_7774.csv", index=False)
# # sub.to_csv(OUTPUT / "sub_fold0_th04.csv", index=False)

# sub.to_csv(OUTPUT / f"sub_fold0_th05_ckp0.csv", index=False)

