# %%
# ## Setup
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

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


import wandb
import warnings 
warnings.filterwarnings('ignore')
torch.set_warn_always(False)

print('torch.__version__:', torch.__version__)
print('transformers.__version__:', transformers.__version__)


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # Config

# %%
class CFG:
    exp_name = 'exp66' 
    task = "ner"
    n_folds = 5 # 10
    trn_folds = [0] #[0] #[1] #[3] #[2] #[1] #[0] # [0,1,2,3,4]
    # model_name = "microsoft/deberta-large"
    model_name = "/workspace/checkpoints/hf_pretrained/deberta-xlarge"
    pretrained_detector_path = '/workspace/checkpoints/tascj/deberta_xlarge_fold0.pth' # 'none'
    pretrained_path = 'none'
    max_length = 1536 #768  
    stride = 128
    # TRAINING HYPERPARAMS
    batch_size = 1 #32 #4       # 16
    epochs = 5 #4 #1
    gradient_checkpointing = True 
    scheduler = 'cosine'
    LR = 4e-5  #6e-5 #2e-5    # 4e-5    ## 3e-05
    WD = 0.01
    WARMUP = 0.1
    use_custom_trainer = False # set True for custom training loop - TODO
    fp16 = True 
    
    train = False # set True for training  
    infer = True
    debug = False # set True for debugging
    wandb = True # False
    wb_project = "NLP-Rater"
    output = "/workspace/results/"

    seed = 42
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
    # ######
    NUM_CORES = 8 #cfg["NUM_CORES"]
    min_tokens = 6
    MIN_TOKENS = {
        "Lead": 6,
        "Position": 3,
        "Evidence": 20,
        "Claim": 1,
        "Concluding Statement": 3,
        "Counterclaim": 7,
        "Rebuttal": 6
    }


# %%
os.chdir('/workspace')

from utils.utils import *

DIR = Path("/workspace/data/")
OUTPUT = Path(CFG.output + CFG.exp_name)
OUTPUT.mkdir(exist_ok=True, parents=True)

# %%
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
seed_everything(CFG.seed)

# %%
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

TYPE2LABEL = {t:l for l, t in enumerate(LABELS_TYPE)}
LABEL2TYPE = {k:v for v,k in TYPE2LABEL.items()}
# CLASSES == LABELS_TYPE

# %% [markdown]
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

# %%
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

# %%
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

# %%
def get_data(fold=0):
    
    df = pd.read_csv(DIR / "train_sgkf_fold5.csv")
    df['essay_id'] = df['id'] = df['essay_id_comp']
    df['discourse_effectiveness_int'] = df['discourse_effectiveness'].map(ner_labels_eff)

    ## add whole essays 
    df_texts = pd.read_csv(DIR / 'train_fold10_sgkf.csv')
    df_texts = df_texts[['essay_id', 'discourse_id', 'essay_text']]
    df = df.merge(df_texts, on=['essay_id', 'discourse_id'])

    if "class" not in df.columns:
        df["class"] = df["discourse_type"]

    print(df.shape)
    # display(df.head())

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    return train_df, valid_df
        

# %% [markdown]


# %%
import time
import pandas as pd
import numpy as np
import gc
from os.path import join as opj
from tqdm import tqdm
import re
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align, nms
from transformers import AutoTokenizer
from transformers import (AutoModelForTokenClassification, AutoModel, AutoTokenizer,
                          AutoConfig, AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
# from public_metric_2021 import score_feedback_comp
from collections import OrderedDict

from losses.jaccard import *
from utils.RATER_MetricCode import compute_scores

# LABEL2TYPE = ('Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal',  'Evidence', 'Concluding Statement')
# TYPE2LABEL = {t:l for l, t in enumerate(LABEL2TYPE)}
# LABEL2TYPE = {l:t for t,l in TYPE2LABEL.items()}


def to_gpu(data):
    if isinstance(data, dict):
        return {k: to_gpu(v) for k, v in data.items()}
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

def save_pkl(data, path):
    '''
    A function to save an object using Pickle
    
    :param data: The data object to save to a file path
    :param path: The path at which to save the target data
    
    :returns: A saved data file at the specified path
    '''
    with open(path, 'wb') as f:  # Open the data file at the specified file
        pickle.dump(data, f)   # Dump all the data into the file



# %%
class DatasetTrain(Dataset):
    def __init__(self,
                 df,
                 tokenizer,
                 mask_prob=0.0,
                 mask_ratio=0.0,
                ):
        
        self.df = df
        self.samples = list(self.df.groupby('id'))
        self.tokenizer = tokenizer
        print(f'Loaded {len(self)} samples.')

        assert 0 <= mask_prob <= 1
        assert 0 <= mask_ratio <= 1
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text_id, df = self.samples[index]
        
        # text_path = opj(self.text_dir, f'{text_id}.txt')
        # with open(text_path) as f:
        #     text = f.read().rstrip()
        text = df['essay_text'].values[0].rstrip()

        tokens = self.tokenizer(text, return_offsets_mapping=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
        offset_mapping = self.strip_offset_mapping(text, offset_mapping)
        num_tokens = len(input_ids)

        # token slices of words (offset_mapping = character slices of tokens)
        woff = self.get_word_offsets(text) # (num_words,2) : character slices of words
        toff = offset_mapping # (num_tokens,2) : character slices of tokens
        wx1, wx2 = woff.T # (num_words,) x 2
        tx1, tx2 = toff.T # (num_tokens,) x 2
        ix1 = np.maximum(wx1[..., None], tx1[None, ...]) # (num_words,num_tokens)
        ix2 = np.minimum(wx2[..., None], tx2[None, ...]) # (num_words,num_tokens)
        ux1 = np.minimum(wx1[..., None], tx1[None, ...]) # (num_words,num_tokens)
        ux2 = np.maximum(wx2[..., None], tx2[None, ...]) # (num_words,num_tokens)
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1 + 1e-12) # (num_words,num_tokens)
        assert (ious > 0).any(-1).all()

        word_boxes = []
        for row in ious: # ious.shape=(num_words,num_tokens)
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
            eff_label = row['discourse_effectiveness'] #ner_labels_eff
            gt_spans.append([start, end + 1, span_label, eff_label])

        gt_spans = torch.LongTensor(gt_spans)

        # random mask augmentation
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids) - 1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = all_inds[:n_mask]
            input_ids[mask_inds] = self.tokenizer.mask_token_id

        return dict(
            text=text,
            text_id=text_id,
            input_ids=input_ids, # token ids
            attention_mask=attention_mask,
            word_boxes=word_boxes, # token slices of words
            gt_spans=gt_spans # word slices of ground truth spans
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



class CustomCollator(object):
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        
    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]
        max_seq_length = len(sample['input_ids'])
        # if self.attention_window is not None:
        #     attention_window = self.attention_window
        #     padded_length = (attention_window -
        #                      max_seq_length % attention_window
        #                      ) % attention_window + max_seq_length
        # else:
        #     padded_length = max_seq_length
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
        gt_spans = sample['gt_spans']

        return dict(text_id=text_id,
                    text=text,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_boxes=word_boxes,
                    gt_spans=gt_spans)
    

# %%
def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp_micro(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = (
        gt_df[["id", "discourse_type", "predictionstring"]]
        .reset_index(drop=True)
        .copy()
    )
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["overlaps"] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = (
        joined.query("potential_TP")
        .sort_values("max_overlap", ascending=False)
        .groupby(["id", "predictionstring_gt"])
        .first()["pred_id"]
        .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


def score_feedback_comp(pred_df, gt_df, return_class_scores=False):
    class_scores = {}
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    for discourse_type, gt_subset in gt_df.groupby("discourse_type"):
        pred_subset = (
            pred_df.loc[pred_df["class"] == discourse_type]
            .reset_index(drop=True)
            .copy()
        )
        class_score = score_feedback_comp_micro(pred_subset, gt_subset)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


            

# %%
class TextSpanDetectorOriginal(nn.Module):
    """
     SpanDetector Original from: 
     https://github.com/Shujun-He/TeamSKT-Feedback-Prize---Predicting-Effective-Arguments-2nd-Place-solution/blob/main/Tom_solution/20_WoSpanDet/code/models_detector.py
    """
    def __init__(self, arch, num_classes=7, local_files_only=True):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            arch,
            num_labels=1 + 2 + num_classes,
            local_files_only=local_files_only
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(arch, local_files_only=local_files_only)


# %%
class TextSpanDetector(nn.Module):
    def __init__(self,
                 model_name,
                 tokenizer,
                 num_classes=7,
                 num_classes2=2,
                 dynamic_positive=True, #False,
                 with_cp=False,
                 hidden_dropout_prob=0, 
                 learning_rate=1e-5,
                 head_learning_rate=1e-3,
                 num_train_steps=0,
                 p_drop=0,
                 warmup_ratio = 0,
                 model_pretraining=None,
                 rnn='none',
                 loss='mse',
                 head='simple',
                 msd='false',
                 multi_layers=1,
                 aug='none',
                 mixup_alpha=1.0,
                 aug_stop_epoch=999,
                 p_aug=0,
                 adv_sift='false',
                 l2norm='false',
                 s=30,
                 weight_decay=0.01,
                 mt='false',
                 w_mt=1,
                 scheduler='cosine',
                 num_cycles=1,
                 loss_weights = [1.0, 2.0], # [1.0, 1.0],
                 freeze_layers='true', ##'false',
                 vb=False,
                ):
        super().__init__()

        self.tokenizer = tokenizer
        self.model_name = model_name
        self.num_labels = 1 + 2 + num_classes
        self.num_labels2 = num_classes2
        self._current_epoch = 1
        self.learning_rate = learning_rate
        self.head_learning_rate = head_learning_rate
        self.hidden_dropout_prob = hidden_dropout_prob
        self.warmup_ratio = warmup_ratio 
        self.num_train_steps = num_train_steps
        self.loss = loss
        self.msd = msd
        self.multi_layers = multi_layers
        self.aug = aug
        self.mixup_alpha = mixup_alpha
        self.aug_stop_epoch = aug_stop_epoch
        self.p_aug = p_aug
        self.adv_sift = adv_sift
        self.l2norm = l2norm
        self.s = s
        self.weight_decay = weight_decay
        self.mt = mt
        self.w_mt = w_mt
        self.scheduler = scheduler
        self.num_cycles = num_cycles
        self.num_classes = num_classes
        self.dynamic_positive = dynamic_positive
        self.loss_weights = [1.0, 1.0]
        self.vb=vb

        if model_pretraining is not None:
            self.transformer = model_pretraining.model.deberta #model_pretraining.transformer
            self.config = model_pretraining.model.config
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.config.update(
                {
                    "output_hidden_states": True,
                    "hidden_dropout_prob": self.hidden_dropout_prob,
                    "add_pooling_layer": False,
                    "num_labels": self.num_labels,
                }
            )
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
            
        # resize
        self.transformer.resize_token_embeddings(len(tokenizer))
            
        #### freeze some layers for large models
        if freeze_layers == 'true':
            if 'deberta-v2-xxlarge' in model_name:
                print('freeze 24/48')
                self.transformer.embeddings.requires_grad_(False)
                self.transformer.encoder.layer[:24].requires_grad_(False) # freeze 24/48
            elif 'deberta-v2-xlarge' in model_name:
                print('freeze 12/24')
                self.transformer.embeddings.requires_grad_(False)
                self.transformer.encoder.layer[:12].requires_grad_(False) # freeze 12/24
            elif 'deberta-xlarge' in model_name:
                print('freeze 12/24')
                self.transformer.embeddings.requires_grad_(False)
                self.transformer.encoder.layer[:12].requires_grad_(False) # freeze 12/24
            elif 'funnel-transformer-xlarge' in model_name:
                self.transformer.embeddings.requires_grad_(False)
                self.transformer.encoder.blocks[:1].requires_grad_(False) # freeze 1/3

            ############
            elif 'deberta-v3-large' in model_name:
                self.transformer.embeddings.requires_grad_(False)

                print('freeze 12/24')
                self.transformer.encoder.layer[:12].requires_grad_(False) # freeze 12/24
                
                # print('freeze 24/48')
                # self.transformer.encoder.layer[:24].requires_grad_(False) # freeze 24/48

                # print('freeze all layers..')
                # self.transformer.encoder.layer[:].requires_grad_(False) 

            elif 'deberta-large' in model_name:
                self.transformer.embeddings.requires_grad_(False)
                print('freeze 12/24')
                self.transformer.encoder.layer[:12].requires_grad_(False) # freeze 12/24
                # print('freeze all layers..')
                # self.transformer.encoder.layer[:].requires_grad_(False) 
            
            elif 'deberta-base' in model_name:
                self.transformer.embeddings.requires_grad_(False)
                print('freeze half layers (6/12)..')
                self.transformer.encoder.layer[:6].requires_grad_(False) # freeze 12/24
                # print('freeze all layers (/12)..')
                # self.transformer.encoder.layer[:].requires_grad_(False) 
   

        #########################
        if with_cp:
            self.transformer.gradient_checkpointing_enable()
            
        self.head = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(self.config.hidden_size, self.num_labels)
        )
        self._init_weights(self.head)

        self.head2 = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(self.config.hidden_size, self.num_labels2)
        )
        self._init_weights(self.head2)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    
    #@torch.no_grad()
    def forward_logits(self, data):
        batch_size = data['input_ids'].size(0)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'
        if self.vb: 
            print('[TRAIN] data word_boxes', data['word_boxes'].shape )
            print('[TRAIN] data input_ids', data['input_ids'].shape )
            print('[TRAIN] data gt_spans', data['gt_spans'].shape )

        outputs = self.transformer(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
        x = outputs.last_hidden_state
        if self.vb: print('[TRAIN] hidden_state', x.shape, x )
        
        logits_tok = self.head(x)
        if self.vb: print('[TRAIN] logits type', logits_tok.shape )

        logits = aggregate_tokens_to_words(logits_tok, data['word_boxes'])
        assert logits.size(0) == data['text'].split().__len__()

        obj_pred = logits[..., 0]
        reg_pred = logits[..., 1:3]
        cls_pred = logits[..., 3:]

        return obj_pred, reg_pred, cls_pred, x


    def predict(self, data, test_score_thr):
        data = to_gpu(data)
        obj_pred, reg_pred, cls_pred, h = self.forward_logits(data)
        obj_pred = obj_pred.sigmoid()
        reg_pred = reg_pred.exp()
        cls_pred = cls_pred.sigmoid()

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        pr_scores = (obj_scores * cls_scores)**0.5
        pos_inds = pr_scores > test_score_thr

        if pos_inds.sum() == 0:
            return dict(text_id=data['text_id'])

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
    
    
    def training_step(self, data, trn_df, test_score_thr):
        data = to_gpu(data)

        obj_pred, reg_pred, cls_pred, h = self.forward_logits(data)
        if self.vb: 
            print('[TRAIN] obj_pred', len(obj_pred), reg_pred.shape, cls_pred.shape)


        obj_target, reg_target, cls_target, cls_target_eff, pos_loc = self.build_target(data['gt_spans'], obj_pred, reg_pred, cls_pred)
        obj_loss, reg_loss, cls_loss = self.get_losses(obj_pred, reg_pred, cls_pred,
                                                       obj_target, reg_target, cls_target,
                                                       pos_loc)
        loss1 = obj_loss + reg_loss + cls_loss
        
        res = self.predict(data, test_score_thr)
        #########################################
        h = h.squeeze(0) # seq_len x hidden_size
        if self.vb: print('[TRAIN] h:', h.shape, h)

        # start_, end_ = res['start'], res['end']
        start_, end_ = data['gt_spans'][:,0].tolist(), data['gt_spans'][:,1].tolist()

        pooled_outs = []
        for st_, en_ in zip(start_, end_):
            # active_logits_eff_word = res1['logits_eff_word'][st_:en_].mean()
            # active_logits_eff_word = self.head_eff(h[st_:en_+1])                   # len_span x 2 
            # active_logits_eff_word = active_logits_eff_word.mean(0).unsqueeze(0) # 1x2
            if self.vb:  print('[TRAIN] start/end', st_, en_)

            tmp = h[st_:en_+1,:] 
            if self.vb:  print('[TRAIN] tmp h', tmp.shape, tmp)
            tmp = tmp.mean(0)
            if self.vb:  print('[TRAIN] tmp h.mean(0)', tmp.shape, tmp)

            pooled_outs.append(tmp.unsqueeze(0))
        if self.vb: 
            print('[TRAIN] len pooled_outs', len(pooled_outs))

        pooled_outs = torch.cat(pooled_outs)
        if self.vb: print('[TRAIN] pooled_outs cat', pooled_outs.shape, pooled_outs)
        
        logits_eff = self.head2(pooled_outs)  
        if self.vb: print('[TRAIN] logits_eff', logits_eff.shape, logits_eff)

        discourse_eff_target = torch.tensor(data['gt_spans'][:,-1])
        if self.vb: print('[TRAIN] eff_target', discourse_eff_target.shape, discourse_eff_target)
        
        loss_fct = nn.CrossEntropyLoss(reduction='mean') 
        loss2 = loss_fct(logits_eff.softmax(-1), discourse_eff_target) 

        # 'sum'
        # loss_fct = nn.CrossEntropyLoss(reduction='sum')
        # cls_eff_loss = loss_fct(logits_eff.softmax(-1), discourse_eff_target) / num_total_samples
        ###############
        # Total loss
        loss = self.loss_weights[0]*loss1 + self.loss_weights[0]*loss2       
        
        logits_eff_softmax = logits_eff.softmax(-1)
        logits_eff_softmax = to_np(logits_eff_softmax)
        if self.vb: print('[TRAIN] logits_eff_softmax', logits_eff_softmax.shape)

        #########################################
        if len(res.keys())>1:
            pred = []
            text_id = res['text_id']
            for c,start,end, p_eff in zip(res['label'],res['start'],res['end'],logits_eff_softmax):
                pred.append([
                    text_id, 
                    LABEL2TYPE[c], 
                    ' '.join(np.arange(start,end+1).astype(str)),
                    c,
                    start, 
                    end,
                    p_eff[0], #logits_eff_softmax[ii,0], 
                    p_eff[1], #logits_eff_softmax[ii,1]  
                    ])
            pred_df = pd.DataFrame(pred, columns=['id','class','predictionstring', 'discourse_type', 'start', 'end', 'score_discourse_effectiveness_0', 'score_discourse_effectiveness_1'])
            gt_df = trn_df[trn_df['id']==res['text_id']].reset_index(drop=True)
            gt_df['discourse_type'] = gt_df['discourse_type'].map(TYPE2LABEL)

            # score_old = score_feedback_comp(pred_df, gt_df, return_class_scores=False)
            scores_dict = compute_scores(pred_df, gt_df, ids_selected=list(gt_df.id.unique()), threshold=0.5, return_segments=False, return_scores=False)
            score = scores_dict['F1_double_w'] 
            score_f1_sw = scores_dict['F1_single_w']
            score_f1_fp1 = scores_dict['F1_FB1']

        else:
            score = 0
            score_f1_sw = 0
            score_f1_fp1 = 0
            
        return {
            'loss':loss,
            'obj_loss':obj_loss.item(),
            'reg_loss':reg_loss.item(),
            'cls_loss':cls_loss.item(),
            'score':score,                #scores_dict['F1_double_w']
            'score_f1_sw':score_f1_sw,    #scores_dict['F1_single_w'], 
            'score_f1_fp1':score_f1_fp1,  #scores_dict['F1_FB1'], 
            ###
            'loss1':loss1,
            'loss2':loss2,
        }
    
    def validation_step(self, data, val_df, test_score_thr):
        data = to_gpu(data)
        obj_pred, reg_pred, cls_pred, h = self.forward_logits(data)
        obj_target, reg_target, cls_target, cls_target_eff, pos_loc = self.build_target(data['gt_spans'], obj_pred, reg_pred, cls_pred)
        obj_loss, reg_loss, cls_loss = self.get_losses(obj_pred, reg_pred, cls_pred,
                                                       obj_target, reg_target, cls_target,
                                                       pos_loc)
        
        loss1 = obj_loss + reg_loss + cls_loss
        
        res = self.predict(data, test_score_thr)
        #########################################
        h = h.squeeze(0) # seq_len x hidden_size

        # start_, end_ = res['start'], res['end']
        start_, end_ = data['gt_spans'][:,0].tolist(), data['gt_spans'][:,1].tolist()

        pooled_outs = []
        for st_, en_ in zip(start_, end_):
            pooled_outs.append(h[st_:en_+1].mean(0).unsqueeze(0))
        if self.vb: 
            print('[VAL] len pooled_outs', len(pooled_outs))

        pooled_outs = torch.cat(pooled_outs)
        logits_eff = self.head2(pooled_outs)  
        if self.vb: print('[VAL] logits_eff', logits_eff.shape)

        discourse_eff_target = torch.tensor(data['gt_spans'][:,-1])
        if self.vb: print('[VAL] eff_target', discourse_eff_target.shape)
        
        loss_fct = nn.CrossEntropyLoss(reduction='mean') 
        loss2 = loss_fct(logits_eff.softmax(-1), discourse_eff_target) 

        # 'sum'
        # loss_fct = nn.CrossEntropyLoss(reduction='sum')
        # cls_eff_loss = loss_fct(logits_eff.softmax(-1), discourse_eff_target) / num_total_samples
        ###############
        # Total loss
        loss = self.loss_weights[0]*loss1 + self.loss_weights[0]*loss2  
        ########

        logits_eff_softmax = logits_eff.softmax(-1) # num_disc x 2 
        logits_eff_softmax = to_np(logits_eff_softmax)

        ###############################################
        gt_df = val_df[val_df['id']==res['text_id']].reset_index(drop=True)
        if len(res.keys())>1:
            pred = []
            text_id = res['text_id']
            for ii, (c,start,end,p_eff) in enumerate(zip(res['label'],res['start'],res['end'],logits_eff_softmax)):
                pred.append([
                    text_id, 
                    LABEL2TYPE[c], 
                    ' '.join(np.arange(start,end+1).astype(str)),
                    c,
                    start, 
                    end,
                    p_eff[0], #logits_eff_softmax[ii,0], 
                    p_eff[1], #logits_eff_softmax[ii,1]  
                    ])
            pred_df = pd.DataFrame(pred, columns=['id','class','predictionstring', 'discourse_type', 'start', 'end', 'score_discourse_effectiveness_0', 'score_discourse_effectiveness_1'])
        else:
            pred_df = pd.DataFrame([], columns=['id','class','predictionstring', 'discourse_type', 'start', 'end', 'score_discourse_effectiveness_0', 'score_discourse_effectiveness_1'])
            
        return {
            'data_id':res['text_id'],
            'loss':loss.item(),
            'obj_loss':obj_loss.item(),
            'reg_loss':reg_loss.item(),
            'cls_loss':cls_loss.item(),
            'pred_df':pred_df,
            'gt_df':gt_df,
            ######
            'loss1':loss1.item(),
            'loss2':loss2.item(),
        }
    
    
    def validation_epoch_end(self, outputs, add_epoch=True):
        losses = []
        obj_losses = []
        reg_losses = []
        cls_losses = []
        eff_losses = []
        seg_losses = []
        pred_dfs = []
        gt_dfs = []
        for o in outputs:
            losses.append(o['loss'])
            obj_losses.append(o['obj_loss'])
            reg_losses.append(o['reg_loss'])
            cls_losses.append(o['cls_loss'])
            eff_losses.append(o['loss2'])
            seg_losses.append(o['loss1'])
            pred_dfs.append(o['pred_df'])
            gt_dfs.append(o['gt_df'])
        loss = np.array(losses).mean()
        obj_loss = np.array(obj_losses).mean()
        reg_loss = np.array(reg_losses).mean()
        cls_loss = np.array(cls_losses).mean()
        eff_loss = np.array(eff_losses).mean()
        seg_loss = np.array(seg_losses).mean()
        
        pred_df = pd.concat(pred_dfs, axis=0).reset_index(drop=True)
        gt_df   = pd.concat(gt_dfs, axis=0).reset_index(drop=True)
        gt_df['discourse_type'] = gt_df['discourse_type'].map(TYPE2LABEL)
        
        #score_old = score_feedback_comp(pred_df, gt_df, return_class_scores=False)
        scores_dict = compute_scores(pred_df, gt_df, ids_selected=list(gt_df.id.unique()), threshold=0.5, return_segments=False, return_scores=False)
        score = scores_dict['F1_double_w']
        # 'F1_FB1'   'F1_single_w'   'F1_double_w'  
    
        if add_epoch:
            self._current_epoch += 1
            
        return {
            'loss':loss,
            'obj_loss':obj_loss,
            'reg_loss':reg_loss,
            'cls_loss':cls_loss,
            'eff_loss':eff_loss,
            'loss1':seg_loss,
            'score':score,
            'score_f1_fp1':scores_dict['F1_FB1'],
            'score_f1_sw':scores_dict['F1_single_w'],
            ####
            'pred_df': pred_df,
            'gt_df': gt_df,
        }
    
    ######### INFERENCE step 
    def test_step(self, batch, test_score_thr=0.5):
        data = to_gpu(batch)
        
        obj_pred, reg_pred, cls_pred, h = self.forward_logits(data)

        res = self.predict(data, test_score_thr)

        if len(res.keys())==0:
            res = self.predict(data, 0.4)
            print('text_id', data['text_id'], 0.4)
            
            if len(res.keys())==0:
                res = self.predict(data, 0.35)
                print('text_id', data['text_id'], 0.35)

                if len(res.keys())==0:
                    res = self.predict(data, 0.30)
                    print('text_id', data['text_id'], 0.30)
                    

        #########################################
        h = h.squeeze(0) # seq_len x hidden_size

        try:
            start_, end_ = res['start'].tolist(), res['end'].tolist()
            # start_, end_ = data['gt_spans'][:,0].tolist(), data['gt_spans'][:,1].tolist()
        except:
            start_, end_ = [1], [10] 
            print('--> Dummy span pred for text_id:', data['text_id'])

        pooled_outs = []
        for st_, en_ in zip(start_, end_):
            pooled_outs.append(h[st_:en_+1].mean(0).unsqueeze(0))

        pooled_outs = torch.cat(pooled_outs)
        logits_eff = self.head2(pooled_outs)  
        logits_eff_softmax = logits_eff.softmax(-1) # num_disc x 2 
        logits_eff_softmax = to_np(logits_eff_softmax)

        ###############################################
        if len(res.keys())>1:
            pred = []
            text_id = res['text_id']
            for ii, (c,start,end,p_type,p_eff) in enumerate(zip(res['label'],res['start'],res['end'],res['score'],logits_eff_softmax)):
                pred.append([
                    text_id, 
                    LABEL2TYPE[c], 
                    ' '.join(np.arange(start,end+1).astype(str)),
                    c,
                    start, 
                    end,
                    p_type,
                    p_eff[0],  
                    p_eff[1],  
                    ])
            pred_df = pd.DataFrame(pred, columns=['id','class','predictionstring', 'discourse_type', 'start', 'end', 'score', 'score_discourse_effectiveness_0', 'score_discourse_effectiveness_1'])
        else:
            pred_df = pd.DataFrame([], columns=['id','class','predictionstring', 'discourse_type', 'start', 'end', 'score', 'score_discourse_effectiveness_0', 'score_discourse_effectiveness_1'])
            
        
        return {
            'text_id':batch['text_id'],
            'obj_pred':obj_pred,
            'reg_pred':reg_pred,
            'cls_pred':cls_pred,
            'eff_pred':logits_eff_softmax,
            'pred_df': pred_df,
            'hidden_state': to_np(h)
        }




    ################################################
    def configure_optimizers(self):
        optimizer = self.fetch_optimizer()
        scheduler = self.fetch_scheduler(optimizer)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def fetch_optimizer(self):
        head_params = list(self.head.named_parameters())
        param_optimizer = list(self.transformer.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n,p in head_params], 
                "weight_decay": 0.01,
                "lr": self.head_learning_rate,
            },
        ]
        optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)
        return optimizer

    def fetch_scheduler(self, optimizer):
        print('self.warmup_ratio = ', self.warmup_ratio)
        print('self.num_train_steps = ', self.num_train_steps)
        print('num_warmup_steps = ', int(self.warmup_ratio * self.num_train_steps))
        if self.scheduler=='cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.warmup_ratio * self.num_train_steps),
                num_training_steps=self.num_train_steps,
                num_cycles=0.5*self.num_cycles,
                last_epoch=-1,
            )
        elif self.scheduler=='cosine_hard':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.warmup_ratio * self.num_train_steps),
                num_training_steps=self.num_train_steps,
                num_cycles=self.num_cycles,
                last_epoch=-1,
            )
        elif self.scheduler=='linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.warmup_ratio * self.num_train_steps),
                num_training_steps=self.num_train_steps,
                last_epoch=-1,
            )
        return scheduler
    
    def get_losses(self, 
                   obj_pred, reg_pred, cls_pred, 
                   obj_target, reg_target, cls_target, 
                   pos_loc):
        
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
            obj_pred, 
            obj_target,
            reduction='sum') / num_total_samples
        
        return obj_loss, reg_loss, cls_loss

    @torch.no_grad()
    def build_target(self, gt_spans, obj_pred, reg_pred, cls_pred): #cls_pred2
        obj_target = torch.zeros_like(obj_pred)
        reg_target = torch.zeros_like(reg_pred)
        cls_target = torch.zeros_like(cls_pred)
        # cls_eff_target = torch.zeros_like(cls_pred2)
        cls_eff_target = torch.zeros((len(cls_pred), 2))
        
        # first token as positive
        pos_loc = gt_spans[:, 0]
        
        obj_target[pos_loc] = 1
        reg_target[pos_loc, 0] = gt_spans[:, 0].float()
        reg_target[pos_loc, 1] = gt_spans[:, 1].float()
        cls_target[pos_loc, gt_spans[:, 2]] = 1
        cls_eff_target[pos_loc, gt_spans[:, 3]] = 1

        # dynamically assign one more positive
        if self.dynamic_positive:
            cls_prob = (obj_pred.sigmoid().unsqueeze(1) *
                        cls_pred.sigmoid()).sqrt()
            
            for start, end, label, label_eff in gt_spans:
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
                cls_eff_target[pos_ind, label_eff] = 1

            pos_loc = (obj_target == 1).nonzero().flatten()
        return obj_target, reg_target, cls_target, cls_eff_target, pos_loc
    

    def predict_eff(self, start_list, end_list, h_state=None):

        # h = res1['hidden_state'].squeeze(0)  # (seq_len, hidden_size)
        h = h_state.squeeze(0)

        pooled_outs = []
        for st_, en_ in zip(start_list, end_list):
            pooled_outs.append(h[st_:en_+1].mean(0).unsqueeze(0))        

        pooled_outs = torch.cat(pooled_outs)
        logits_eff = self.head2(pooled_outs)  

        logits_eff_pred = logits_eff.softmax(-1)
        _, eff_cls = logits_eff_pred.max(-1) 

        eff_proba_0 = logits_eff_pred[:, 0] ## negative class
        eff_proba_1 = logits_eff_pred[:, 1] ## positive class

        return dict(
            # text_id=batch['text_id'],
            eff_proba_0=to_np(eff_proba_0),
            eff_proba_1=to_np(eff_proba_1),
            eff_cls=to_np(eff_cls),
            )




# %%
def train_fold(args, trn_df, val_df, pseudo_df=None, fold=0):

    output_path = opj(args.output, args.exp_name, f"fold{str(args.fold)}")
    os.makedirs(output_path, exist_ok=True)
    
    if True:
        if 'deberta-v2' in args.model or 'deberta-v3' in args.model:
            tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, trim_offsets=False)
            # special_tokens_dict = {'additional_special_tokens': ['\n\n'] + [f'[{s.upper()}]' for s in list(TYPE2LABEL.keys())]}
            special_tokens_dict = {'additional_special_tokens': ['\n\n'] }
            _ = tokenizer.add_special_tokens(special_tokens_dict)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trim_offsets=False)
            # special_tokens_dict = {'additional_special_tokens': [f'[{s.upper()}]' for s in list(TYPE2LABEL.keys())]}
            # _ = tokenizer.add_special_tokens(special_tokens_dict)

        trn_dataset = DatasetTrain(
            trn_df,
            tokenizer=tokenizer,
            mask_prob=args.mask_prob,
            mask_ratio=args.mask_ratio,
        )
        trn_dataloader = DataLoader(
            trn_dataset,
            batch_size=args.trn_batch_size,
            shuffle=True,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=True,
        )
        
        val_dataset = DatasetTrain(
            val_df,
            tokenizer=tokenizer,
            mask_prob=0,
            mask_ratio=0,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=False,
        )
        
        #model
        num_trn_dataset = len(trn_dataset)
        num_train_steps = int(num_trn_dataset / args.trn_batch_size / args.accumulate_grad_batches * args.epochs)
        print('num_train_steps = ', num_train_steps)
        
        model_pretraining = None
        if 'tasckj' in args.pretrained_detector_path:
            # from models_detector import TextSpanDetectorOriginal
            model_pretraining = TextSpanDetectorOriginal(args.model_name, num_classes=7)
            model_pretraining.load_state_dict(torch.load(args.pretrained_detector_path)["state_dict"])
            print('Loaded encoder weights (taskj):', args.pretrained_detector_path)

        elif args.pretrained_detector_path != 'none':
            model_pretraining = TextSpanDetector_stage1(
                args.model,
                tokenizer,
                num_classes=args.num_classes,
                dynamic_positive=True,
                with_cp=False,
            )
            model_pretraining.load_state_dict(torch.load(args.pretrained_detector_path))
            print('Loaded encoder weights:', args.pretrained_detector_path)



        
        model = TextSpanDetector(
            args.model,
            tokenizer,
            num_classes=args.num_classes,
            dynamic_positive=True,
            with_cp=(args.check_pointing=='true'),
            hidden_dropout_prob=args.hidden_drop_prob, 
            p_drop=args.p_drop,
            scheduler=args.scheduler,
            learning_rate=args.lr,
            head_learning_rate=args.head_lr,
            num_train_steps=num_train_steps,
            warmup_ratio=args.warmup_ratio,
            model_pretraining=model_pretraining,
        )
        
        model = model.cuda()
        if args.pretrain_path != 'none':
            model.load_state_dict(torch.load(args.pretrain_path))
            print('Loaded weights:', args.pretrain_path)
            
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16=='true'))
        
        [optimizer], [scheduler] = model.configure_optimizers()
        scheduler = scheduler['scheduler']
        
        #training
        val_score_best  = -1e+99
        val_score_best2 = -1e+99
        val_loss_best   = 1e+99
        epoch_best = 0
        counter_ES = 0
        trn_score = 0
        start_time = time.time()
        for epoch in range(1, args.epochs+1):
            if epoch > args.stop_epoch:
                break
            if epoch < args.restart_epoch:
                scheduler.step()
                continue
                
            print('\nlr : ', [ group['lr'] for group in optimizer.param_groups ])
            
            #train
            trn_loss = 0
            trn_obj_loss = 0
            trn_reg_loss = 0
            trn_cls_loss = 0
            trn_eff_loss = 0
            trn_score = 0
            trn_score_f1_fp1 = 0
            trn_score_f1_sw = 0
            counter = 0
            tk0 = tqdm(trn_dataloader, total=int(len(trn_dataloader)))
            optimizer.zero_grad()
            trn_preds = []
            trn_trues = []
            
            if args.awp=='true':
                awp = AWP(
                    model,
                    criterion=nn.CrossEntropyLoss(reduction='none'), 
                    optimizer=optimizer,
                    apex=(args.fp16=='true'),
                    adv_lr=args.awp_lr, 
                    adv_eps=args.awp_eps
                )
            
            for i,data in enumerate(tk0):
                batch = trn_dataloader.batch_size
                model.train()
                with torch.cuda.amp.autocast(enabled=(args.fp16=='true')):
                    trn_res = model.training_step(data, trn_df, test_score_thr=args.test_score_thr)
                    loss = trn_res['loss']
                    
                    if args.accumulate_grad_batches > 1:
                        loss = loss / args.accumulate_grad_batches
                    scaler.scale(loss).backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                    
                    trn_loss += loss.item() * batch * args.accumulate_grad_batches
                    trn_obj_loss += trn_res['obj_loss']  * batch
                    trn_reg_loss += trn_res['reg_loss']  * batch
                    trn_cls_loss += trn_res['cls_loss']  * batch
                    trn_score += trn_res['score'] * batch

                    trn_score_f1_fp1 += trn_res['score_f1_fp1'] * batch # scores_dict['F1_single_w'], 
                    trn_score_f1_sw  += trn_res['score_f1_sw'] * batch   # scores_dict['F1_FB1'], 
                             
                    if args.awp=='true':
                        if epoch >= args.awp_start_epoch:
                            loss = awp.attack_backward(data, trn_df, test_score_thr=args.test_score_thr)
                            if args.accumulate_grad_batches > 1:
                                loss = loss / args.accumulate_grad_batches
                            scaler.scale(loss).backward()
                            awp._restore()
                            trn_loss += loss.item() * batch * args.accumulate_grad_batches

                    if (i + 1) % args.accumulate_grad_batches == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                
                counter += 1
                tk0.set_postfix(loss=(trn_loss / (counter * batch) ), score=(trn_score / (counter * batch)), score_f1=(trn_score_f1_fp1 / (counter * batch)))
                
                # eval
                if args.eval_step!=-1 and (i+1)%args.eval_step==0:
                    model.eval()
                    outputs = []
                    for i,data in enumerate(val_dataloader):
                        with torch.no_grad():
                            outputs.append(model.validation_step(data, val_df, test_score_thr=args.test_score_thr))
                        #release GPU memory cache
                        del data
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    val_res = model.validation_epoch_end(outputs, add_epoch=False)
                    val_loss = val_res['loss']
                    val_score = val_res['score']
                    val_score_f1_fp1 = val_res['score_f1_fp1']
                    val_score_f1_sw = val_res['score_f1_sw']

                    #monitoring
                    print('\nepoch {:.0f}: trn_loss = {:.4f}, val_loss = {:.4f}, trn_score = {:.4f}, val_score = {:.4f}, val_score_f1_fp1 = {:.4f}, val_score_f1_sw = {:.4f}'.format(
                        epoch, 
                        trn_loss / (counter * trn_dataloader.batch_size), 
                        val_loss, 
                        trn_score / (counter * trn_dataloader.batch_size),
                        val_score, 
                        val_score_f1_fp1, 
                        val_score_f1_sw
                    ))
                    
                    print('trn_obj_loss={:.4f}, trn_reg_loss={:.4f}, trn_cls_loss={:.4f}, trn_eff_loss={:.4f}'.format(
                        trn_obj_loss / (counter * trn_dataloader.batch_size),
                        trn_reg_loss / (counter * trn_dataloader.batch_size), 
                        trn_cls_loss / (counter * trn_dataloader.batch_size),
                        trn_eff_loss / (counter * trn_dataloader.batch_size)
                    ))
                    
                    print('val_obj_loss={:.4f}, val_reg_loss={:.4f}, val_cls_loss={:.4f}, val_eff_loss={:.4f}'.format(
                        val_res['obj_loss'], val_res['reg_loss'], val_res['loss2']
                    ))
                          
                    # if args.slack_url!='none':
                    #     from utils import post_message
                    #     post_message(name='bot',
                    #                  message='epoch {:.0f}: trn_loss = {:.4f}, val_loss={:.4f}, trn_score = {:.4f}, val_score = {:.4f}'.format(
                    #                      epoch, trn_loss, val_loss, trn_score, val_score), 
                    #                  incoming_webhook_url=args.slack_url)
                    if args.early_stopping=='true':
                        if val_loss < val_loss_best: #val_score > val_score_best:
                            val_score_best = val_score #update
                            val_loss_best  = val_loss #update
                            epoch_best     = epoch #update
                            counter_ES     = 0 #reset
                            torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save
                            print('model (best loss) saved')
                        else:
                            counter_ES += 1
                        if counter_ES > args.patience:
                            print('early stopping, epoch_best {:.0f}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(
                                epoch_best, val_loss_best, val_score_best))
                            break
                    else:
                        torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save

                    if val_score > val_score_best2:
                        val_score_best2 = val_score #update
                        torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}.pth')) #save
                        print('model (best score) saved')
                
            trn_loss  = trn_loss / len(trn_dataset)
            trn_score = trn_score / len(trn_dataset)
            
            #release GPU memory cache
            del data, loss
            torch.cuda.empty_cache()
            gc.collect()
            
            # save
            torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_epoch{epoch}.pth')) #save

            #eval
            model.eval()
            outputs = []
            tk1 = tqdm(val_dataloader, total=int(len(val_dataloader)))
            for i,data in enumerate(tk1):
                with torch.no_grad():
                    outputs.append(model.validation_step(data, val_df, test_score_thr=args.test_score_thr))
                #release GPU memory cache
                del data
                torch.cuda.empty_cache()
                gc.collect()
            
            val_res = model.validation_epoch_end(outputs)
            # print(val_res.keys())
            # dict_keys(['loss', 'obj_loss', 'reg_loss', 'cls_loss', 'eff_loss', 'loss1', 'score', 'score_f1_fp1', 'score_f1_sw', 'pred_df', 'gt_df'])

            val_loss  = val_res['loss']
            val_score = val_res['score']
            val_score_f1_fp1 = val_res['score_f1_fp1']
            val_score_f1_sw  = val_res['score_f1_sw']
            
            #monitering
            print('epoch {:.0f}: trn_loss = {:.4f}, val_loss = {:.4f}, trn_score = {:.4f}, val_score = {:.4f}, val_score_f1_fp1 = {:.4f}, val_score_f1_sw = {:.4f}'.format(
                epoch, 
                trn_loss,
                val_loss,
                trn_score, 
                val_score,
                val_score_f1_fp1,
                val_score_f1_sw
            ))
            
            print('trn_obj_loss={:.4f}, trn_reg_loss={:.4f}, trn_cls_loss={:.4f}'.format(
                trn_obj_loss / len(trn_dataset),
                trn_reg_loss / len(trn_dataset), 
                trn_cls_loss / len(trn_dataset)
            ))
            
            print('val_obj_loss={:.4f}, val_reg_loss={:.4f}, val_cls_loss={:.4f}, val_eff_loss={:.4f}'.format(
                val_res['obj_loss'], val_res['reg_loss'], val_res['cls_loss'], val_res['eff_loss']
            ))
                  
            # if args.slack_url!='none':
            #     from utils import post_message
            #     post_message(name='bot',
            #                  message='epoch {:.0f}: trn_loss = {:.4f}, val_loss={:.4f}, trn_score = {:.4f}, val_score = {:.4f}'.format(
            #                      epoch, trn_loss, val_loss, trn_score, val_score), 
            #                  incoming_webhook_url=args.slack_url)
            if epoch%10 == 0: print(' elapsed_time = {:.1f} min'.format((time.time() - start_time)/60))
                
            if args.early_stopping=='true':
                if val_loss < val_loss_best: #val_score > val_score_best:
                    val_score_best = val_score #update
                    val_loss_best  = val_loss #update
                    epoch_best     = epoch #update
                    counter_ES     = 0 #reset
                    torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save
                    print('model (best loss) saved')
                else:
                    counter_ES += 1
                if counter_ES > args.patience:
                    print('early stopping, epoch_best {:.0f}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(
                        epoch_best, val_loss_best, val_score_best))
                    break
            else:
                torch.save(model.state_dict(), opj(output_path, f'model_seed{args.seed}_fold{args.fold}_bestloss.pth')) #save
               
            if val_score > val_score_best2:
                val_score_best2 = val_score #update
                torch.save(model.state_dict(), opj(output_path, f'model_seed{args.seed}_fold{args.fold}.pth')) #save
                print('model (best score) saved')
                
        #best model
        if args.early_stopping=='true' and counter_ES<=args.patience:
            print('epoch_best {:d}, val_loss_best {:.5f}, val_score_best {:.5f}'.format(epoch_best, val_loss_best, val_score_best))
        
        del model
        torch.cuda.empty_cache()
        gc.collect()



# %%
def eval_fold(args, trn_df, val_df, fold=0):

    output_path = opj(args.output, args.exp_name, f"fold{str(args.fold)}")
    os.makedirs(output_path, exist_ok=True)
    
    if True:
        if 'deberta-v2' in args.model or 'deberta-v3' in args.model:
            from transformers.models.deberta_v2 import DebertaV2TokenizerFast
            tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, trim_offsets=False)
            special_tokens_dict = {'additional_special_tokens': ['\n\n'] + [f'[{s.upper()}]' for s in list(TYPE2LABEL.keys())]}
            _ = tokenizer.add_special_tokens(special_tokens_dict)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trim_offsets=False)
            special_tokens_dict = {'additional_special_tokens': [f'[{s.upper()}]' for s in list(TYPE2LABEL.keys())]}
            _ = tokenizer.add_special_tokens(special_tokens_dict)

        trn_dataset = DatasetTrain(
            trn_df,
            tokenizer=tokenizer,
            mask_prob=args.mask_prob,
            mask_ratio=args.mask_ratio,
        )
        trn_dataloader = DataLoader(
            trn_dataset,
            batch_size=args.trn_batch_size,
            shuffle=True,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=True,
        )
        
        val_dataset = DatasetTrain(
            val_df,
            tokenizer=tokenizer,
            mask_prob=0,
            mask_ratio=0,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=False,
        )
        
        # #model
        num_trn_dataset = len(trn_dataset)
        num_train_steps = int(num_trn_dataset / args.trn_batch_size / args.accumulate_grad_batches * args.epochs)
        print('num_train_steps = ', num_train_steps)
        # model_pretraining = None
        
        
        model = TextSpanDetector(
            args.model,
            tokenizer,
            num_classes=args.num_classes,
            dynamic_positive=True,
            with_cp=(args.check_pointing=='true'),
            hidden_dropout_prob=args.hidden_drop_prob, 
            p_drop=args.p_drop,
            learning_rate=args.lr,
            head_learning_rate=args.head_lr,
            num_train_steps=num_train_steps,
            warmup_ratio=args.warmup_ratio,
            model_pretraining=None,
            rnn=args.rnn,
            loss=args.loss,
            # head=args.head,
            scheduler=args.scheduler,
            # msd=args.msd,
            # multi_layers=args.multi_layers,
            # aug=args.aug,
            # mixup_alpha=args.mixup_alpha,
            # aug_stop_epoch=args.aug_stop_epoch,
            # p_aug=args.p_aug,
            # adv_sift=args.adv_sift,
            # l2norm=args.l2norm,
            # weight_decay=args.weight_decay,
            # freeze_layers=args.freeze_layers,
            # mt=args.mt,
            # w_mt=args.w_mt,
            # num_cycles=args.num_cycles,
        )
        
        model = model.cuda()
        if args.pretrain_path != 'none':
            model.load_state_dict(torch.load(args.pretrain_path))
            
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16=='true'))
        
        [optimizer], [scheduler] = model.configure_optimizers()
        scheduler = scheduler['scheduler']
        
        # #training
        # val_score_best  = -1e+99
        # val_score_best2 = -1e+99
        # val_loss_best   = 1e+99
        # epoch_best = 0
        # counter_ES = 0
        # trn_score = 0
        # start_time = time.time()
        
        #eval
        model.eval()
        outputs = []
        tk1 = tqdm(val_dataloader, total=int(len(val_dataloader)))
        for i,data in enumerate(tk1):
            with torch.no_grad():
                outputs.append(model.validation_step(data, val_df, test_score_thr=args.test_score_thr))
            #release GPU memory cache
            del data
            torch.cuda.empty_cache()
            gc.collect()
        #######
        val_res = model.validation_epoch_end(outputs)
        val_loss, val_score = val_res['loss'], val_res['score']
        
        print('Fold {:.0f}:   val_loss = {:.4f}, val_score = {:.4f}'.format(fold, val_loss, val_score))
        
        print('val_obj_loss={:.4f}, val_reg_loss={:.4f}, val_cls_loss={:.4f}'.format(
            val_res['obj_loss'], val_res['reg_loss'], val_res['cls_loss']
        ))
                  
       
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return val_res['pred_df']

# %%
args = CFG 

args.fold = -1 #0
args.seed = 100
args.version = '1'
args.model = CFG.model_name 
args.lr = 6e-5 
args.head_lr = 6e-5 
args.warmup_ratio = 0.05
args.epochs = 5
args.trn_batch_size = 1
args.val_batch_size = 1
args.mask_prob = 0.7
args.mask_ratio = 0.25
args.accumulate_grad_batches = 4
args.gradient_clip_val = 1.0 
args.num_classes = 7
args.check_pointing = True
args.hidden_drop_prob = 0.1
args.p_drop = 0.0
args.loss = 'xentropy'
args.pretrain_path = 'none'
args.early_stopping = 'false'
args.patience = 100
args.slack_url = 'none'
args.eval_step = -1
args.stop_epoch = 5
args.test_score_thr = 0.5 # 0.4, 0.6 
args.fp16 = 'false' #'true'


args.pretrain_path = 'none'
print(args.pretrained_detector_path)

# %%

if CFG.train:
    train_df, valid_df = get_data(fold=args.fold)

    print('train.shape:', train_df.shape,'\t','test.shape:', valid_df.shape)

    train_df['discourse_effectiveness_str'] = train_df['discourse_effectiveness']
    train_df['discourse_effectiveness'] = train_df['discourse_effectiveness'].map(ner_labels_eff)

    valid_df['discourse_effectiveness_str'] = valid_df['discourse_effectiveness']
    valid_df['discourse_effectiveness'] = valid_df['discourse_effectiveness'].map(ner_labels_eff)
    # train_df['discourse_effectiveness'].value_counts()

# %%


if CFG.train:
    train_fold(args, train_df, valid_df, pseudo_df=None, fold=args.fold)




# %% [markdown]
# # Infer args

ap = argparse.ArgumentParser()
ap.add_argument('--model_paths', nargs='+', required=True)
ap.add_argument("--model_name", type=str, required=True)
ap.add_argument("--save_name", type=str, required=True)
ap.add_argument("--fold", type=int, required=True, default=-1)
ap.add_argument("--epoch", type=int, required=True, default=4)
ap.add_argument("--save_dir", type=str, required=True)
ap.add_argument("--test_th", type=float, default=0.50)
ap.add_argument("--debug", type=bool, default=False, required=False)
cargs = ap.parse_args()


TEST_CSV_PATH  = '/workspace/data/test.csv'
TOKENIZER_PATH = '/workspace/checkpoints/hf_pretrained/deberta-xlarge/' # args.model 

FOLD = cargs.fold
EPOCH = cargs.epoch
TEST_TH = cargs.test_th
OUTPUT = cargs.save_dir
SAVE_NAME = cargs.save_name
MODEL_CKP = cargs.model_paths
print('num checkpoints:', len(MODEL_CKP))

CFG.infer = True   


# %%
import codecs
from text_unidecode import unidecode
from typing import Any, Optional, Tuple


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end

def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


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

    print(df_sub.shape)
    print(df_sub.id.nunique())
 
    sample_sub = pd.read_csv(DIR / 'sample_submission.csv')

    sub = df_sub.copy()
    sub = sub.rename(columns={'id': 'essay_id_comp'})

    sub = sub[sample_sub.columns]

    # display(sub.head())
    print(sub.essay_id_comp.nunique() == 10402)
    
    ###
    if verbose:
        print(pred_df['class'].value_counts())

    return sub

# %%
if CFG.infer:

    test = pd.read_csv(TEST_CSV_PATH)
    test.columns = ['id', 'essay_text']
    test['essay_id'] = test['id'] 

    test['essay_text'] = test['essay_text'].apply(resolve_encodings_and_normalize)
    print('apply resolve_encodings..')

    print(test.shape)
    # display(test.head(2))


    if 'deberta-v2' in args.model or 'deberta-v3' in args.model:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(TOKENIZER_PATH, trim_offsets=False)
        special_tokens_dict = {'additional_special_tokens': ['\n\n'] }
        _ = tokenizer.add_special_tokens(special_tokens_dict)
    else:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trim_offsets=False)

    CFG.tokenizer = tokenizer

    test_ds = SpanDetectorDatasetTest(test, tokenizer=CFG.tokenizer)
    # print(test_ds[0].keys())
    
    test_loader = DataLoader(test_ds,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=CustomCollatorTest(CFG.tokenizer, CFG.model_name)
                            )
    
if CFG.infer:

    for fold_i, weight_path in enumerate(MODEL_CKP):
        print(f"Fold {fold_i}: {weight_path}")
        
        model = TextSpanDetector(
                args.model,
                tokenizer,
                num_classes=args.num_classes,
                dynamic_positive=True,
                with_cp=(args.check_pointing=='true'),
                hidden_dropout_prob=args.hidden_drop_prob, 
                p_drop=args.p_drop,
                scheduler=args.scheduler,
                learning_rate=args.lr,
                head_learning_rate=args.head_lr,
                freeze_layers='false',
            )
        model.load_state_dict(torch.load(weight_path), strict=False)
        model = model.cuda()
        model.eval()
        print("Loaded model weights", weight_path)
        print('test th:', TEST_TH)
        
        outputs = []
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            with torch.no_grad():
                output = model.test_step(batch, test_score_thr = TEST_TH)
                
                if len(output['pred_df'])==0:
                    # continue
                    print('--> batch_idx', batch_idx, batch['text_id'])
                    output = model.test_step(batch, test_score_thr = TEST_TH-0.10) #0.4
                    print('infer with lower th', output['pred_df'].shape)
                    if len(output['pred_df'])==0:
                        output = model.test_step(batch, test_score_thr = TEST_TH-0.15)
                        print('infer with lower th', output['pred_df'].shape)
                        if len(output['pred_df'])==0:
                            output = model.test_step(batch, test_score_thr = TEST_TH-0.20)
                            print('infer with lower th', output['pred_df'].shape)
                            if len(output['pred_df'])==0:
                                output = model.test_step(batch, test_score_thr = TEST_TH-0.30)
                                print('infer with lower th', output['pred_df'].shape)
                                if len(output['pred_df'])==0:
                                    output = model.test_step(batch, test_score_thr = 0.30)
                                    print('infer with lower th', output['pred_df'].shape)
                                    if len(output['pred_df'])==0:
                                        output = model.test_step(batch, test_score_thr = 0.20)
                                        print('infer with lower th 0.20', output['pred_df'].shape)
                                        if len(output['pred_df'])==0:
                                            output = model.test_step(batch, test_score_thr = 0.10)
                                            print('infer with lower th 0.10', output['pred_df'].shape)

                outputs.append(output)
        del model
        torch.cuda.empty_cache()

        print(outputs[0].keys())
        print(len(outputs))

        pred_dfs = []
        discourse_ids = []
        prob_seqs = []
        for o in outputs:
            pred_dfs.append(o['pred_df'])

            
        pred_df = pd.concat(pred_dfs, axis=0).reset_index(drop=True)

        # save_pkl(outputs, OUTPUT + f"test_raw.pkl")
        pred_df.to_csv(OUTPUT + f"test_preds.csv", index=False)
        print('saved ok')

        # ####
        sub = generate_submission(pred_df, test)
        sub.to_csv(OUTPUT + f"{SAVE_NAME}.csv", index=False)
        # sub.to_csv(OUTPUT + f"sub_f{FOLD}_ep{EPOCH}_th{TEST_TH}.csv", index=False)


        print('sub saved ok')
        # print(sub['discourse_type'].value_counts())
        assert sub.isnull().sum().sum()==0, "Error: Check NaN!"

        gc.collect()
        torch.cuda.empty_cache()






# %%
## Overlap Removal
def modified(pred_string, diff_words):
    return len(pred_string) > len(diff_words)

def get_original(modified, pred_string):
    if modified:
        return pred_string
    else:
        return "--"

def remove_overlaps(pred_df, by_discourse=False, verbose=False, save=False):
    pred_df['predictionstring'] = [[int(x) for x in pred.split(' ')] for pred in pred_df['predictionstring']]  # Map to integers. Keep as list for better vectorization.
    pred_df["nb_words"] = pred_df["predictionstring"].map(lambda x: len(x))
    pred_df["start_word"] = pred_df["predictionstring"].map(lambda x: min(x))  # Some Dataframes already have this information, but computing in case they don't in future
    # pred_df["end_word"] = pred_df["predictionstring"].map(lambda x: max(x))

    if by_discourse:  # If only removing overlaps within the same discourse element
        grouped_df = pred_df.groupby(["id", "discourse_type"])  # Group by essay and discourse type
    else:
        grouped_df = pred_df.groupby("id") # Group by essay only
    
    processed_groups = []
    removed_groups = []  # Keep track of cleaned up segments
    nb_groups = len(grouped_df)
    pbar = tqdm(total=nb_groups)
    segments_affected = 0
    for name, grp in grouped_df:
        pbar.update(1)
        grp_sorted = grp.sort_values(["start_word",'nb_words'],ascending=True)
        
        grp_sorted["cum_words"] = grp_sorted["predictionstring"].cumsum().apply(set)  # Compute cumulative words seen so far
        grp_sorted["cum_words"] = np.concatenate((np.array([{}]), grp_sorted["cum_words"].iloc[0:-1])) # Shift by 1
        # Need to shift to consider cumulation up to but excluding current segment.
        grp_sorted["set_diff_words"] = (grp_sorted["predictionstring"].map(set) - grp_sorted["cum_words"].map(set)).apply(list)  # Compute set difference with current segment in a vectorized way
        # Reference: https://stackoverflow.com/questions/28457149/how-to-map-a-function-using-multiple-columns-in-pandas -> ListComp faster than apply for some reason.
        grp_sorted["modified"] = [modified(pred, diff_w) for pred, diff_w in zip(grp_sorted["predictionstring"], grp_sorted["set_diff_words"])]  # Check if set difference is shorter (i.e., there is an overlap) in a vect way
        grp_sorted["noncontiguous"] = grp_sorted["set_diff_words"].map(lambda x: np.any(np.diff(x) > 1))  # Non-contiguous segments post overlap removal will be deleted
        grp_sorted["nonzero"] = grp_sorted["set_diff_words"].map(lambda x: len(x) > 0)  # Same for empty segments
        grp_sorted['original_predstring'] = [get_original(mod, pred) for mod, pred in zip(grp_sorted["modified"], grp_sorted["predictionstring"])]
        grp_sorted["predictionstring"] = grp_sorted["set_diff_words"].map(lambda x: " ".join([str(y) for y in x]))
        # Now filter out
        grp_processed = grp_sorted[((~grp_sorted["noncontiguous"]) & grp_sorted["nonzero"])]
        processed_groups.append(grp_processed)
        grp_removed = grp_sorted[((grp_sorted["noncontiguous"]) | ~grp_sorted["nonzero"])]
        removed_groups.append(grp_removed)
        
        if verbose and len(grp_removed) > 0:
            print("Discarded segments")
            print(grp_removed)
    
    # End of all group for loops
    new_segments = pd.concat(processed_groups) #[["id", "discourse_type", "score_discourse_effectiveness", "score_discourse_effectiveness_0", 
                                            #    "score_discourse_effectiveness_1", 
                                            #    "predictionstring", "modified", "original_predstring"]]#.to_csv("processed.csv")
    removed_segments = pd.concat(removed_groups) #[["id", "discourse_type", "score_discourse_effectiveness", "score_discourse_effectiveness_0", 
                                            #    "score_discourse_effectiveness_1", 
                                                #  "predictionstring", "modified", "original_predstring"]]#.to_csv("removed.csv")
    if verbose:
        print(str(segments_affected) + " segments have been modified/removed by pre-processing")
    pbar.close()

    if save:
        new_segments.to_csv("./preds_clean.csv")
        removed_segments.to_csv("./preds_removed.csv")

    return new_segments, removed_segments

# # %% 

# # ### Remove overlaps (only in debug mode)
# if CFG.debug:
#     try:
#         pred_clean, pred_removed = remove_overlaps(pred_df, verbose=False)
#         print(pred_clean.shape)
#         print(pred_removed.shape)
#         print(pred_df['discourse_type'].value_counts())
#         print(pred_clean['discourse_type'].value_counts())
#         print(pred_removed['discourse_type'].value_counts())
#     except:
#         pass

