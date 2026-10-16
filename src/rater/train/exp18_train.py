# %%
# ## Setup
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.utils.checkpoint



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

from transformers import (AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, 
                          TrainingArguments, 
                          Trainer,
                          TrainerCallback)

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import warnings 
warnings.filterwarnings('ignore')

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


class CFG:

    exp_name = 'exp18'
    task = "ner"

    n_folds = 5 # 10
    trn_folds = [0] # [0,1,2,3,4]

    model_name = "microsoft/deberta-large"
    # model_name = "microsoft/deberta-base"
    # "allenai/longformer-base-4096"
    # "microsoft/deberta-large"
    # max_length = 512 ##1536 #1024
    max_length = 1536 #768  
    stride = 128
    min_tokens = 6
    # TRAINING HYPERPARAMS
    batch_size = 16 #32 #4       # 16
    epochs = 4 #1
    
    GRAD_ACC = 1 #2 #4  #   8 # 4 ## 8
    LR = 2e-5    # 4e-5    ## 3e-05
    WD = 0.01
    WARMUP = 0.1
    use_custom_trainer = False # set True for custom training loop - TODO
    fp16 = True 
    seed = 42
    train = True
    debug = False # set True for debugging
    wandb = False
    wb_project = "NLP-Rater"
    output = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/"

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

    

# %%
import yaml 
import importlib


# config = yaml.safe_load(open(parser_args.config).read()) ## {CFG.exp_name}
with open(f'/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/config/exp02.yaml', 'r') as f:
    config = yaml.safe_load(f)

for k, v in config.items():
    if type(v) == dict:
        config[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**config)
print(cfg)

# %%
os.chdir('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3')

from utils.utils import *

cfg.experiment_name = CFG.exp_name

DIR = Path("/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/")
OUTPUT = Path(CFG.output + CFG.exp_name)
OUTPUT.mkdir(exist_ok=True, parents=True)


# seed everything
CFG.device = device
set_seed(CFG.seed)

# %%
if CFG.wandb:
    api_key = os.getenv('wb_api_key_core')
    wandb.login(key=api_key)
    wandb.init(project=CFG.wb_project, entity="core") #
    # anony = None

# %%
# cfg.dataset.id_column = 'essay_id' #'essay_id_comp' #'essay_id'



# # Utils


# CODE FROM : Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter/ len_pred
    return [overlap_1, overlap_2]

def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """

    if "essay_id" in pred_df.columns:
        pred_df = pred_df.rename(columns={"essay_id": "id"})
    
    if "essay_id" in gt_df.columns:
        gt_df = gt_df.rename(columns={"essay_id": "id"})

    # print(pred_df.shape, gt_df.shape)
    # print(pred_df.head(1))
    # print(gt_df.head(1))

    gt_df = gt_df[['id','discourse_type','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id','class','predictionstring']] \
        .reset_index(drop=True).copy()
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

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])
    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1','overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id','predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    #calc microf1
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score

def generate_token_to_word_mapping(txt, offset):
    # GET WORD POSITIONS IN CHARS
    w = []
    blank = True
    for i in range(len(txt)):
        if not txt[i].isspace() and blank==True:
            w.append(i)
            blank=False
        elif txt[i].isspace():
            blank=True
    w.append(1e6)

    # MAPPING FROM TOKENS TO WORDS
    word_map = -1 * np.ones(len(offset),dtype='int32')
    w_i = 0
    for i in range(len(offset)):
        if offset[i][1]==0: continue
        while offset[i][0]>=w[w_i+1]: w_i += 1
        word_map[i] = int(w_i)
        
    return word_map



class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        self.bestScore = 0

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.evaluation_strategy != "no", "SaveBestModelCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_value = metrics.get("eval_textoverlapfbeta")
        if metric_value > self.bestScore:
            print(f"** TextOverlapFBeta score improved from {np.round(self.bestScore, 4)} to {np.round(metric_value, 4)} **")
            self.bestScore = metric_value
            control.should_save = True
        else:
            print(f"TextOverlapFBeta score {np.round(metric_value, 4)} (Prev. Best {np.round(self.bestScore, 4)}) ")


MIN_TOKENS = CFG.MIN_TOKENS

class_to_label = {
    'Claim': 0, 
    'Evidence': 1, 
    'Lead':2, 
    'Position':3, 
    'Concluding Statement':4,
    'Counterclaim':5, 
    'Rebuttal':6
}

# Threshold found from CV
label_to_threshold = {
    0 : 0.275, #Claim
    1 : 0.375, #Evidence
    2 : 0.325, #Lead
    3 : 0.325, #Position
    4 : 0.4, #Concluding Statement
    5 : 0.275, #Counterclaim
    6 : 0.275 #Rebuttal
}

label_to_class = {v:k for k, v in class_to_label.items()}



# train_ori = pd.read_csv("/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/train.csv")
# train_ori = train_ori.dropna(subset=['discourse_effectiveness'])
# train_ori = train_ori.reset_index(drop=True)
# train_ori.shape

# # Data
df_folds = pd.read_csv(DIR / 'train_sgkf_fold5.csv')
df_folds['essay_id'] = df_folds['essay_id_comp']

print('train folds shape:', df_folds.shape)
# df_folds.head()


## add whole essays 
df_texts = pd.read_csv('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/train_fold10_sgkf.csv')
df_texts = df_texts[['essay_id', 'discourse_id', 'essay_text']]

df_folds = df_folds.merge(df_texts, on=['essay_id', 'discourse_id'])

# pd.crosstab(df_folds.fold, df_folds.discourse_effectiveness)

# pd.crosstab(df_folds.fold, df_folds.discourse_type)

ner_labels = ['O']
for curr_label in df_folds['discourse_type'].unique():
    ner_labels.append('B-' + curr_label)
    ner_labels.append('I-' + curr_label)
ner_labels = dict((x,i) for i,x in enumerate(ner_labels))
# ner_labels


inverted_ner_labels = dict((v,k) for k,v in ner_labels.items())
inverted_ner_labels[-100] = 'Special Token'

ner_labels_eff = ['O']
for curr_label in df_folds['discourse_effectiveness'].unique():
    ner_labels_eff.append('B-' + curr_label)
    ner_labels_eff.append('I-' + curr_label)
ner_labels_eff = dict((x,i) for i,x in enumerate(ner_labels_eff))
ner_labels_eff

inverted_ner_labels_eff = dict((v,k) for k,v in ner_labels_eff.items())
inverted_ner_labels_eff[-100] = 'Special Token'


FOLD_NUM = CFG.trn_folds[0]

## try a single fold -- TODO convert to KFold training

train_df = df_folds[df_folds.fold!=FOLD_NUM].reset_index(drop=True)
valid_df = df_folds[df_folds.fold==FOLD_NUM].reset_index(drop=True)

train_files = train_df['essay_id'].unique()
valid_files = valid_df['essay_id'].unique()

print('num train files', len(train_files))
print('num valid files', len(valid_files))


# # NER Labels


## generate NER labels 

# accepts file path, returns tuple of (file_ID, txt, NER labels)
def generate_NER_labels_for_file(uuid, df, id_col='essay_id'):

    # curr_id=input_filename.split('.')[0]
    # with open(os.path.join(TRAIN_DIR, '{}.txt'.format(input_filename))) as f:
    #     curr_txt = f.read()
    curr_id = uuid 
    curr_txt = df[df[id_col]==uuid]['essay_text'].values[0].rstrip()

    # Set all token labels initially to non-label
    curr_labels = [ner_labels['O']] * len(curr_txt)

    curr_labels_eff = [ner_labels_eff['O']] * len(curr_txt)
    
    # Iterate through all labels associated w/ file and update labels
    curr_df = df[df[id_col]==curr_id]
    for curr_discourse in curr_df.itertuples():

        curr_discourse_label     = curr_discourse.discourse_type 
        curr_discourse_label_eff = curr_discourse.discourse_effectiveness 

        for curr_txt_idx in range(curr_discourse.discourse_start, 
                                  min(curr_discourse.discourse_end+1, len(curr_labels))):
            if curr_txt_idx == curr_discourse.discourse_start:
                iob_label     = ner_labels['B-' + curr_discourse_label]
                iob_label_eff = ner_labels_eff['B-' + curr_discourse_label_eff]

            else:
                iob_label     = ner_labels['I-' + curr_discourse_label]
                iob_label_eff = ner_labels_eff['I-' + curr_discourse_label_eff]

            curr_labels[curr_txt_idx]     = iob_label
            curr_labels_eff[curr_txt_idx] = iob_label_eff

    assert curr_labels != [ner_labels['O']] * len(curr_txt)
    assert curr_labels_eff != [ner_labels_eff['O']] * len(curr_txt)

    return curr_id, curr_txt, curr_labels, curr_labels_eff

### ucomment to check one sample 

# t_labels = generate_NER_labels_for_file(train_files[10], df=train_df)

# print(np.unique(t_labels[2]))
# print(np.unique(t_labels[3]))
# len(t_labels[2]), len(t_labels[3])





with mp.Pool(CFG.NUM_CORES) as p:
    ner_train_rows = p.map(partial(generate_NER_labels_for_file, df=train_df), train_files)


with mp.Pool(CFG.NUM_CORES) as p:
    ner_valid_rows = p.map(partial(generate_NER_labels_for_file, df=valid_df), valid_files)


# len(ner_train_rows), len(ner_valid_rows)


tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

# Check is rust-based fast tokenizer
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


# tokenizer.save_pretrained(OUTPUT / 'tokenizer/')



def tokenize_and_align_labels(ner_raw_data):
    tokenized_inputs = tokenizer(
        [x[1] for x in ner_raw_data], 
        max_length=CFG.max_length,
        return_offsets_mapping=True,
        truncation=True
        )
    labels = []
    word_ids = []
    for i, char_label in enumerate([x[2] for x in ner_raw_data]):
        curr_word_ids = tokenized_inputs.word_ids(batch_index=i)
        curr_offset_mappings = tokenized_inputs['offset_mapping'][i]
        label_ids = []
        for j, word_idx in enumerate(curr_word_ids):
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            else:
                # Use label of 1st character of word
                # See offset that's 1 index after end of tokens for some reason
                char_idx = min(curr_offset_mappings[j][0], len(char_label)-1)
                label_ids.append(char_label[char_idx])
        word_ids.append(curr_word_ids)
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    tokenized_inputs['word_ids'] = word_ids
    tokenized_inputs['essay_id'] = [x[0] for x in ner_raw_data]
    return tokenized_inputs


## modify for multitask 

def tokenize_and_align_labels2(ner_raw_data):
    tokenized_inputs = tokenizer(
        [x[1] for x in ner_raw_data], 
        max_length=CFG.max_length,
        return_offsets_mapping=True,
        truncation=True
        )
    labels = []
    labels2 = []
    word_ids = []
    for i, (char_label, char_label2) in enumerate(zip(
        [x[2] for x in ner_raw_data],  [x[3] for x in ner_raw_data]
        )):
        
        curr_word_ids = tokenized_inputs.word_ids(batch_index=i)
        curr_offset_mappings = tokenized_inputs['offset_mapping'][i]
        
        label_ids = []
        label_ids2 = []
        for j, word_idx in enumerate(curr_word_ids):
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
                label_ids2.append(-100)
            else:
                # Use label of 1st character of word
                # See offset that's 1 index after end of tokens for some reason
                char_idx = min(curr_offset_mappings[j][0], len(char_label)-1)
                label_ids.append(char_label[char_idx])

                char_idx2 = min(curr_offset_mappings[j][0], len(char_label2)-1)
                label_ids2.append(char_label2[char_idx2])


        word_ids.append(curr_word_ids)
        labels.append(label_ids)
        labels2.append(label_ids2)

    tokenized_inputs['labels'] = labels
    tokenized_inputs['labels_eff'] = labels2
    tokenized_inputs['word_ids'] = word_ids
    tokenized_inputs['essay_id'] = [x[0] for x in ner_raw_data]
    return tokenized_inputs


tokenized_all_train = tokenize_and_align_labels2(ner_train_rows)


tokenized_all_valid = tokenize_and_align_labels2(ner_valid_rows)

print('train tokenized', len(tokenized_all_train['labels']))
print('valid tokenized', len(tokenized_all_valid['labels']))

# tokenized_all_train[0]

# tokenized_all_valid[0]

# tokenized_all_train[0].tokens

# tokenized_all_train[0].special_tokens_mask



### --------------------------
###  NER Dataset 
### --------------------------

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict
        
    def __getitem__(self, index):
        return {k:self.input_dict[k][index] for k in self.input_dict.keys() if k not in {'essay_id', 'offset_mapping', 'word_ids'}}
    
    def get_filename(self, index):
        return self.input_dict['essay_id'][index]
    
    def get_offset(self, index):
        return self.input_dict['offset_mapping'][index]
    
    def __len__(self):
        return len(self.input_dict['input_ids'])


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

### --------------------------
### Custom Data Collator 
### --------------------------


@dataclass
class DataCollatorForTokenClassification2(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        label_name2 = "labels_eff" 
        labels2 = [feature[label_name2] for feature in features] if label_name2 in features[0].keys() else None

        # no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
        no_labels_features = [{k: v for k, v in feature.items() if k not in [label_name, label_name2]} for feature in features]

        batch = self.tokenizer.pad(
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]

        if padding_side == "right":
            batch[label_name2] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels2
            ]
        else:
            batch[label_name2] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels2
            ]

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        batch[label_name2] = torch.tensor(batch[label_name2], dtype=torch.int64)
        return batch

# %%
train_dataset = NERDataset(tokenized_all_train)
valid_dataset = NERDataset(tokenized_all_valid)

print('len train dataset', len(train_dataset))
print('len valid dataset', len(valid_dataset))



# data_collator = DataCollatorForTokenClassification(tokenizer)

## use custom 
data_collator = DataCollatorForTokenClassification2(tokenizer)

# # Metric


## FP1 metric 

class TextOverlapFBetaScore:
    def __init__(self, test_df, test_dataset):
        self.test_df = test_df
        self.test_dataset = test_dataset

    def __call__(self, raw_predictions):
        # print(len(raw_predictions))
        # print(raw_predictions)

        predictions, _ = raw_predictions
        # predictions, predictions_eff, _ = raw_predictions

        predictions = predictions[0]
        predictions_eff = predictions[1]

        try:
            print(predictions[0].shape) # (3120, 1536, 15)
            print(predictions[1].shape)
        except:
            pass

        soft_predictions = softmax(predictions, -1)
        soft_predictions = np.max(soft_predictions, axis=-1)
        predictions = np.argmax(predictions, axis=-1)

        # soft_predictions_eff = softmax(predictions_eff, -1)
        # soft_predictions_eff = np.max(soft_predictions_eff, axis=-1)
        # predictions_eff = np.argmax(predictions_eff, axis=-1)


        all_preds = []
        # Clumsy gathering of predictions at word lvl - only populate with 1st subword pred
        for curr_sample_id in range(len(self.test_dataset)):
            sample_preds = predictions[curr_sample_id]
            sample_offset = self.test_dataset.get_offset(curr_sample_id)
            sample_txt = ner_valid_rows[curr_sample_id][1]
            sample_word_map = generate_token_to_word_mapping(sample_txt, sample_offset)

            word_preds = [''] * (max(sample_word_map) + 1)
            word_probs = dict()
            for i, curr_word_id in enumerate(sample_word_map):
                if curr_word_id != -1 and word_preds[curr_word_id] == '': # only use 1st subword
                    word_preds[curr_word_id] = inverted_ner_labels[sample_preds[i]]
                    word_probs[curr_word_id] = soft_predictions[curr_sample_id, i]

            # Dict to hold Lead, Position, Concluding Statement
            let_one_dict = dict() # K = Type, V = (Prob of start token, start, end)

            # If we see tokens I-X, I-Y, I-X in a sequence -> change I-Y to I-X
            for j in range(1, len(word_preds) - 1):
                pred_trio = [word_preds[k] for k in [j - 1, j, j + 1]]
                splitted_trio = [x.split('-')[0] for x in pred_trio]
                if all([x == 'I' for x in splitted_trio]) and pred_trio[0] == pred_trio[2] and pred_trio[0] != pred_trio[1]:
                    word_preds[j] = word_preds[j-1]

            j = 0 # start of candidate discourse
            while j < len(word_preds): 
                cls = word_preds[j] 
                cls_splitted = cls.split('-')[-1]
                end = j + 1 # try to extend discourse as far as possible

                if word_probs[j] > 0.63: 
                    # Must match suffix i.e., I- to I- only; no B- to I-
                    while end < len(word_preds) and (word_preds[end].split('-')[-1] == cls_splitted if cls_splitted in ['Lead', 'Position', 'Concluding Statement'] else word_preds[end] == f'I-{cls_splitted}'):
                        end += 1
                    # if we're here, end is not the same pred as start
                    if cls != 'O' and end - j > MIN_TOKENS[cls_splitted]: # needs to be longer than class-specified min
                        if cls_splitted in ['Lead', 'Position', 'Concluding Statement']:
                            lpc_max_prob = max(word_probs[c] for c in range(j, end))
                            if cls_splitted in let_one_dict: # Already existing, check contiguous or higher prob
                                prev_prob, prev_start, prev_end = let_one_dict[cls_splitted]
                                if j - prev_end < 3: # If close enough, combine
                                    let_one_dict[cls_splitted] = (max(prev_prob, lpc_max_prob), prev_start, end)
                                elif lpc_max_prob > prev_prob: # Overwrite if current candidate is more likely
                                    let_one_dict[cls_splitted] = (lpc_max_prob, j, end)
                            else: # Add to it
                                let_one_dict[cls_splitted] = (lpc_max_prob, j, end)
                        else:
                            # Lookback and add preceding I- tokens
                            while j - 1 > 0 and word_preds[j-1] == cls:
                                j = j - 1
                            # Try to add the matching B- tag if immediately precedes the current I- sequence
                            if j - 1 > 0 and word_preds[j-1] == f'B-{cls_splitted}':
                                j = j - 1

                            all_preds.append((self.test_dataset.get_filename(curr_sample_id), 
                                              cls_splitted, 
                                              ' '.join(map(str, list(range(j, end+1))))))

                j = end 

            # Add the Lead, Position, Concluding Statement
            for k, v in let_one_dict.items():
                pred_start = v[1]
                # Lookback and add preceding I- tokens
                while pred_start - 1 > 0 and word_preds[pred_start-1] == f'I-{k}':
                    pred_start = pred_start -1
                # Try to add the matching B- tag if immediately precedes the current I- sequence
                if pred_start - 1 > 0 and word_preds[pred_start - 1] == f'B-{k}':
                    pred_start = pred_start - 1

                all_preds.append((self.test_dataset.get_filename(curr_sample_id), 
                                  k, 
                                  ' '.join(map(str, list(range(pred_start, v[2]))))))

        output_df = pd.DataFrame(all_preds)
        output_df.columns = ['id', 'class', 'predictionstring']

        f1s = []
        CLASSES = output_df['class'].unique()
        for c in CLASSES:
            pred_df = output_df.loc[output_df['class']==c].copy()
            gt_df = self.test_df.loc[self.test_df['discourse_type']==c].copy()
            f1 = score_feedback_comp(pred_df, gt_df)
            f1s.append(f1)

        return {"textoverlapfbeta": np.mean(f1s)}


# # HF Model 

# %%
# model = AutoModelForTokenClassification.from_pretrained(
#     CFG.model_name, 
#     num_labels=len(ner_labels),
#     trust_remote_code=True
#     )


CFG.num_labels = len(ner_labels)
CFG.num_labels_eff = len(ner_labels_eff)

# %%
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput


@dataclass
class TokenClassifierOutput2(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits_eff: torch.FloatTensor = None
    loss1: Optional[torch.FloatTensor] = None
    loss2: Optional[torch.FloatTensor] = None


from typing import Dict, List, NamedTuple, Optional, Tuple, Union

class EvalPrediction(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray
    classes: List

class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    classes: Optional[List]



# %%
from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaModel
# from transformers.modeling_outputs import TokenClassifierOutput


class DebertaForTokenClassificationLSTM2(DebertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = CFG.num_labels #config.num_labels # 15 
        self.num_labels_eff = CFG.num_labels_eff # 5 

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, num_layers=2, dropout=config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.head_eff = nn.Linear(config.hidden_size, self.num_labels_eff)
        
        self.loss_fct = nn.CrossEntropyLoss()
        self.loss_fct2 = nn.CrossEntropyLoss()
        self.loss_weights = [1.0, 1.0]

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        labels_eff=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels  for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        Labels2 for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)

        logits = self.classifier(lstm_output)
        # print('logits', logits.shape)
        # print('logits', np.unique(labels.detach().cpu().numpy()))

        logits_eff = self.head_eff(lstm_output)
        # print('logits_eff', logits_eff.shape)
        # print('logits_eff', np.unique(labels_eff.detach().cpu().numpy()))

        loss = None
        if labels is not None and labels_eff is not None:
            # loss_fct = nn.CrossEntropyLoss()
            loss1 = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss2 = self.loss_fct2(logits_eff.view(-1, self.num_labels_eff), labels_eff.view(-1))

            loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2 

        if not return_dict:
            # output = (logits,) + outputs[1:]
            output = (logits,) + (logits_eff,) + outputs[1:]
  
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput2(
            loss=loss, 
            logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
            logits_eff=logits_eff,
            loss1=loss1, 
            loss2=loss2, 
        )

# %%
# from models.hf_debertabilstmV2 import DebertaForTokenClassificationLSTM2

model = DebertaForTokenClassificationLSTM2.from_pretrained(CFG.model_name)

# %%
# # check

# dl = DataLoader(train_dataset, batch_size=2, collate_fn=data_collator)
# sample = next(iter(dl))
# sample = {k: v.to(CFG.device) for k, v in sample.items()}



# %%
# out = model.forward(**sample, return_dict=False)

# %%
# out[0]
# out[1].shape
# out[2].shape
# len(out)

# %%
# out = model.forward(**sample, return_dict=True)
# print('token clf logits', out.logits.shape)
# print('token eff logits', out.logits_eff.shape)
# print(out.loss, out.loss1, out.loss2)

# %%
loss_fn1 = nn.CrossEntropyLoss()
loss_fn2 = nn.CrossEntropyLoss()
# jaccard_loss = JaccardLoss(log_loss=False, from_logits=True)

CFG.loss_weights = [1.0, 1.0]

# %%
class CustomTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels_eff = inputs.pop("labels_eff")

        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits_eff = outputs.get("logits_eff")
        
        # loss = ce_loss(logits.view(-1, len(ner_labels)), labels.view(-1)) 
        # + jaccard_loss(logits.view(-1, len(ner_labels)), labels.view(-1))

        loss1 = loss_fn1(logits.view(-1, len(ner_labels)), labels.view(-1))
        loss2 = loss_fn2(logits_eff.view(-1, len(ner_labels_eff)), labels_eff.view(-1))

        loss = CFG.loss_weights[0]*loss1 + self.loss_weights[1]*loss2 
        
        return (loss, outputs) if return_outputs else loss
    

    

# %%


# %%
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'offline'

# %%
model_name = CFG.model_name.split('/')[-1]

args = TrainingArguments(
    f'{OUTPUT.as_posix()}/{model_name}-f{FOLD_NUM}',
    CFG.model_name,
    lr_scheduler_type='linear',
    evaluation_strategy = 'steps',
    eval_steps=200, #400 == half epoch,
    dataloader_num_workers=4, #8,
    learning_rate=CFG.LR,
    log_level='warning',
    fp16 = CFG.fp16,
    per_device_train_batch_size=CFG.batch_size, 
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=CFG.GRAD_ACC,
    gradient_checkpointing=True,
    num_train_epochs=CFG.epochs,
    save_strategy='no',
    save_total_limit=1,
    report_to='none', # 'wandb' if CFG.wandb else 'none',
    )


# %%
# trainer = CustomTrainer

trainer = Trainer(
    model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=TextOverlapFBetaScore(test_df=valid_df, test_dataset=valid_dataset),
    callbacks=[SaveBestModelCallback],
    data_collator=data_collator,
    tokenizer=tokenizer
    )

trainer.train()



