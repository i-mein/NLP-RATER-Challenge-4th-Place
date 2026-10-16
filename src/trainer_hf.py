import os, sys, gc 
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import random
import time
from pathlib import Path 
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold 
from sklearn.preprocessing import OneHotEncoder
import collections
from types import SimpleNamespace
import re
import yaml
from functools import partial
import multiprocessing as mp
from scipy.special import softmax

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

from transformers import (AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, 
                          TrainingArguments, 
                          Trainer,
                          TrainerCallback)

import wandb
import warnings 
warnings.filterwarnings('ignore')

### -----------------------------------------
### args
### -----------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--config", type=str)
ap.add_argument("--fold", type=int)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--pretrained_model", type=str)
cargs = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(cargs.gpu)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if ("deberta-v3" in cargs.pretrained_model)|("deberta-v2" in cargs.pretrained_model):
    from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast as AutoTokenizer
else:
    from transformers import AutoTokenizer

### -----------------------------------------
### Read config file
### -----------------------------------------

# cfg = json.load(open('SETTINGS.json', 'r'))
# read yaml file 

with open(cargs.config) as file:
    cfg = yaml.safe_load(file)

for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
CFG = SimpleNamespace(**cfg)
# print(cfg)

# Assign variables
# DATA_BASE_DIR = cfg["DATA_BASE_DIR"]
# LR = cfg["LR"]
# NUM_EPOCHS = cfg["NUM_EPOCHS"]
# NUM_CORES = cfg["NUM_CORES"]
# BATCH_SIZE = cfg["BATCH_SIZE"]
# USE_FP16 = cfg["USE_FP16"]
# GRAD_ACCUM_STEPS = cfg["GRAD_ACCUM_STEPS"]
# MAX_SEQ_LENGTH = cfg["MAX_SEQ_LENGTH"]
# SPLIT_NUM = cargs.fold
# PRETRAINED_MODEL = cargs.pretrained_model

os.chdir('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3')
from utils.utils import *

DIR = Path("/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/")
OUTPUT = Path(CFG.output + CFG.exp_name)
OUTPUT.mkdir(exist_ok=True, parents=True)

# seed everything
CFG.device = device
set_seed(CFG.seed)

if CFG.wandb:
    api_key = os.getenv('wb_api_key_core')
    wandb.login(key=api_key)
    wandb.init(project=CFG.wb_project, entity="core") #
    # anony = None
else:
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'offline'


MIN_TOKENS = {
    "Lead": 6,
    "Position": 3,
    "Evidence": 20,
    "Claim": 1,
    "Concluding Statement": 3,
    "Counterclaim": 7,
    "Rebuttal": 6
}


### -----------------------------------------
### Utils
### -----------------------------------------

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

class TextOverlapFBetaScore:
    def __init__(self, test_df, test_dataset):
        self.test_df = test_df
        self.test_dataset = test_dataset

    def __call__(self, raw_predictions):
        predictions, _ = raw_predictions
        soft_predictions = softmax(predictions, -1)
        soft_predictions = np.max(soft_predictions, axis=-1)
        predictions = np.argmax(predictions, axis=-1)

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



### -----------------------------------------
### Data
### -----------------------------------------

df_folds = pd.read_csv(DIR / CFG.TRAIN_CSV) #'train_fold10_sgkf.csv'

print('train folds shape:', df_folds.shape)
# df_folds.head()

ner_labels = ['O']
for curr_label in df_folds['discourse_type'].unique():
    ner_labels.append('B-' + curr_label)
    ner_labels.append('I-' + curr_label)
ner_labels = dict((x,i) for i,x in enumerate(ner_labels))
print(ner_labels)

inverted_ner_labels = dict((v,k) for k,v in ner_labels.items())
inverted_ner_labels[-100] = 'Special Token'


FOLD_NUM = cargs.fold #CFG.trn_folds[0]

train_df = df_folds[df_folds.fold!=FOLD_NUM].reset_index(drop=True)
valid_df = df_folds[df_folds.fold==FOLD_NUM].reset_index(drop=True)

train_files = train_df['essay_id'].unique()
valid_files = valid_df['essay_id'].unique()

print('num train files', len(train_files))
print('num valid files', len(valid_files))


### -----------------------------------------
### NER Labels
### -----------------------------------------
## generate NER labels 

# accepts file path, returns tuple of (file_ID, txt, NER labels)
def generate_NER_labels_for_file(uuid, df, id_col='essay_id'):

    # curr_id=input_filename.split('.')[0]
    # with open(os.path.join(TRAIN_DIR, '{}.txt'.format(input_filename))) as f:
    #     curr_txt = f.read()
    curr_id = uuid 
    curr_txt = df[df[id_col]==uuid]['essay_text'].values[0]

    # Set all token labels initially to non-label
    curr_labels = [ner_labels['O']] * len(curr_txt)
    # Iterate thru all labels associated w/ file and update labels
    curr_df = df[df[id_col]==curr_id]
    for curr_discourse in curr_df.itertuples():
        curr_discourse_label = curr_discourse.discourse_type 
        for curr_txt_idx in range(curr_discourse.discourse_start, 
                                  min(curr_discourse.discourse_end+1, len(curr_labels))):
            if curr_txt_idx == curr_discourse.discourse_start:
                iob_label = ner_labels['B-' + curr_discourse_label]
            else:
                iob_label = ner_labels['I-' + curr_discourse_label]
            curr_labels[curr_txt_idx] = iob_label
    assert curr_labels != [ner_labels['O']] * len(curr_txt)
    return curr_id, curr_txt, curr_labels


with mp.Pool(CFG.NUM_CORES) as p:
    ner_train_rows = p.map(partial(generate_NER_labels_for_file, df=train_df), train_files)

with mp.Pool(CFG.NUM_CORES) as p:
    ner_valid_rows = p.map(partial(generate_NER_labels_for_file, df=valid_df), valid_files)


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


tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
data_collator = DataCollatorForTokenClassification(tokenizer)
tokenizer.save_pretrained(OUTPUT / 'tokenizer/')

# Check is rust-based fast tokenizer
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

tokenized_all_train = tokenize_and_align_labels(ner_train_rows)
tokenized_all_valid = tokenize_and_align_labels(ner_valid_rows)

print('train tokenized', len(tokenized_all_train))
print('valid tokenized', len(tokenized_all_valid))


### -----------------------------------------
### Dataset
### -----------------------------------------

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


train_dataset = NERDataset(tokenized_all_train)
valid_dataset = NERDataset(tokenized_all_valid)

print('len train dataset', len(train_dataset))
print('len valid dataset', len(valid_dataset))

### -----------------------------------------
### HF Trainer 
### -----------------------------------------

model = AutoModelForTokenClassification.from_pretrained(
    CFG.model_name, 
    num_labels=len(ner_labels),
    trust_remote_code=True
    )

model_name = CFG.model_name.split('/')[-1]
save_dir = f'{OUTPUT.as_posix()}/{model_name}-f{FOLD_NUM}'

args = TrainingArguments(
    save_dir,
    CFG.model_name,
    lr_scheduler_type='linear',
    evaluation_strategy = 'steps',
    eval_steps=500,
    dataloader_num_workers=4, #8,
    learning_rate=CFG.LR,
    per_device_train_batch_size=CFG.batch_size, 
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=CFG.GRAD_ACC,
    gradient_checkpointing=True,
    num_train_epochs=CFG.epochs,
    save_strategy='no',
    save_total_limit=1,
    report_to='wandb' if CFG.wandb else 'none',
    log_level='warning',
    fp16 = CFG.fp16,
    )

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









