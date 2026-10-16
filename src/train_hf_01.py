
## HuggingFace Training Baseline
## conda env: dev_ime_hf 
## author: IME 
## date: 27/02/2024



# ## Setup
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# %%
import os, sys, gc 
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path 

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import DebertaTokenizer, DebertaForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification

import wandb
import warnings 
warnings.filterwarnings('ignore')

print('torch.__version__:', torch.__version__)
print('transformers.__version__:', transformers.__version__)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# # Config
class CFG:

    exp_name = 'exp01'
    task = "ner"

    n_folds = 10 #5 # 10
    trn_folds = [0] # [0,1,2,3,4]

    model_name = "microsoft/deberta-base"
    # "allenai/longformer-base-4096"
    # "microsoft/deberta-large"
    # max_length = 512 ##1536 #1024
    max_length = 768  
    stride = 128
    min_tokens = 6
    # TRAINING HYPERPARAMS
    batch_size = 16 #32 #4       # 16
    epochs = 4 #1
    
    GRAD_ACC = 2 #4  #   8 # 4 ## 8
    LR = 2e-5    # 4e-5    ## 3e-05
    WD = 0.01
    WARMUP = 0.1
    use_custom_trainer = True #False # set True for custom training loop - TODO
    fp16 = True 
    seed = 42
    train = True
    debug = False # set True for debugging
    wandb = False
    wb_project = "NLP-Rater"
    output = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/"


### ------------------------------------------------
### Utils
### ------------------------------------------------

# seed everything
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# %cd /mnt/md1/Work/dev_IME/Git_hacks/Feedback3
os.chdir('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3')
sys.path.append('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/src/')
sys.path.append('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/')


from src.utils import *

DIR = Path("/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/")
OUTPUT = Path(CFG.output + CFG.exp_name)
OUTPUT.mkdir(exist_ok=True, parents=True)


# seed everything
seed_everything(CFG.seed)

# # logger TODO 
# import logging
# logging.basicConfig(filename=OUTPUT/'train.log', level=logging.INFO)
# logger = logging.getLogger(__name__)

# %%
# setup wandb for experiment tracking
# source: https://www.kaggle.com/debarshichanda/pytorch-w-b-jigsaw-starter

if CFG.wandb:
    api_key = os.getenv('wb_api_key_core')
    wandb.login(key=api_key)
    wandb.init(project=CFG.wb_project,) # entity='core' #"ime"
    # anony = None


# ## Data Preprocessing
 
# read train data

train = pd.read_csv(DIR / 'train.csv')
train['discourse_id'] = train['discourse_id'].astype('long').astype('str')
train['discourse_start'] = train['discourse_start'].astype('int')
train['discourse_end'] = train['discourse_end'].astype('int')


print('train shape:', train.shape)
# train.head(3)

# %%
print('num essays:', train.essay_id_comp.nunique())
print('num disc:', train.discourse_id.nunique())
print('discourse_types:', train.discourse_type.nunique())
print('discourse_effectiveness:', train.discourse_effectiveness.nunique())
print('essay scores:', train.holistic_essay_score.nunique())


# train.discourse_type.value_counts()
# train.discourse_effectiveness.value_counts()
 
MIN_TOKENS = {
    "Lead": 6,
    "Position": 3,
    "Evidence": 20,
    "Claim": 1,
    "Concluding Statement": 3,
    "Counterclaim": 7,
    "Rebuttal": 6
}
 
## map effectiveness to int 

eff_map = {
    'Ineffective': 0, 
    'Effective': 1,
    'Adequate': 0.5 # or 1 
    }

train['effectiveness_int'] = train.discourse_effectiveness.map(eff_map)
 
# train.effectiveness_int.value_counts()
# train.holistic_essay_score.value_counts()

# view one row 
# train.iloc[0]['full_text'] == train.iloc[2]['full_text']
# train.iloc[1]['full_text']
# # check unique classes
classes = train.discourse_type.unique().tolist()
classes = classes[1:] # remove Unannotated
print(classes)

### FB1 labels 
# ['Lead',
#  'Position',
#  'Evidence',
#  'Claim',
#  'Concluding Statement',
#  'Counterclaim',
#  'Rebuttal']

# %%
ner_labels = ['O']
for curr_label in classes: #train['discourse_type'].unique():
    ner_labels.append('B-' + curr_label)
    ner_labels.append('I-' + curr_label)
ner_labels = dict((x,i) for i,x in enumerate(ner_labels))

inverted_ner_labels = dict((v,k) for k,v in ner_labels.items())
inverted_ner_labels[-100] = 'Special Token'
# ner_labels

# target_id_map = {
#     "B-Lead": 0,
#     "I-Lead": 1,
#     "B-Position": 2,
#     "I-Position": 3,
#     "B-Evidence": 4,
#     "I-Evidence": 5,
#     "B-Claim": 6,
#     "I-Claim": 7,
#     "B-Concluding Statement": 8,
#     "I-Concluding Statement": 9,
#     "B-Counterclaim": 10,
#     "I-Counterclaim": 11,
#     "B-Rebuttal": 12,
#     "I-Rebuttal": 13,
#     "O": 14,
#     "PAD": -100,
# }

# id_target_map = {v: k for k, v in target_id_map.items()}

 
# setup label indices

from collections import defaultdict
# tags = defaultdict()

# for i, c in enumerate(classes):
#     tags[f'B-{c}'] = i
#     tags[f'I-{c}'] = i + len(classes)
# tags[f'O'] = len(classes) * 2
# tags[f'Special'] = -100
    
# target_id_map = dict(tags)

# id_target_map = defaultdict()
# for k, v in target_id_map.items(): 
#     id_target_map[v] = k
# id_target_map[-100] = 'Special'

# id_target_map = dict(id_target_map)

N_LABELS = len(ner_labels) - 1 # not accounting for -100

# %%
# # read test data

# test = pd.read_csv(DIR / 'test.csv')

# print('test shape:', test.shape)
# print('num test essays:', test.essay_id_comp.nunique())
# # test shape: (10402, 2)
# # num test essays: 10402

# # test.head(3)
# sub = pd.read_csv(DIR / 'sample_submission.csv')
# print('sub shape:', sub.shape)
# sub

# del test
gc.collect()

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

 
# some helper functions

def get_raw_text(ids):
    with open(f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/train/{ids}.txt', 'r') as file: 
        data = file.read()
    return data

def get_test_text(ids):
    with open(f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/test/{ids}.txt', 'r') as file: 
        data = file.read()
    return data

 
# group training data + labels by essay  

t1 = train.groupby('essay_id_comp')['discourse_type'].apply(list).reset_index(name='classlist')
t2 = train.groupby('essay_id_comp')['discourse_start'].apply(list).reset_index(name='starts')
t3 = train.groupby('essay_id_comp')['discourse_end'].apply(list).reset_index(name='ends')
t4 = train.groupby('essay_id_comp')['predictionstring'].apply(list).reset_index(name='predictionstrings')
t5 = train.groupby('essay_id_comp')['discourse_effectiveness'].apply(list).reset_index(name='classlist_effectiveness')
t6 = train.groupby('essay_id_comp')['effectiveness_int'].apply(list).reset_index(name='classlist_effective_int')


df = pd.merge(t1, t2, how='inner', on='essay_id_comp')
df = pd.merge(df, t3, how='inner', on='essay_id_comp')
df = pd.merge(df, t4, how='inner', on='essay_id_comp')
df = pd.merge(df, t5, how='inner', on='essay_id_comp')
df = pd.merge(df, t6, how='inner', on='essay_id_comp')
df['text'] = train.groupby('essay_id_comp')['full_text'].first().values
# df['text'] = df['id'].apply(get_raw_text)

del t1, t2, t3, t4, t5, t6
gc.collect()

# df.head()
print(df.shape)

# df.iloc[0]
# df.iloc[0]['classlist']
# df.iloc[0]['classlist_effectiveness']
# df.iloc[0]['classlist_effective_int']

# debugging
if CFG.debug: df = df.sample(n=100).reset_index(drop=True)


### split to folds 

from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

#https://www.kaggle.com/abhishek/creating-folds-properly-hopefully-p
def make_fold(df, n_folds=10, save=False):
    df = pd.read_csv(DIR / 'train.csv')

    df1 = pd.get_dummies(df, columns=['discourse_type']).groupby(['essay_id_comp'], as_index=False).sum()
    label_col = [c for c in df1.columns if c.startswith('discourse_type_') and c != 'discourse_type_num']
    col = label_col +['essay_id_comp']
    df1 = df1[col]
    df1.loc[:,'fold'] = -1

    mskf  = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_index, valid_index) in enumerate(mskf.split(df1, df1[label_col])):
        df1.loc[valid_index, 'fold'] = fold
        print(f'fold {fold} len train/valid:', len(train_index), len(valid_index))

    df_out = df.merge(df1[['essay_id_comp', 'fold']], on='essay_id_comp', how='left')
    print(df_out.fold.value_counts())
    if save:
        df_out.to_csv(DIR / 'train_fold10.csv', index=False)
    # exit(0)
    return df_out

 
df_folds = make_fold(df, save=False)
# df_folds.head()
print(df_folds.shape)

# df_folds.essay_id_comp.nunique()
# df_folds.groupby('id')['fold'].apply(lambda x: x.tolist()[0])

df = df.merge(
    df_folds.groupby('essay_id_comp')['fold'].apply(lambda x: x.tolist()[0]),
    on='essay_id_comp', how='right')
print(df.shape)
# display(df.head(3))

COLS = [c for c in df.columns if c not in ['fold'] ]
FOLD = 0


# # we will use HuggingFace datasets
# from datasets import Dataset, load_metric

# df_train = df[df.fold!=FOLD][COLS].reset_index(drop=True)
# df_valid = df[df.fold==FOLD][COLS].reset_index(drop=True)

train_df = df[df.fold!=FOLD][COLS].reset_index(drop=True)
valid_df = df[df.fold==FOLD][COLS].reset_index(drop=True)

train_files = train_df['essay_id_comp'].unique()
valid_files = valid_df['essay_id_comp'].unique()

# # datasets = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
# ds_train = Dataset.from_pandas(df_train)
# ds_valid = Dataset.from_pandas(df_valid)



# from transformers import AutoTokenizer
# from transformers import DebertaTokenizer, DebertaForTokenClassification
# from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
from transformers.models.deberta.tokenization_deberta_fast import DebertaTokenizerFast 
from scipy.special import softmax
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
    )


# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
# tokenizer = DebertaTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
tokenizer = DebertaTokenizerFast.from_pretrained(CFG.model_name, add_prefix_space=True)

# %%
# # Not sure if this is needed, but in case we create a span with certain class without starting token of that class,
# # let's convert the first token to be the starting token.

e = [0,7,7,7,1,1,8,8,8,9,9,9,14,4,4,4]

def fix_beginnings(labels):
    for i in range(1,len(labels)):
        curr_lab = labels[i]
        prev_lab = labels[i-1]
        if curr_lab in range(7,14):
            if prev_lab != curr_lab and prev_lab != curr_lab - 7:
                labels[i] = curr_lab -7
    return labels

# fix_beginnings(e)

# %%
# MIN_TOKENS = {
#     "Lead": 6,
#     "Position": 3,
#     "Evidence": 20,
#     "Claim": 1,
#     "Concluding Statement": 3,
#     "Counterclaim": 7,
#     "Rebuttal": 6
# }

# %%
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

# %%
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
            sample_txt = self.test_df.iloc[curr_sample_id]['text'] # ner_valid_rows[curr_sample_id][1]
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



# %%
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




# tokenize and add labels

target_id_map = ner_labels

def tokenize_and_align_labels(examples):

    o = tokenizer(
        examples['text'], 
        truncation=True, 
        padding=True, 
        return_offsets_mapping=True, 
        max_length=CFG.max_length, 
        stride=CFG.stride, 
        return_overflowing_tokens=True
        )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = o["overflow_to_sample_mapping"]
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = o["offset_mapping"]
    
    o["id"] = []
    o["labels"] = []
    for i in range(len(offset_mapping)):
                   
        sample_index = sample_mapping[i]
        labels = [target_id_map['O'] for i in range(len(o['input_ids'][i]))]

        for label_start, label_end, label in \
        list(zip(examples['starts'][sample_index], examples['ends'][sample_index], examples['classlist'][sample_index])):
            ### ++ 
            if label not in classes: 
                continue
            for j in range(len(labels)):
                token_start = offset_mapping[i][j][0]
                token_end = offset_mapping[i][j][1]
                if token_start == label_start: 
                    labels[j] = target_id_map[f'B-{label}']    
                if token_start > label_start and token_end <= label_end: 
                    labels[j] = target_id_map[f'I-{label}']

        for k, input_id in enumerate(o['input_ids'][i]):
            if input_id in [0,1,2]:
                labels[k] = -100

        labels = fix_beginnings(labels)
        o["labels"].append(labels)
        ###++
        o['id'].append(examples['essay_id_comp']) 

    return o

 
def tokenize_and_align_labels2(examples, label2id, return_word_ids=False):
    tokenized_inputs = tokenizer(
        examples['text'],
        truncation=True,
        max_length=CFG.max_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    all_word_ids = []
    for i, label in enumerate(examples['label']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[label[word_idx]])
            previous_word_idx = word_idx
            
        if return_word_ids:
            all_word_ids.append(word_ids)

        labels.append(label_ids)
    
    labels = fix_beginnings(labels) #++
    tokenized_inputs['label'] = labels 
    
    if return_word_ids:
        tokenized_inputs['text'] = all_word_ids
    
    return tokenized_inputs

 
# tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, \
#                                   batch_size=20000, remove_columns=datasets["train"].column_names)

 
from datasets import Dataset, load_metric

ds_train = Dataset.from_pandas(train_df)
ds_valid = Dataset.from_pandas(valid_df)

print('[HF dataset API] len ds_train:', len(ds_train))
print('[HF dataset API] len ds_valid:', len(ds_valid))
 
tokenized_ds_tr = ds_train.map(tokenize_and_align_labels, batched=True, \
                                  batch_size=20000, remove_columns=ds_train.column_names)

tokenized_ds_vl = ds_valid.map(tokenize_and_align_labels, batched=True, \
                                  batch_size=20000, remove_columns=ds_valid.column_names)

print('[HF dataset API] len tokenized_ds_tr:', len(tokenized_ds_tr))
print('[HF dataset API] len tokenized_ds_vl:', len(tokenized_ds_vl)) 
# print(tokenized_ds_tr)
# print(tokenized_ds_vl)

# accepts file path, returns tuple of (file_ID, txt, NER labels)
def generate_NER_labels_for_file(input_filename, df):
    curr_id = input_filename.split('.')[0]
    # with open(os.path.join(TRAIN_DIR, '{}.txt'.format(input_filename))) as f:
    #     curr_txt = f.read()

    # Set all token labels initially to non-label
    curr_labels = [ner_labels['O']] * len(curr_txt)
    # Iterate thru all labels associated w/ file and update labels
    curr_df = df[df['id']==curr_id]
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

# with mp.Pool(NUM_CORES) as p:
#     ner_train_rows = p.map(partial(generate_NER_labels_for_file, df=train_df), train_files)

# with mp.Pool(NUM_CORES) as p:
#     ner_valid_rows = p.map(partial(generate_NER_labels_for_file, df=valid_df), valid_files)


# tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
# Check is rust-based fast tokenizer
# assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

 
def tokenize_and_align_labels_cd(ner_raw_data):
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
    tokenized_inputs['id'] = [x[0] for x in ner_raw_data]
    return tokenized_inputs


 
# train_df[['essay_id_comp', 'text', 'classlist']].head(3)

 
tokenized_all_train = tokenize_and_align_labels_cd(
    train_df[['essay_id_comp', 'text', 'classlist']].values.tolist() # ner_train_rows
    )

tokenized_all_valid = tokenize_and_align_labels_cd(
    valid_df[['essay_id_comp', 'text', 'classlist']].values.tolist() # ner_valid_rows
    )

print('[Torch dataset] len tokenized_all_train:', len(tokenized_all_train))
print('[Torch dataset] len tokenized_all_valid:', len(tokenized_all_valid)) 
# print(tokenized_all_train)
# print(tokenized_all_valid)


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict
        self.id_col = 'essay_id_comp' #'id'
        
    def __getitem__(self, index):
        return {k:self.input_dict[k][index] for k in self.input_dict.keys() if k not in {self.id_col, 'id', 'offset_mapping', 'word_ids'}}
    
    def get_filename(self, index):
        return self.input_dict['id'][index]
    
    def get_offset(self, index):
        return self.input_dict['offset_mapping'][index]
    
    def __len__(self):
        return len(self.input_dict['input_ids'])


# ## dataset wrapper for comp metric 

# train_dataset = NERDataset(tokenized_all_train) # # try with tokenized_all_train
# valid_dataset = NERDataset(tokenized_all_valid)

#### with my ds
train_dataset = NERDataset(tokenized_ds_tr.to_dict()) 
valid_dataset = NERDataset(tokenized_ds_vl.to_dict())

print('[NERdataset] len tokenized_all_train:', len(train_dataset))
print('[NERdataset] len tokenized_all_valid:', len(valid_dataset)) 


# # Metric
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
 

 
# ## Model and Training
 
# we will use auto model for token classification
from transformers import DebertaForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer


model = AutoModelForTokenClassification.from_pretrained(CFG.model_name, num_labels=len(ner_labels)) #N_LABELS
# model = DebertaForTokenClassification.from_pretrained("microsoft/deberta-base", num_labels=N_LABELS)

 
deepspeed_config_dict = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}


### ---------------------------------------------
### Train args
### ---------------------------------------------

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)
m_name = CFG.model_name.split("/")[-1]

args = TrainingArguments(
    output_dir=OUTPUT / f"{m_name}-f{FOLD}",
    evaluation_strategy = 'steps', #"epoch", 
    eval_steps=100, #250, #500,
    logging_strategy = 'steps', #"epoch",
    save_strategy='no',  #  "epoch",
    lr_scheduler_type='linear',
    learning_rate=CFG.LR,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=1, #CFG.batch_size,
    num_train_epochs=1 if CFG.debug else CFG.epochs,
    weight_decay=CFG.WD,
    gradient_accumulation_steps=CFG.GRAD_ACC,
    warmup_ratio=CFG.WARMUP,
    dataloader_num_workers=4, #4, #8,
    ##########
    gradient_checkpointing=True,
    # optim="adamw_hf", ## OptimizerNames = "adamw_hf"
    # load_best_model_at_end=True, 
    log_level='warning',
    # report_to='wandb', 
    # metric_for_best_model=None, 
    # greater_is_better=None, 
    # deepspeed=deepspeed_config_dict, # None, 
    seed=CFG.seed, #42, 
    fp16=CFG.fp16,  
    save_total_limit=1
)


# args = TrainingArguments(f'{model_name}-{SPLIT_NUM}',
#                          PRETRAINED_MODEL,
#                          lr_scheduler_type='linear',
#                          evaluation_strategy = 'steps',
#                          eval_steps=500,
#                          dataloader_num_workers=8,
#                          learning_rate=LR,
#                          log_level='warning',
#                          fp16 = USE_FP16,
#                          per_device_train_batch_size=BATCH_SIZE,
#                          per_device_eval_batch_size=1,
#                          gradient_accumulation_steps=GRAD_ACCUM_STEPS,
#                          gradient_checkpointing=True,
#                          num_train_epochs=NUM_EPOCHS,
#                          save_strategy='no',
#                          save_total_limit=1)

 


# # # this is not the competition metric, but for now this will be better than nothing...
# # metric = load_metric("seqeval")

# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     # Remove ignored index (special tokens)
#     true_predictions = [
#         [id_target_map[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [id_target_map[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]

#     results = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }

 
label_list = ['O', 'B-Claim', 'I-Claim', 'B-Concluding Statement', 'I-Concluding Statement', 
              'B-Counterclaim', 'I-Counterclaim', 'B-Evidence', 'I-Evidence','B-Lead', 'I-Lead', 
              'B-Position', 'I-Position', 'B-Rebuttal', 'I-Rebuttal']

label2id = {label:id_ for id_, label in enumerate(label_list)}
id2label = {id_:label for id_, label in enumerate(label_list)}


# Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(pred, ground_truth):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(pred.split(' '))
    set_gt = set(ground_truth.split(' '))
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
    gt_df = gt_df[['id','discourse_type','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id','discourse_type','predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given discourse_type are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id','discourse_type'],
                           right_on=['id','discourse_type'],
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = [calc_overlap(pred, gt) for pred, gt in joined[['predictionstring_pred', 'predictionstring_gt']].values]

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
    denominator = (TP + 0.5*(FP+FN))
    if denominator == 0:
        return 0.0
    my_f1_score = TP / denominator
    return {
        "F1": round(my_f1_score, 4),
        "Precision": TP/(TP+FP),
        "Recall": TP/(TP+FN), # This was calculated incorrectly in the runs
    }
        

# id2label={i: l for l, i in label2id.items()}

# https://www.kaggle.com/zzy990106/pytorch-ner-infer?scriptVersionId=82677278&cellId=13
def get_label_predictions(dataset, preds):

    ids = dataset["id"]
    word_ids = dataset['words']
    words = dataset['text']
    
    all_preds = []

    for id_, sample_preds, sample_word_ids, words in zip(ids, preds, word_ids, words):
        label_preds = [""]*len(words)

        for pred, w_id in zip(sample_preds, sample_word_ids):
            if w_id is None:
                continue
            if label_preds[w_id] == "":
                label_preds[w_id] = id2label[pred]

        j = 0
        while j < len(label_preds):
            label = label_preds[j]

            if label.startswith("B"):
                label = label.replace("B", "I")
                end = j + 1
                while end < len(label_preds) and label_preds[end] == label:
                    end += 1

                if end - j > 7:
                    all_preds.append((id_, label.lstrip("BI-"), ' '.join(map(str, list(range(j, end))))))

                j = end
            else:
                j += 1
                
    return all_preds

### ------------------------------------
## custom trainer
### ------------------------------------


class FeedbackPrizeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if "eval_dataset" in kwargs:
            # The Trainer hides some columns that are necessary for CV
            # This code makes those columns accessible
            dataset_type = kwargs["eval_dataset"].format["type"]
            dataset_columns = list(kwargs["eval_dataset"].features.keys())
            self.cv_dataset = kwargs["eval_dataset"].with_format(type=dataset_type, columns=dataset_columns)
        super().__init__(*args, **kwargs)
        
        
    def evaluation_loop(
        self, 
        dataloader,
        description,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        
        eval_output =  super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix
        )
        
        # Custom CV F1 calculation
        # This same loop gets called during predict, and we can't do CV when predicting
        if metric_key_prefix == "eval":
            
            eval_id_preds = eval_output.predictions.argmax(-1)
            eval_label_preds = get_label_predictions(self.cv_dataset, eval_id_preds)
            
            eval_pred_df = pd.DataFrame(eval_label_preds, columns=["essay_id_comp", "discourse_type", "predictionstring"])
            
            eval_gt_df = df[df["id"].isin(self.cv_dataset["essay_id_comp"])].reset_index(drop=True).copy()
            
            classes = ['Lead', 'Position', 'Evidence', 'Claim', 'Concluding Statement', 'Counterclaim', 'Rebuttal']
            f1_scores = []
            for class_ in classes:
                gt_df = eval_gt_df.loc[eval_gt_df['discourse_type'] == class_].copy()
                pred_df = eval_pred_df.loc[eval_pred_df['discourse_type'] == class_].copy()
                eval_scores = score_feedback_comp(pred_df, gt_df)
                for score_name, score in eval_scores.items():
                    eval_output.metrics[f"{metric_key_prefix}_{class_}_CV_{score_name}"] = score
                f1_scores.append(eval_scores["F1"])
                
            eval_output.metrics[f"{metric_key_prefix}_Overall_CV_F1"] = np.mean(f1_scores)
        
        return eval_output 


### ----------------------------------------------
### Train
### ----------------------------------------------

if CFG.use_custom_trainer:
    trainer = FeedbackPrizeTrainer(
        model,
        args=args,
        train_dataset=tokenized_ds_tr,  
        eval_dataset=tokenized_ds_vl,   
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SaveBestModelCallback],
        compute_metrics=TextOverlapFBetaScore(test_df=valid_df, test_dataset=valid_dataset),
        # compute_metrics=compute_metrics,
        # optimizers=(optimizer, lr_scheduler)
    )
else:
    trainer = Trainer(
        model,
        args=args,
        train_dataset=train_dataset, #tokenized_ds_tr,  
        eval_dataset=valid_dataset, #tokenized_ds_vl,   
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SaveBestModelCallback],
        compute_metrics=TextOverlapFBetaScore(test_df=valid_df, test_dataset=valid_dataset),
        # compute_metrics=compute_metrics,
        # optimizers=(optimizer, lr_scheduler)
    )
 
trainer.train()

if CFG.wandb: wandb.finish()
trainer.save_model(OUTPUT / f"final")

 
# ## Validation
def tokenize_for_validation(examples):

    o = tokenizer(examples['text'], truncation=True, return_offsets_mapping=True, max_length=4096)

    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = o["offset_mapping"]
    
    o["labels"] = []

    for i in range(len(offset_mapping)):
                   
        labels = [l2i['O'] for i in range(len(o['input_ids'][i]))]

        for label_start, label_end, label in \
        list(zip(examples['starts'][i], examples['ends'][i], examples['classlist'][i])):
            for j in range(len(labels)):
                token_start = offset_mapping[i][j][0]
                token_end = offset_mapping[i][j][1]
                if token_start == label_start: 
                    labels[j] = l2i[f'B-{label}']    
                if token_start > label_start and token_end <= label_end: 
                    labels[j] = l2i[f'I-{label}']

        for k, input_id in enumerate(o['input_ids'][i]):
            if input_id in [0,1,2]:
                labels[k] = -100

        labels = fix_beginnings(labels)
                   
        o["labels"].append(labels)
        
    return o

 
# tokenized_val = datasets.map(tokenize_for_validation, batched=True)
# tokenized_val_test = ds_valid.map(tokenize_for_validation, batched=True)
# tokenized_val_test

## ground truth for validation

# l = []
# for example in tokenized_val_test:
#     for c, p in list(zip(example['classlist'], example['predictionstrings'])):
#         l.append({
#             'id': example['essay_comp_id'],
#             'discourse_type': c,
#             'predictionstring': p,
#         })
# gt_df = pd.DataFrame(l)
## gt_df

# # %%
# # visualization with displacy

# import pandas as pd
# import os
# from pathlib import Path
# from pylab import cm, matplotlib
# # import spacy
# # from spacy import displacy

# # %%
# path = Path('../input/train')

# colors = {
#     'Lead': '#8000ff',
#     'Position': '#2b7ff6',
#     'Evidence': '#2adddd',
#     'Claim': '#80ffb4',
#     'Concluding Statement': 'd4dd80',
#     'Counterclaim': '#ff8042',
#     'Rebuttal': '#ff0000',
#     'Other': '#007f00',
#     }

# def visualize(df, text):
#     ents = []
#     example = df['id'].loc[0]

#     for i, row in df.iterrows():
#         ents.append({
#                         'start': int(row['discourse_start']), 
#                          'end': int(row['discourse_end']), 
#                          'label': row['discourse_type']
#                     })

#     doc2 = {
#         "text": text,
#         "ents": ents,
#         "title": example
#     }

#     options = {"ents": train.discourse_type.unique().tolist() + ['Other'], "colors": colors}
#     displacy.render(doc2, style="ent", options=options, manual=True, jupyter=True)

 
# ### load best ckp

# ### fold 1
# # best_ckp_path = 'output/0227_r01_deberta-base-f1/checkpoint-880'
# best_ckp_path = 'output/0227_r01_deberta-base-f1/checkpoint-1760'
# # best_ckp_path = 'output/0227_r01_deberta-base-f1/checkpoint-2200'

# ### fold 3
# # best_ckp_path = 'output/0227_r01_deberta-base-f3/checkpoint-880'


# model = AutoModelForTokenClassification.from_pretrained(best_ckp_path, num_labels=N_LABELS, local_files_only=True)
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=tokenized_ds_tr, # tokenized_datasets["train"],
#     eval_dataset=tokenized_ds_vl,  # tokenized_datasets["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics, 
# )

 
# predictions, labels, _ = trainer.predict(tokenized_val_test)

 
# preds = np.argmax(predictions, axis=-1)
# preds.shape

 
# ## Visualize Preds 
 
# code that will convert our predictions into prediction strings, and visualize it at the same time
# this most likely requires some refactoring

def get_class(c):
    if c == 14: return 'Other'
    else: return i2l[c][2:]

def pred2span(pred, example, viz=False, test=False):
    example_id = example['id']
    n_tokens = len(example['input_ids'])
    classes = []
    all_span = []
    for i, c in enumerate(pred.tolist()):
        if i == n_tokens-1:
            break
        if i == 0:
            cur_span = example['offset_mapping'][i]
            classes.append(get_class(c))
        elif i > 0 and (c == pred[i-1] or (c-7) == pred[i-1]):
            cur_span[1] = example['offset_mapping'][i][1]
        else:
            all_span.append(cur_span)
            cur_span = example['offset_mapping'][i]
            classes.append(get_class(c))
    all_span.append(cur_span)
    
    if test: text = get_test_text(example_id)
    else: text = get_raw_text(example_id)
    
    # abra ka dabra se soli fanta ko pelo
    
    # map token ids to word (whitespace) token ids
    predstrings = []
    for span in all_span:
        span_start = span[0]
        span_end = span[1]
        before = text[:span_start]
        token_start = len(before.split())
        if len(before) == 0: token_start = 0
        elif before[-1] != ' ': token_start -= 1
        num_tkns = len(text[span_start:span_end+1].split())
        tkns = [str(x) for x in range(token_start, token_start+num_tkns)]
        predstring = ' '.join(tkns)
        predstrings.append(predstring)
                    
    rows = []
    for c, span, predstring in zip(classes, all_span, predstrings):
        e = {
            'id': example_id,
            'discourse_type': c,
            'predictionstring': predstring,
            'discourse_start': span[0],
            'discourse_end': span[1],
            'discourse': text[span[0]:span[1]+1]
        }
        rows.append(e)


    df = pd.DataFrame(rows)
    df['length'] = df['discourse'].apply(lambda t: len(t.split()))
    
    # short spans are likely to be false positives, we can choose a min number of tokens based on validation
    df = df[df.length > min_tokens].reset_index(drop=True)
    if viz: visualize(df, text)

    return df

# %%
# pred2span(preds[0], tokenized_val_test[0], viz=True)

# %%
# pred2span(preds[1], tokenized_val_test[1], viz=True)

# # %%
# dfs = []
# for i in range(len(tokenized_val_test)):
#     dfs.append(pred2span(preds[i], tokenized_val_test[i]))

# pred_df = pd.concat(dfs, axis=0)
# pred_df['class'] = pred_df['discourse_type']
# pred_df


# source: https://www.kaggle.com/robikscube/student-writing-competition-twitch#Competition-Metric-Code
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

 
# ## CV Score

# score_feedback_comp(pred_df, gt_df, return_class_scores=True)

################################################################
######### exp01 fold1 - last
# (0.6340678972661318,
#  {'Claim': 0.5682895303748384,
#   'Concluding Statement': 0.8181818181818182,
#   'Counterclaim': 0.47986289631533846,
#   'Evidence': 0.7016597510373443,
#   'Lead': 0.7997982854261221,
#   'Position': 0.6649180327868852,
#   'Rebuttal': 0.4057649667405765})

######### exp01 fold1 - 880   :: 0.6111902187743594
######### exp01 fold1 - 1760  ::
######### exp01 fold1 - 2200  :: 0.6331322663810338


######### exp01 fold3 - last
# (0.6229633353417053,
#  {'Claim': 0.5350793143830512,
#   'Concluding Statement': 0.8140105078809107,
#   'Counterclaim': 0.5075690115761353,
#   'Evidence': 0.6930450744442178,
#   'Lead': 0.7698533131006575,
#   'Position': 0.6533812479581836,
#   'Rebuttal': 0.3878048780487805})

######### exp01 fold3 - 880  :: 0.610611854940914
######### exp01 fold3 - 1320 :: 0.6195683173266759
######### exp01 fold3 - 1760 :: 0.6097080073301234
######### exp01 fold3 - 2640 ::
######### exp01 fold3 - 3520 :: 0.6229909408249436
######### exp01 fold3 - 3960 ::


# ### longformer-4096 baseline
# (0.626089187405909,
#  {'Claim': 0.560397715252803,
#   'Concluding Statement': 0.8055367994598245,
#   'Counterclaim': 0.4775330396475771,
#   'Evidence': 0.7056198011767093,
#   'Lead': 0.7827827827827828,
#   'Position': 0.6626944720291295,
#   'Rebuttal': 0.3880597014925373})


