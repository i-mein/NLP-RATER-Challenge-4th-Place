# %%
# ## Setup
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
import importlib
from glob import glob

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

# import pytorch_lightning as pl
# from pytorch_lightning import callbacks
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
import warnings 
warnings.filterwarnings('ignore')

print('torch.__version__:', torch.__version__)
print('transformers.__version__:', transformers.__version__)
# print('PLightning.__version__:', pl.__version__)


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class CFG:
    exp_name = 'exp18'
    task = "ner"
    n_folds = 5 # 10
    trn_folds = [0] # [0,1,2,3,4]

    # model_name = "microsoft/deberta-large"
    model_name = "/workspace/hf_pretrained/deberta-large"

    # max_length = 512 ##1536 #1024
    max_length = 1536 #768  
    stride = 128
    min_tokens = 6
    # TRAINING HYPERPARAMS
    batch_size = 1 #32 #4       # 16
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
    output = "/workspace/results/"
    # ######
    NUM_CORES = 8 #cfg["NUM_CORES"]
    TEST_CSV = "/workspace/data/test.csv"

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
from utils.wbf import *

DIR = Path("/workspace/data/")
OUTPUT = "/workspace/results/exp18"

# OUTPUT = Path(CFG.output + CFG.exp_name)
# OUTPUT.mkdir(exist_ok=True, parents=True)
# os.makedirs(OUTPUT, exist_ok=True)

# seed everything
CFG.device = device
set_seed(CFG.seed)

# %% [markdown]
# # Test data 



# %%
## check test data 

# df_test = pd.read_csv(CFG.TEST_CSV)
df_test = pd.read_csv(DIR / 'test.csv')
test_ids = df_test['essay_id_comp'].unique().tolist()

print('test shape:', df_test.shape)
print('num essays:', df_test.essay_id_comp.nunique())


# df_test['n_chars'] = df_test['full_text'].apply(len)
# print(df_test['n_chars'].min(), df_test['n_chars'].max())
# df_test['n_chars'].hist(bins=50);
# df_test.head()
# # plt.show()

# %%
def extract_discourse_text(row):
    cur_txt = df_test[df_test.essay_id_comp==row['id']]['full_text'].values[0]
    cur_txt_words = cur_txt.split()
    ps = row['predictionstring']
    ps = ps.split()
    start, end = int(ps[0]), int(ps[-1])
    return " ".join(cur_txt_words[start:end])



# %%
### exp18.a 5 classes NEW inference 

# !python generate_test_preds_hf_V1.py --model_paths /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f0/checkpoint-2800 \
#     --model_name debertalstmV2 \
#     --save_name test_v2 \
#     --max_len 1536 \
#     --save_dir $OUTPUT
#     --debug True

# %% [markdown]
# # Infer exp18 (3 classes)

# %%
### SUBMIT OK (fold 1) -- LB 0.5738 

# %cd /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/src
# OUTPUT = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f1"

# !python generate_test_preds_hf_V2.py --model_paths /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f1/deberta-large-f1/checkpoint-000 \
#     --model_name debertalstmV2FP1 \
#     --save_name exp18b_f1 \
#     --max_len 1536 \
#     --save_dir $OUTPUT \
#     ##--debug True  


# %%

# %cd /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/src
# !CUDA_VISIBLE_DEVICES=0 python generate_test_preds_hf_V2.py --model_paths /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f2/deberta-large-f2/checkpoint-1200 \
#     --model_name debertalstmV2FP1 \
#     --save_name $SAVE_NAME \
#     --max_len 1536 \
#     --save_dir $OUTPUT \
#     ##--debug True  

# os.system("CUDA_VISIBLE_DEVICES=0 python generate_test_preds_hf.py --model_paths /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f2/deberta-large-f2/checkpoint-1200 \
#      --model_name debertalstmV2FP1 \
#      --save_name $SAVE_NAME \
#      --max_len 1536 \
#      --save_dir $OUTPUT ")  

# OUTPUT = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f3"
# SAVE_NAME = "exp18_f3_ckp1200"

# !CUDA_VISIBLE_DEVICES=1 python generate_test_preds_hf_V2.py --model_paths /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f3/deberta-large-f3/checkpoint-1200 \
#     --model_name debertalstmV2FP1 \
#     --save_name $SAVE_NAME \
#     --save_dir $OUTPUT \
#     --max_len 1536



# %cd /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/src
# OUTPUT = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f3"
# SAVE_NAME = "exp18_f3_3ckp"


# !CUDA_VISIBLE_DEVICES=1 python generate_test_preds_hf_V2.py --model_paths /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f3/deberta-large-f3/checkpoint-1200 /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f3/checkpoint1-200  /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f3/checkpoint2-400 \
#     --model_name debertalstmV2FP1 \
#     --save_name $SAVE_NAME \
#     --save_dir $OUTPUT \
#     --max_len 1536


# # ###  SUBMIT (fold1 + fold 3) -- LB 0.5805

# %cd /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/src

# OUTPUT = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f3"
# SAVE_NAME = "exp18_f1_f3"


# !CUDA_VISIBLE_DEVICES=1 python generate_test_preds_hf_V2.py --model_paths /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f1/deberta-large-f1/checkpoint-000 /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18_f3/deberta-large-f3/checkpoint-1200 \
#     --model_name debertalstmV2FP1 \
#     --save_name $SAVE_NAME \
#     --save_dir $OUTPUT \
#     --max_len 1536


##### CONFIG 

ap = argparse.ArgumentParser()
# ap.add_argument('--model_paths', nargs='+', required=True)
# ap.add_argument("--model_name", type=str, required=True)
# ap.add_argument("--save_name", type=str, required=True)
ap.add_argument("--fold", type=int, required=True, default=-1)
ap.add_argument("--save_dir", type=str, required=True)
ap.add_argument("--test_th", type=float, default=0.54, required=False)
ap.add_argument("--debug", type=bool, default=False, required=False)
cargs = ap.parse_args()



FOLD = cargs.fold
TEST_TH = cargs.test_th
OUTPUT = cargs.save_dir
# MODEL_CKP = cargs.model_paths
# print('num checkpoints:', len(MODEL_CKP))
print('FOLD', FOLD)


os.makedirs(OUTPUT, exist_ok=True)


# %%
###  SUBMIT 5folds -- LB 0.57x

# %cd /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/src
# os.chdir('scripts_local')
os.chdir('scripts')

# !CUDA_VISIBLE_DEVICES=1 python generate_test_preds_hf_V2.py --model_paths \
#     /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f0/checkpoint-1200 \
#     /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f1/checkpoint-000 \
#     /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f2/checkpoint-1200 \
#     /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f3/checkpoint-1200 \
#     /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f4/checkpoint-1600 \
#     --model_name debertalstmV2FP1 \
#     --save_name $SAVE_NAME \
#     --save_dir $OUTPUT \
#     --max_len 1536

if FOLD==0:
    SAVE_NAME = 'exp18_f0'
    print(SAVE_NAME)
    os.system(
        f"CUDA_VISIBLE_DEVICES=0 python generate_test_preds_hf.py \
          --model_paths /workspace/checkpoints/exp18/deberta-large-f0/checkpoint-1200 \
     --model_name debertalstmV2FP1 \
     --save_name exp18_f0 \
     --max_len 1536 \
     --save_dir {OUTPUT} --proba_th {TEST_TH}" 
     )  

elif FOLD==1:
    SAVE_NAME='exp18_f1'
    os.system(
        f"CUDA_VISIBLE_DEVICES=0 python generate_test_preds_hf.py \
          --model_paths /workspace/checkpoints/exp18/deberta-large-f1/checkpoint-000 \
     --model_name debertalstmV2FP1 \
     --save_name exp18_f1 \
     --max_len 1536 \
     --save_dir {OUTPUT} --proba_th {TEST_TH}"
     )  
elif FOLD==2:
    SAVE_NAME='exp18_f2'
    os.system(
        f"""CUDA_VISIBLE_DEVICES=0 python generate_test_preds_hf.py \
        --model_paths /workspace/checkpoints/exp18/deberta-large-f2/checkpoint-1200 \ 
        --model_name debertalstmV2FP1 --save_name exp18_f2 --max_len 1536 \
        --save_dir {OUTPUT} --proba_th {TEST_TH} """
        )  

elif FOLD==3:
    SAVE_NAME='exp18_f3'
    os.system(
        f"""CUDA_VISIBLE_DEVICES=0 python generate_test_preds_hf.py \
        --model_paths /workspace/checkpoints/exp18/deberta-large-f3/checkpoint-1200 \
        --model_name debertalstmV2FP1 --save_name exp18_f3 --max_len 1536 \
        --save_dir {OUTPUT} --proba_th {TEST_TH} """
        )  

elif FOLD==4:
    SAVE_NAME='exp18_f4'
    os.system(
        f"""CUDA_VISIBLE_DEVICES=0 python generate_test_preds_hf.py \
            --model_paths /workspace/checkpoints/exp18/deberta-large-f4/checkpoint-1600 \
        --model_name debertalstmV2FP1 --save_name exp18_f4 --max_len 1536 \
        --save_dir {OUTPUT} --proba_th {TEST_TH} """
        )  
        
elif FOLD==-1:
    SAVE_NAME='exp18_5f'
    os.system(
        f"""CUDA_VISIBLE_DEVICES=0 python generate_test_preds_hf.py \
            --model_paths /workspace/checkpoints/exp18/deberta-large-f0/checkpoint-1200 \
                /workspace/checkpoints/exp18/deberta-large-f1/checkpoint-000 \
                /workspace/checkpoints/exp18/deberta-large-f2/checkpoint-1200 \
                /workspace/checkpoints/exp18/deberta-large-f3/checkpoint-1200 \
                /workspace/checkpoints/exp18/deberta-large-f4/checkpoint-1600 \
        --model_name debertalstmV2FP1 --save_name exp18_5f --max_len 1536 \
        --save_dir {OUTPUT} --proba_th {TEST_TH} """
        )  
else:
    print('No valid command')



# %%
# 0 - Lead
# 1 - Position
# 2 - Claim
# 3 - Evidence
# 4 - Counterclaim
# 5 - Rebuttal
# 6 - Concluding Statement


rater_class_map = {
    'Lead': 0,
    'Position': 1,
    'Claim': 2, 
    'Evidence': 3, 
    'Counterclaim': 4,
    'Rebuttal': 5, 
    'Concluding Statement': 6, 
}

inv_rater_class_map = {v: k for k, v in rater_class_map.items()}


# # Threshold found from CV
# label_to_threshold = {
#     0 : 0.275, #Claim
#     1 : 0.375, #Evidence
#     2 : 0.325, #Lead
#     3 : 0.325, #Position
#     4 : 0.4, #Concluding Statement
#     5 : 0.275, #Counterclaim
#     6 : 0.275 #Rebuttal
# }

label_to_threshold = {
    0 : 0.325, #Lead          #0.275, #Claim
    1 : 0.325, #Position      #0.375, #Evidence
    2 : 0.275, #Claim         #0.325, #Lead
    3 : 0.375, #Evidence      #0.325, #Position
    4 : 0.275, #Counterclaim  #0.4, #Concluding Statement
    5 : 0.275, #Rebuttal      #0.275, #Counterclaim
    6 : 0.4,   #Concluding Statement #0.275 #Rebuttal
}

# %%
def generate_submission(fname, save_name, out_dir=OUTPUT, verbose=False):

    pred_df = pd.read_csv(f'{out_dir}/{fname}.csv') 
    print(pred_df.shape)
    print(pred_df.id.nunique())
    # display(pred_df)

    ## convert effective scores to RATER format 
    pred_df['score_discourse_effectiveness_0'] = -100
    pred_df['score_discourse_effectiveness_1'] = -100

    for i, row in pred_df.iterrows():
        if row['class_eff'] == 'Effective':
            pred_df.loc[i, 'score_discourse_effectiveness_1'] = row['scores_eff']
            pred_df.loc[i, 'score_discourse_effectiveness_0'] = 1 - row['scores_eff']
        else:
            pred_df.loc[i, 'score_discourse_effectiveness_0'] = row['scores_eff']
            pred_df.loc[i, 'score_discourse_effectiveness_1'] = 1 - row['scores_eff']


    sample_sub = pd.read_csv(DIR / 'sample_submission.csv')

    sub = pred_df.copy()
    sub = sub.rename(columns={'id': 'essay_id_comp'})
    sub["discourse_type"] = sub["class"].map(rater_class_map)

    sub = sub[sample_sub.columns]

    sub.to_csv(f'{out_dir}{save_name}.csv', index=False)

    assert sub.essay_id_comp.nunique() == 10402, "ERROR: check num ids.."

    # display(sub.head())
    print(sub.essay_id_comp.nunique() == 10402)

    return sub


sub = generate_submission(f"{SAVE_NAME}", f"sub_{SAVE_NAME}", verbose=True)




