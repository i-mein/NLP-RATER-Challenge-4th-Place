
import os
import argparse
from tqdm import tqdm 


ap = argparse.ArgumentParser()
ap.add_argument('--model_paths', nargs='+', required=True)
ap.add_argument("--save_name", type=str, required=True)
ap.add_argument("--max_len", type=int, required=True)
ap.add_argument("--save_dir", type=str, required=True)
args = ap.parse_args()

# if args.save_name == "yoso":
#     os.system("cp -r ../input/hf-transformers/transformers-4.16.0 .")
#     os.system("pip install -U --no-build-isolation --no-deps /kaggle/working/transformers-4.16.0")

# if args.save_name == "longformerwithlstm":
    # os.system("cp -r ../input/longformerwithbilstmhead/model.py .")
    # from model import LongformerForTokenClassificationwithbiLSTM
    
# if args.save_name == "debertawithlstm":
#     os.system("cp -r ../input/deberta-lstm/model.py .")
#     from model import DebertaForTokenClassificationwithbiLSTM

import sys 

from models.debertabilstm import DebertaForTokenClassificationwithbiLSTM
from models.longformerbilstm import LongformerForTokenClassificationwithbiLSTM

from models.hf_debertabilstmV2 import DebertaForTokenClassificationLSTM2
# from models.hf_longformerbilstmV2 import LongformerForTokenClassificationLSTM2


import gc
import pickle
import numpy as np
import pandas as pd
import torch
import transformers
import multiprocessing as mp
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import (AutoModelForTokenClassification, 
                          AutoTokenizer, 
                          TrainingArguments, 
                          Trainer)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

NUM_CORES = 8 #16
BATCH_SIZE = 1 #4
MAX_SEQ_LENGTH = args.max_len
PRETRAINED_MODEL_PATHS = args.model_paths
OUTPUT = args.save_dir

if len(PRETRAINED_MODEL_PATHS) == 0:
    PRETRAINED_MODEL_PATHS = list(PRETRAINED_MODEL_PATHS)
print(PRETRAINED_MODEL_PATHS)

if "debertal_chris" in args.save_name:
    print('==> using -1 in offset mapping...')
if ("v3" in args.save_name)|("v2" in args.save_name):
    print('==> using -1 in offset mapping...')
    
AGG_FUNC = np.mean
print('==> using span token mean...')

TEST_DIR = '/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/input/feedback1/test/' #'../input/feedback-prize-2021/test/'

### default for Fprize 2021
# MIN_TOKENS = {
#     "Lead": 32,
#     "Position": 5,
#     "Evidence": 35,
#     "Claim": 7,
#     "Concluding Statement": 6,
#     "Counterclaim": 6,
#     "Rebuttal": 6
# }

### from training data
MIN_TOKENS = {
    "Lead": 6,
    "Position": 3,
    "Evidence": 20,
    "Claim": 1,
    "Concluding Statement": 3,
    "Counterclaim": 7,
    "Rebuttal": 6
}

# MAX_SPAN_LEAD = 33 
# MAX_SPAN_CONCLUDING = 23 
# MAX_SPAN_POSITION = 18
# MIN_SPAN = 6 ## Final token count for discourse types

MAX_SPAN_LEAD = 16 #33 
MAX_SPAN_CONCLUDING = 11 # 23 
MAX_SPAN_POSITION = 9 # 18
MIN_SPAN = 6 ## Final token count for discourse types




# if "chris" not in args.save_name:
print(f"[INFO] ==> Using default BIO")
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
    
# else:
# print(f"[INFO] ==> Using cdeotte BIO")
# ner_labels = {
#     'O': 14,
#     'B-Lead': 0,
#     'I-Lead': 1,
#     'B-Position': 2,
#     'I-Position': 3,
#     'B-Evidence': 4,
#     'I-Evidence': 5,
#     'B-Claim': 6,
#     'I-Claim': 7,
#     'B-Concluding Statement': 8,
#     'I-Concluding Statement': 9,
#     'B-Counterclaim': 10,
#     'I-Counterclaim': 11,
#     'B-Rebuttal': 12,
#     'I-Rebuttal': 13
#     }

inverted_ner_labels = dict((v,k) for k,v in ner_labels.items())
inverted_ner_labels[-100] = 'Special Token'


# ner_labels_eff = ['O']
# for curr_label in df_folds['discourse_effectiveness'].unique():
#     ner_labels_eff.append('B-' + curr_label)
#     ner_labels_eff.append('I-' + curr_label)
# ner_labels_eff = dict((x,i) for i,x in enumerate(ner_labels_eff))

ner_labels_eff = {
    'O': 0,
    'B-Non-Effective': 1,
    'I-Non-Effective': 2,
    'B-Effective': 3,
    'I-Effective': 4
    }

inverted_ner_labels_eff = dict((v,k) for k,v in ner_labels_eff.items())
inverted_ner_labels_eff[-100] = 'Special Token'



df_test = pd.read_csv('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/test.csv')
test_files = df_test['essay_id_comp'].unique().tolist() #os.listdir(TEST_DIR)
print(f'[INFO] num test_files: {len(test_files)}')


# accepts file path, returns tuple of (file_ID, txt split, NER labels)
def generate_text_for_file(uuid):
    # curr_id = input_filename.split('.')[0]
    # with open(os.path.join(TEST_DIR, input_filename)) as f:
    #     curr_txt = f.read()

    curr_id = uuid
    curr_txt = df_test[df_test['essay_id_comp']==uuid]['full_text'].values[0].rstrip()

    return curr_id, curr_txt

with mp.Pool(NUM_CORES) as p:
    ner_test_rows = p.map(generate_text_for_file, test_files)
    
# if ("v3" in args.save_name)|("v2" in args.save_name):
#     from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
#     tokenizer = DebertaV2TokenizerFast.from_pretrained(PRETRAINED_MODEL_PATHS[0])
# else:

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATHS[0])
# Check is rust-based fast tokenizer
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


# tokenize and store word ids
def tokenize_with_word_ids(ner_raw_data):
    # ner_raw_data is shaped (num_examples, 3) where cols are (ID, words, word-level labels)
    tokenized_inputs = tokenizer(
        [x[1] for x in ner_raw_data], 
        max_length=MAX_SEQ_LENGTH, #CFG.max_length,
        return_offsets_mapping=True,
        truncation=True
        )
    
    tokenized_inputs['id'] = [x[0] for x in ner_raw_data]
    tokenized_inputs['offset_mapping'] = [tokenized_inputs['offset_mapping'][i] for i in range(len(ner_raw_data))]
    
    return tokenized_inputs

class NERDataset(Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict
        
    def __getitem__(self, index):
        return {k:self.input_dict[k][index] for k in self.input_dict.keys() if k not in {'id', 'offset_mapping'}}
    
    def get_filename(self, index):
        return self.input_dict['id'][index]
    
    def get_offset(self, index):
        return self.input_dict['offset_mapping'][index]
    
    def __len__(self):
        return len(self.input_dict['input_ids'])


ner_test_rows = sorted(ner_test_rows, key=lambda x: len(tokenizer(x[1], max_length=MAX_SEQ_LENGTH, truncation=True)['input_ids']))
tokenized_all = tokenize_with_word_ids(ner_test_rows)
test_dataset = NERDataset(tokenized_all)
print('len test dataset:', len(test_dataset))


soft_predictions = None
soft_predictions_eff = None

hfargs = TrainingArguments(
    output_dir='None',
    log_level='warning',
    per_device_eval_batch_size=BATCH_SIZE)

for idx, curr_path in enumerate(PRETRAINED_MODEL_PATHS):

    print(f'[{idx}] loading model:', curr_path)

    if args.save_name == "longformerwithlstm":
        model = LongformerForTokenClassificationwithbiLSTM.from_pretrained(curr_path)
    elif args.save_name == "debertawithlstm":
        model = DebertaForTokenClassificationwithbiLSTM.from_pretrained(curr_path)
    elif args.save_name == "debertalstmV2":
        model = DebertaForTokenClassificationLSTM2.from_pretrained(curr_path)
    elif args.save_name == "longformerlstmV2":
        model = LongformerForTokenClassificationLSTMV2.from_pretrained(curr_path)
    else:
        model = AutoModelForTokenClassification.from_pretrained(curr_path, trust_remote_code=True)
    
    trainer = Trainer(model, hfargs, tokenizer=tokenizer)
    
    curr_preds_, _, _ = trainer.predict(test_dataset)

    curr_preds = curr_preds_[0]
    curr_preds_eff = curr_preds_[1]

    curr_preds = curr_preds.astype(np.float32) # np.float16
    # curr_preds = softmax(curr_preds, -1)
    curr_preds = torch.softmax(torch.tensor(curr_preds), -1).numpy()
    
    curr_preds_eff = curr_preds_eff.astype(np.float32) #np.float16
    # curr_preds_eff = softmax(curr_preds_eff, -1)
    curr_preds_eff = torch.softmax(torch.tensor(curr_preds_eff), -1).numpy()


    if soft_predictions is not None:
        soft_predictions = soft_predictions + curr_preds
    else:
        soft_predictions = curr_preds

    
    if soft_predictions_eff is not None:
        soft_predictions_eff = soft_predictions_eff + curr_preds_eff
    else:
        soft_predictions_eff = curr_preds_eff


    del model, trainer, curr_preds
    del curr_preds_eff
    gc.collect()

soft_predictions = soft_predictions / len(PRETRAINED_MODEL_PATHS)

np.save(os.path.join(OUTPUT, 'soft_predictions.npy'), soft_predictions)

soft_claim_predictions = soft_predictions[:, :, 8]

predictions = np.argmax(soft_predictions, axis=2)
soft_predictions = np.max(soft_predictions, axis=2)



print('[discourse] soft_predictions shape:', soft_predictions.shape)
print('[discourse] predictions shape:', predictions.shape)


soft_predictions_eff = soft_predictions_eff / len(PRETRAINED_MODEL_PATHS)

np.save(os.path.join(OUTPUT, 'soft_predictions_eff.npy'), soft_predictions_eff)

predictions_eff = np.argmax(soft_predictions_eff, axis=2)
soft_predictions_eff = np.max(soft_predictions_eff, axis=2)

print('[effective] soft_predictions shape:', soft_predictions_eff.shape)
print('[effective] predictions shape:', predictions_eff.shape)

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
        while offset[i][0]>=(w[w_i+1]-("debertal_chris" in args.save_name)-("v3" in args.save_name)\
                             -("v2" in args.save_name) ): w_i += 1
        word_map[i] = int(w_i)

    return word_map

all_preds = []
# Clumsy gathering of predictions at word lvl - only populate with 1st subword pred
for curr_sample_id in range(len(test_dataset)):
    curr_preds = []
    sample_preds = predictions[curr_sample_id]
    sample_offset = test_dataset.get_offset(curr_sample_id)
    sample_txt = ner_test_rows[curr_sample_id][1]
    sample_word_map = generate_token_to_word_mapping(sample_txt, sample_offset)

    sample_preds_eff = predictions_eff[curr_sample_id]

    word_preds = [''] * (max(sample_word_map) + 1)
    word_probs = dict(zip(range((max(sample_word_map) + 1)),[0]*(max(sample_word_map) + 1)))
    claim_probs = dict(zip(range((max(sample_word_map) + 1)),[0]*(max(sample_word_map) + 1)))

    word_preds_eff = [''] * (max(sample_word_map) + 1)
    word_probs_eff = dict(zip(range((max(sample_word_map) + 1)),[0]*(max(sample_word_map) + 1)))

    for i, curr_word_id in enumerate(sample_word_map):
        if curr_word_id != -1:
            if word_preds[curr_word_id] == '': # only use 1st subword
                word_preds[curr_word_id] = inverted_ner_labels[sample_preds[i]]
                word_probs[curr_word_id] = soft_predictions[curr_sample_id, i]
                claim_probs[curr_word_id] = soft_claim_predictions[curr_sample_id, i]
                ### ++
                word_preds_eff[curr_word_id] = inverted_ner_labels_eff[sample_preds_eff[i]]
                word_probs_eff[curr_word_id] = soft_predictions_eff[curr_sample_id, i]

            elif 'B-' in inverted_ner_labels[sample_preds[i]]:
                word_preds[curr_word_id] = inverted_ner_labels[sample_preds[i]]
                word_probs[curr_word_id] = soft_predictions[curr_sample_id, i]
                claim_probs[curr_word_id] = soft_claim_predictions[curr_sample_id, i]
                ### ++
                word_preds_eff[curr_word_id] = inverted_ner_labels_eff[sample_preds_eff[i]]
                word_probs_eff[curr_word_id] = soft_predictions_eff[curr_sample_id, i]

            #######
                    

    # Dict to hold Lead, Position, Concluding Statement
    let_one_dict = dict() # K = Type, V = (Prob of start token, start, end)

    # 1) If we see tokens I-X, I-Y, I-X -> change I-Y to I-X
    for j in range(1, len(word_preds) - 1):
        pred_trio = [word_preds[k] for k in [j - 1, j, j + 1]]
        splitted_trio = [x.split('-')[0] for x in pred_trio]
        if all([x == 'I' for x in splitted_trio]) and pred_trio[0] == pred_trio[2] and pred_trio[0] != pred_trio[1]:
            word_preds[j] = word_preds[j-1]
            # word_probs_eff[j] = word_probs_eff[j-1]

    # 2) B-X, ? (not B), I-X -> change ? to I-X
    for j in range(1, len(word_preds) - 1):
        if 'B-' in word_preds[j-1] and word_preds[j+1] == f"I-{word_preds[j-1].split('-')[-1]}" and word_preds[j] != word_preds[j+1] and 'B-' not in word_preds[j]:
            word_preds[j] = word_preds[j+1]
            # word_probs_eff[j] = word_probs_eff[j+1]

    # 3) If we see tokens I-X, O, I-X, change center token to the same for stated discourse types
    for j in range(1, len(word_preds) - 1):
        if word_preds[j - 1] in ['I-Lead', 'I-Position', 'I-Concluding Statement'] and word_preds[j-1] == word_preds[j+1] and word_preds[j] == 'O':
            word_preds[j] = word_preds[j-1]
            # word_probs_eff[j] = word_probs_eff[j-1]

    ### apply same rules (1-3) for effectiveness
    
    # 1.b) If we see tokens I-X, I-Y, I-X -> change I-Y to I-X
    for j in range(1, len(word_preds_eff) - 1):
        pred_trio = [word_preds_eff[k] for k in [j - 1, j, j + 1]]
        splitted_trio = [x.split('-')[0] for x in pred_trio]
        if all([x == 'I' for x in splitted_trio]) and pred_trio[0] == pred_trio[2] and pred_trio[0] != pred_trio[1]:
            word_preds_eff[j] = word_preds_eff[j-1]


    # 2.b) B-X, ? (not B), I-X -> change ? to I-X
    for j in range(1, len(word_preds_eff) - 1):
        if 'B-' in word_preds_eff[j-1] and word_preds_eff[j+1] == f"I-{word_preds_eff[j-1].split('-')[-1]}" and word_preds_eff[j] != word_preds_eff[j+1] and 'B-' not in word_preds_eff[j]:
            word_preds_eff[j] = word_preds_eff[j+1]

    # 3.b) If we see tokens I-X, O, I-X, change center token to the same for stated discourse types
    for j in range(1, len(word_preds_eff) - 1):
        if word_preds_eff[j - 1] in ['I-Non-Effective', 'I-Effective'] and word_preds_eff[j-1] == word_preds_eff[j+1] and word_preds_eff[j] == 'O':
            word_preds_eff[j] = word_preds_eff[j-1]



    j = 0 # start of candidate discourse
    while j < len(word_preds): 
        cls = word_preds[j] 
        cls_splitted = cls.split('-')[-1]
        end = j + 1 # try to extend discourse as far as possible

        if word_probs[j] > 0.54: 
            # Must match suffix i.e., I- to I- only; no B- to I-
            while end < len(word_preds) and (word_preds[end].split('-')[-1] == cls_splitted if cls_splitted in ['Lead', 'Position', 'Concluding Statement'] else word_preds[end] == f'I-{cls_splitted}'):
                end += 1
            # if we're here, end is not the same pred as start
            if cls != 'O' and (end - j > MIN_TOKENS[cls_splitted] or max(word_probs[l] for l in range(j, end)) > 0.73): # needs to be longer than class-specified min
                if cls_splitted in ['Lead', 'Position', 'Concluding Statement']:
                    lpc_max_prob = max(word_probs[c] for c in range(j, end))
                    if cls_splitted in let_one_dict: # Already existing, check contiguous or higher prob
                        prev_prob, prev_start, prev_end = let_one_dict[cls_splitted]
                        if cls_splitted in ['Lead', 'Concluding Statement'] and j - prev_end < 49: # If close enough, combine
                            let_one_dict[cls_splitted] = (max(prev_prob, lpc_max_prob), prev_start, end)
                            
                            # Delete other preds that lie inside the joined LC discourse
                            for l in range(len(curr_preds) - 1, 0, -1):
                                check_span = curr_preds[l][2]
                                check_start, check_end = int(check_span[0]), int(check_span[-1])
                                if check_start > prev_start and check_end < end:
                                    del curr_preds[l]
                            
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


                    #############################################################
                    # Run a bunch of adjustments to discourse predictions based on CV 
                    adj_start, adj_end = j, end + 1

                    # Run some heuristics against previous discourse
                    if len(curr_preds) > 0:
                        prev_span = list(map(int, curr_preds[-1][2].split()))
                        prev_start, prev_end = prev_span[0], prev_span[-1]

                        # Join adjacent rebuttals
                        if cls_splitted in 'Rebuttal':                        
                            if curr_preds[-1][1] == cls_splitted and adj_start - prev_end < 32:
                                del curr_preds[-1]
                                combined_list = prev_span + list(range(adj_start, adj_end))                                
                                curr_preds.append((
                                    test_dataset.get_filename(curr_sample_id), 
                                    cls_splitted, 
                                    ' '.join(map(str, combined_list)),
                                    AGG_FUNC([word_probs[i] for i in combined_list if i in word_probs.keys()]),
                                    AGG_FUNC([word_probs_eff[i] for i in combined_list if i in word_probs_eff.keys()])
                                        ))
                                j = end
                                continue
                                
                        elif cls_splitted in 'Counterclaim':                        
                            if curr_preds[-1][1] == cls_splitted and adj_start - prev_end < 24:
                                del curr_preds[-1]
                                combined_list = prev_span + list(range(adj_start, adj_end))                                
                                curr_preds.append((
                                    test_dataset.get_filename(curr_sample_id), 
                                    cls_splitted, 
                                    ' '.join(map(str, combined_list)),
                                    AGG_FUNC([word_probs[i] for i in combined_list if i in word_probs.keys()]),
                                    AGG_FUNC([word_probs_eff[i] for i in combined_list if i in word_probs_eff.keys()])
                                    ))
                                j = end
                                continue

                        elif cls_splitted in 'Evidence':                        
                            if curr_preds[-1][1] == cls_splitted and 8 < adj_start - prev_end < 25:
                                if max(claim_probs[l] for l in range(prev_end+1, adj_start)) > 0.35:
                                    claim_tokens = [str(l) for l in range(prev_end+1, adj_start) if claim_probs[l] > 0.15]
                                    if len(claim_tokens) > 2:
                                        curr_preds.append((
                                            test_dataset.get_filename(curr_sample_id), 
                                            'Claim', 
                                            ' '.join(claim_tokens),
                                            AGG_FUNC([word_probs[int(i)] for i in claim_tokens if int(i) in word_probs.keys()]),
                                            AGG_FUNC([word_probs_eff[int(i)] for i in claim_tokens if int(i) in word_probs_eff.keys()])
                                            ))
                        # If gap with discourse of same type, extend to it 
                        elif curr_preds[-1][1] == cls_splitted and adj_start - prev_end > 2:
                            adj_start -= 1

                    # Adjust discourse lengths if too long or short
                    if cls_splitted == 'Evidence':
                        if adj_end - adj_start < 45:
                            adj_start -= 9
                        else:
                            adj_end -= 1
                    elif cls_splitted == 'Claim':
                        if adj_end - adj_start > 24:
                            adj_end -= 1
                    elif cls_splitted == 'Counterclaim':
                        if adj_end - adj_start > 24:
                            adj_end -= 1
                        else:
                            adj_start -= 1
                            adj_end += 1
                    elif cls_splitted == 'Rebuttal':
                        if adj_end - adj_start > 32:
                            adj_end -= 1
                        else:
                            adj_start -= 1
                            adj_end += 1
                    adj_start = max(0, adj_start)
                    adj_end = min(len(word_preds) - 1, adj_end)
                    curr_preds.append((
                        test_dataset.get_filename(curr_sample_id), 
                        cls_splitted, 
                        ' '.join(map(str, list(range(adj_start, adj_end)))),
                        AGG_FUNC([word_probs[i] for i in range(adj_start, adj_end) if i in word_probs.keys()]),
                        AGG_FUNC([word_probs_eff[i] for i in range(adj_start, adj_end) if i in word_probs_eff.keys()])
                        ))

        j = end 

    # Add the Lead, Position, Concluding Statement
    for k, v in let_one_dict.items():
        pred_start = v[1]
        pred_end = v[2]

        # Lookback and add preceding I- tokens
        while pred_start - 1 > 0 and word_preds[pred_start-1] == f'I-{k}':
            pred_start = pred_start - 1
        # Try to add the matching B- tag if immediately precedes the current I- sequence
        if pred_start - 1 > 0 and word_preds[pred_start - 1] == f'B-{k}':
            pred_start = pred_start - 1

        ## NOTE check this again!
        # Extend short Leads and Concluding Statements
            
        if k == 'Lead':
            if pred_end - pred_start < MAX_SPAN_LEAD: #33:
                pred_end = min(len(word_preds), pred_end + 5)
            else:
                pred_end -= 5
        elif k == 'Concluding Statement':
            if pred_end - pred_start < MAX_SPAN_CONCLUDING: #23:
                pred_start = max(0, pred_start - 1)
                pred_end = min(len(word_preds), pred_end + 10)
        elif k == 'Position':
            if pred_end - pred_start < MAX_SPAN_POSITION: #18:
                pred_end = min(len(word_preds), pred_end + 3)

        pred_start = max(0, pred_start)
        if pred_end - pred_start > MIN_SPAN: #6: ## NOTE check this again!
            curr_preds.append((
                test_dataset.get_filename(curr_sample_id), 
                k, 
                ' '.join(map(str, list(range(pred_start, pred_end)))),
                AGG_FUNC([word_probs[i] for i in range(pred_start, pred_end) if i in word_probs.keys()]),
                AGG_FUNC([word_probs_eff[i] for i in range(pred_start, pred_end) if i in word_probs_eff.keys()])
                ))
            
        # ##### to avoid null rows 
        # else:
        #     curr_preds.append((test_dataset.get_filename(curr_sample_id), 
        #                        k, 
        #                        ' ', #.join(map(str, list(range(pred_start, pred_end+1)))),
        #                        AGG_FUNC([word_probs[i] for i in range(pred_start, pred_end+1) if i in word_probs.keys()])))

    all_preds.extend(curr_preds)

output_df = pd.DataFrame(all_preds)
# output_df.columns = ['id', 'class', 'predictionstring', 'scores']
output_df.columns = ['id', 'class', 'predictionstring', 'scores', 'scores_eff']

fname = os.path.join(OUTPUT, args.save_name)
output_df.to_csv(f'{fname}_preds.csv', index=False)
# output_df.to_csv(f'{OUTPUT.as_posix()}/{args.save_name}.csv', index=False)

print('out df shape:', output_df.shape)
print('num ids:', output_df['id'].nunique())



