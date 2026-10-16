

import os, sys, gc
import argparse
from tqdm import tqdm 
from functools import partial
import multiprocessing as mp
import pickle
import numpy as np
import pandas as pd
import torch
import transformers
from scipy.special import softmax
from torch.utils.data import Dataset
from transformers import (AutoModelForTokenClassification, 
                          AutoTokenizer, 
                          TrainingArguments, 
                          Trainer)


ap = argparse.ArgumentParser()
ap.add_argument('--model_paths', nargs='+', required=True)
ap.add_argument("--model_name", type=str, required=True)
ap.add_argument("--save_name", type=str, required=True)
ap.add_argument("--max_len", type=int, required=True)
ap.add_argument("--save_dir", type=str, required=True)
ap.add_argument("--fold", type=int, required=True, default=-1)
ap.add_argument("--debug", type=bool, default=False)
ap.add_argument("--proba_th", type=float, default=0.54)
args = ap.parse_args()

##############################
# args = CFG
# args.fold = 1 # 0 
# # args.model_paths = ["/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f0/checkpoint-1200"]
# args.model_paths = ["/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f1/checkpoint-000"]

# args.max_len = 1536
# args.model_name = 'debertalstmV2FP1'
# args.save_name='oof_f1'
# args.save_dir = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/"
# args.proba_th = 0.54
#############################


from models.debertabilstm import DebertaForTokenClassificationwithbiLSTM
from models.longformerbilstm import LongformerForTokenClassificationwithbiLSTM

from models.hf_debertabilstmV2 import DebertaForTokenClassificationLSTM2
from models.hf_debertabilstmV2 import DebertaForTokenClassificationLSTM2_FP1
from models.hf_debertabilstmV2 import DebertaForTokenClassificationV2_CD

from models.hf_longformerbilstmV2 import LongformerForTokenClassificationLSTMV2
from models.hf_longformerbilstmV2 import LongformerForTokenClassificationFP102


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

## HParams 
NUM_CORES = 8 #16
BATCH_SIZE = 1 #4
MAX_SEQ_LENGTH = args.max_len
PRETRAINED_MODEL_PATHS = args.model_paths
OUTPUT = args.save_dir
WORD_PROB_TH = args.proba_th

AGG_FUNC = np.mean
print('==> using span token mean...')


os.makedirs(OUTPUT, exist_ok=True)

###################
if len(PRETRAINED_MODEL_PATHS) == 0:
    PRETRAINED_MODEL_PATHS = list(PRETRAINED_MODEL_PATHS)
print(PRETRAINED_MODEL_PATHS)


TEST_DIR = '/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/input/feedback1/test/' #'../input/feedback-prize-2021/test/'


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


MAX_SPAN_LEAD = 16 #33 
MAX_SPAN_CONCLUDING = 11 # 23 
MAX_SPAN_POSITION = 9 # 18
MIN_SPAN = 6 ## Final token count for discourse types

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


inverted_ner_labels = dict((v,k) for k,v in ner_labels.items())
inverted_ner_labels[-100] = 'Special Token'

ner_labels_eff = {
    'O': 0,
    'Non-Effective': 1,
    'Effective': 2,
    }

inverted_ner_labels_eff = dict((v,k) for k,v in ner_labels_eff.items())
inverted_ner_labels_eff[-100] = 'Special Token'

print('num type labels', len(ner_labels))
print('num effective labels', len(ner_labels_eff))
print(ner_labels_eff)


FOLD_NUM = args.fold if args.fold != -1 else 0


def get_data(fold=0):
    
    df = pd.read_csv("/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/train_sgkf_fold5.csv")
    df['essay_id'] = df['id'] = df['essay_id_comp']

    ## add whole essays 
    df_texts = pd.read_csv('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/train_fold10_sgkf.csv')
    df_texts = df_texts[['essay_id', 'discourse_id', 'essay_text']]
    df = df.merge(df_texts, on=['essay_id', 'discourse_id'])

    if "class" not in df.columns:
        df["class"] = df["discourse_type"]

    # print(df.shape)
    # display(df.head())

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    print('fold:', fold)
    print('train.shape:', train_df.shape,'\t','test.shape:', valid_df.shape)
    print('train ids:', train_df.id.nunique(),'\t','test ids:', valid_df.id.nunique())

    return train_df, valid_df



# df_folds = pd.read_csv('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/train_sgkf_fold5.csv')
# df_folds['essay_id'] = df_folds['essay_id_comp']
# if args.debug:
#     df_folds = df_folds.sample(10).reset_index(drop=True)
# test_files = df_folds['essay_id_comp'].unique().tolist() #os.listdir(TEST_DIR)
# print(f'[INFO] num test_files: {len(test_files)}')


# train_df = df_folds[df_folds.fold!=FOLD_NUM].reset_index(drop=True)
# valid_df = df_folds[df_folds.fold==FOLD_NUM].reset_index(drop=True)

train_df, valid_df = get_data(fold=FOLD_NUM)

train_files = train_df['essay_id'].unique()
valid_files = valid_df['essay_id'].unique()

print(f'[Fold {FOLD_NUM}] num train files', len(train_files))
print(f'[Fold {FOLD_NUM}] num valid files', len(valid_files))





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
                # iob_label_eff = ner_labels_eff['B-' + curr_discourse_label_eff]

            else:
                iob_label     = ner_labels['I-' + curr_discourse_label]
                # iob_label_eff = ner_labels_eff['I-' + curr_discourse_label_eff]

            iob_label_eff = ner_labels_eff[curr_discourse_label_eff]

            curr_labels[curr_txt_idx]     = iob_label
            curr_labels_eff[curr_txt_idx] = iob_label_eff

    assert curr_labels != [ner_labels['O']] * len(curr_txt)
    assert curr_labels_eff != [ner_labels_eff['O']] * len(curr_txt)

    return curr_id, curr_txt, curr_labels, curr_labels_eff



with mp.Pool(NUM_CORES) as p:
    ner_train_rows = p.map(partial(generate_NER_labels_for_file, df=train_df), train_files)

with mp.Pool(NUM_CORES) as p:
    ner_valid_rows = p.map(partial(generate_NER_labels_for_file, df=valid_df), valid_files)

# len(ner_train_rows), len(ner_valid_rows)

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATHS[0])

# Check is rust-based fast tokenizer
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

def tokenize_and_align_labels2(ner_raw_data):
    tokenized_inputs = tokenizer(
        [x[1] for x in ner_raw_data], 
        max_length=MAX_SEQ_LENGTH, #CFG.max_length,
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


train_dataset = NERDataset(tokenized_all_train)
valid_dataset = NERDataset(tokenized_all_valid)

print('len train dataset', len(train_dataset))
print('len valid dataset', len(valid_dataset))
data_collator = DataCollatorForTokenClassification2(tokenizer)


class TextOverlapFBetaScore:
    def __init__(self, valid_df, valid_dataset):
        self.test_df = valid_df
        self.test_dataset = valid_dataset

    def __call__(self, raw_predictions):

        predictions, _ = raw_predictions

        # predictions = predictions[0]
        # predictions_eff = predictions[1]
 
        # print(predictions[0].shape) # (3120, 1536, 15)
        # print(predictions[1].shape)


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

                            all_preds.append((
                                self.test_dataset.get_filename(curr_sample_id), 
                                cls_splitted, 
                                ' '.join(map(str, list(range(j, end+1))))
                                ))

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




# model = DebertaForTokenClassificationLSTM2.from_pretrained(CFG.model_name)

soft_predictions = None
soft_predictions_eff = None

hfargs = TrainingArguments(
    output_dir='None',
    log_level='warning',
    per_device_eval_batch_size=BATCH_SIZE)

print(f'--> loading model:', args.model_name)
for idx, curr_path in enumerate(PRETRAINED_MODEL_PATHS):
    print(f'[{idx}] loading weights:', curr_path)

if args.model_name == "longformerwithlstm":
    model = LongformerForTokenClassificationwithbiLSTM.from_pretrained(curr_path)

elif args.model_name == "debertawithlstm":
    model = DebertaForTokenClassificationwithbiLSTM.from_pretrained(curr_path)

elif args.model_name == "debertalstmV2":
    model = DebertaForTokenClassificationLSTM2.from_pretrained(curr_path)

elif args.model_name == "debertalstmV2FP1":
    model = DebertaForTokenClassificationLSTM2_FP1.from_pretrained(curr_path)

elif args.model_name == "longformerlstmV2":
    model = LongformerForTokenClassificationLSTMV2.from_pretrained(curr_path)

elif args.model_name == "longformerlstmV2FP102":
    model = LongformerForTokenClassificationFP102.from_pretrained(curr_path)

elif args.model_name == "debertaV2_CD":
    model = DebertaForTokenClassificationV2_CD.from_pretrained(curr_path)


else:
    model = AutoModelForTokenClassification.from_pretrained(curr_path, trust_remote_code=True)

### HF trainer 
trainer = Trainer(model, hfargs, tokenizer=tokenizer)

curr_preds_, _, _ = trainer.predict(valid_dataset)

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


if len(PRETRAINED_MODEL_PATHS)>1:
    soft_predictions = soft_predictions / len(PRETRAINED_MODEL_PATHS)
    soft_predictions_eff = soft_predictions_eff / len(PRETRAINED_MODEL_PATHS)

np.save(os.path.join(OUTPUT,  f'val_soft_predictions_f{FOLD_NUM}.npy'), soft_predictions)
np.save(os.path.join(OUTPUT,  f'val_soft_predictions_eff_f{FOLD_NUM}.npy'), soft_predictions_eff)


######## Post process 

soft_claim_predictions = soft_predictions[:, :, 8]

predictions = np.argmax(soft_predictions, axis=2)
soft_predictions = np.max(soft_predictions, axis=2)

print('[discourse] soft_predictions shape:', soft_predictions.shape)
print('[discourse] predictions shape:', predictions.shape)


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
    word_map = -1 * np.ones(len(offset), dtype='int32')
    w_i = 0
    for i in range(len(offset)):
        if offset[i][1]==0: continue
        while offset[i][0]>=(w[w_i+1]-("debertal_chris" in args.save_name)-("v3" in args.save_name)\
                             -("v2" in args.save_name) ): w_i += 1
        word_map[i] = int(w_i)

    return word_map


### ----------------------
### Post Processing 
### ----------------------

all_preds = []
# Clumsy gathering of predictions at word lvl - only populate with 1st subword pred
for curr_sample_id in range(len(valid_dataset)):
    curr_preds = []
    sample_preds = predictions[curr_sample_id]
    sample_offset = valid_dataset.get_offset(curr_sample_id)
    sample_txt = ner_valid_rows[curr_sample_id][1]
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
            
    # # 1.b) If we see tokens I-X, I-Y, I-X -> change I-Y to I-X
    # for j in range(1, len(word_preds_eff) - 1):
    #     pred_trio = [word_preds_eff[k] for k in [j - 1, j, j + 1]]
    #     splitted_trio = [x.split('-')[0] for x in pred_trio]
    #     if all([x == 'I' for x in splitted_trio]) and pred_trio[0] == pred_trio[2] and pred_trio[0] != pred_trio[1]:
    #         word_preds_eff[j] = word_preds_eff[j-1]


    # # 2.b) B-X, ? (not B), I-X -> change ? to I-X
    # for j in range(1, len(word_preds_eff) - 1):
    #     if 'B-' in word_preds_eff[j-1] and word_preds_eff[j+1] == f"I-{word_preds_eff[j-1].split('-')[-1]}" and word_preds_eff[j] != word_preds_eff[j+1] and 'B-' not in word_preds_eff[j]:
    #         word_preds_eff[j] = word_preds_eff[j+1]

    # # 3.b) If we see tokens I-X, O, I-X, change center token to the same for stated discourse types
    # for j in range(1, len(word_preds_eff) - 1):
    #     if word_preds_eff[j - 1] in ['I-Non-Effective', 'I-Effective'] and word_preds_eff[j-1] == word_preds_eff[j+1] and word_preds_eff[j] == 'O':
    #         word_preds_eff[j] = word_preds_eff[j-1]



    j = 0 # start of candidate discourse
    while j < len(word_preds): 
        cls = word_preds[j] 
        cls_splitted = cls.split('-')[-1]
        end = j + 1 # try to extend discourse as far as possible

        cls2 = word_preds_eff[j] 

        if word_probs[j] > WORD_PROB_TH: # 0.54: 
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
                                    valid_dataset.get_filename(curr_sample_id), 
                                    cls_splitted, 
                                    ' '.join(map(str, combined_list)),
                                    AGG_FUNC([word_probs[i] for i in combined_list if i in word_probs.keys()]),
                                    AGG_FUNC([word_probs_eff[i] for i in combined_list if i in word_probs_eff.keys()]),
                                    cls2
                                        ))
                                j = end
                                continue
                                
                        elif cls_splitted in 'Counterclaim':                        
                            if curr_preds[-1][1] == cls_splitted and adj_start - prev_end < 24:
                                del curr_preds[-1]
                                combined_list = prev_span + list(range(adj_start, adj_end))                                
                                curr_preds.append((
                                    valid_dataset.get_filename(curr_sample_id), 
                                    cls_splitted, 
                                    ' '.join(map(str, combined_list)),
                                    AGG_FUNC([word_probs[i] for i in combined_list if i in word_probs.keys()]),
                                    AGG_FUNC([word_probs_eff[i] for i in combined_list if i in word_probs_eff.keys()]),
                                    cls2
                                    ))
                                j = end
                                continue

                        elif cls_splitted in 'Evidence':                        
                            if curr_preds[-1][1] == cls_splitted and 8 < adj_start - prev_end < 25:
                                if max(claim_probs[l] for l in range(prev_end+1, adj_start)) > 0.35:
                                    claim_tokens = [str(l) for l in range(prev_end+1, adj_start) if claim_probs[l] > 0.15]
                                    if len(claim_tokens) > 2:
                                        curr_preds.append((
                                            valid_dataset.get_filename(curr_sample_id), 
                                            'Claim', 
                                            ' '.join(claim_tokens),
                                            AGG_FUNC([word_probs[int(i)] for i in claim_tokens if int(i) in word_probs.keys()]),
                                            AGG_FUNC([word_probs_eff[int(i)] for i in claim_tokens if int(i) in word_probs_eff.keys()]),
                                            cls2
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
                        valid_dataset.get_filename(curr_sample_id), 
                        cls_splitted, 
                        ' '.join(map(str, list(range(adj_start, adj_end)))),
                        AGG_FUNC([word_probs[i] for i in range(adj_start, adj_end) if i in word_probs.keys()]),
                        AGG_FUNC([word_probs_eff[i] for i in range(adj_start, adj_end) if i in word_probs_eff.keys()]),
                        cls2
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
                valid_dataset.get_filename(curr_sample_id), 
                k, 
                ' '.join(map(str, list(range(pred_start, pred_end)))),
                AGG_FUNC([word_probs[i] for i in range(pred_start, pred_end) if i in word_probs.keys()]),
                AGG_FUNC([word_probs_eff[i] for i in range(pred_start, pred_end) if i in word_probs_eff.keys()]),
                word_preds_eff[pred_start] ## TODO check again!!
                ))
            
        # ##### to avoid null rows 
        # else:
        #     curr_preds.append((test_dataset.get_filename(curr_sample_id), 
        #                        k, 
        #                        ' ', #.join(map(str, list(range(pred_start, pred_end+1)))),
        #                        AGG_FUNC([word_probs[i] for i in range(pred_start, pred_end+1) if i in word_probs.keys()])))

    all_preds.extend(curr_preds)

output_df = pd.DataFrame(all_preds)
output_df.columns = ['id', 'class', 'predictionstring', 'score', 'score_eff', 'class_eff']  

fname = os.path.join(OUTPUT,  args.save_name)
output_df.to_csv(f'{fname}.csv', index=False)

print('out df shape:', output_df.shape)
print('num ids:', output_df['id'].nunique())

#### Get validation score

from utils.rater_metric import get_f1_score 

# assert valid_df['id'].nunique() == output_df['id'].nunique()

valid_df['discourse_effectiveness_int'] = valid_df['discourse_effectiveness'].map(ner_labels_eff)
output_df['discourse_type'] = output_df['class'] 
output_df['class'] = output_df['class'].map(discourse_type_to_int) 

val_score = get_f1_score(output_df, valid_df, slient=False)
print(f'[Fold {FOLD_NUM}] Validation f1', val_score)
