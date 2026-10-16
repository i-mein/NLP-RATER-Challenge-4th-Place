import os
import pickle
import numpy as np
import pandas as pd
import scipy
import random
import math
import time
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nets import *


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=cfg.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values
        if cfg.aux_target: 
            self.aux_labels = df[cfg.aux_target_cols].values


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)

        if self.cfg.aux_target: 
            aux_label = torch.tensor(self.aux_labels[item], dtype=torch.float)
            return inputs, label, aux_label
        else:
            return inputs, label
    

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs





### -------------------------
## Datasets
### -------------------------

class FeedbackPrize3Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        texts, 
        tokenizer, 
        max_length=512, 
        texts_pair=None, 
        labels=None,
        token_dropout_probability=0.0,
    ):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts_pair = texts_pair
        self.labels = labels
        self.token_dropout_probability = token_dropout_probability
        
    def __len__(self):
        return len(self.texts)
    

    def __getitem__(self, index):
        text = self.texts[index]
        text_pair = None
        
        if self.texts_pair is not None:
            text_pair = self.texts_pair[index]
        
        encoded = self.tokenizer(
            text=text, 
            text_pair=text_pair,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=True,
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
            return_offsets_mapping=False,
            return_tensors=None,
        )
        
        if self.token_dropout_probability > 0.0:
            encoded["input_ids"] = apply_token_dropout(
                inputs=encoded["input_ids"], 
                special_tokens_mask=encoded["special_tokens_mask"], 
                p=self.token_dropout_probability,
                tokenizer=self.tokenizer, 
            ).squeeze().tolist()
        
        if self.labels is not None:
            labels = list(self.labels[index])
            encoded["labels"] = labels
        
        return encoded




class Collator:
    def __init__(
        self, 
        tokenizer, 
        max_length=512, 
        padding="max_length", 
        pad_to_multiple_of=None, 
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, batch):
        lengths = [len(sample["input_ids"]) for sample in batch]
        max_length = max(lengths)

        if self.max_length is not None:
            max_length = min(max_length, self.max_length)

        batch = self.tokenizer.pad(
            encoded_inputs=batch,                       
            max_length=max_length, 
            padding=self.padding, 
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return {
            key: torch.tensor(value) if not isinstance(value[0], str) else value
            for key, value in batch.items()
        }




# ====================================================
# Loss
# ====================================================

class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


