import numpy as np
import re
import os
import torch
from torch.utils.data import Dataset
from transformers.models.auto import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


tr_collate_fn = None
val_collate_fn = None


def charpos2tokpos(charpos, offset_mapping):
    return np.where(charpos >= offset_mapping[:, 0])[0][-1]


def charspan2tokspan(charspan, offset_mapping):
    return [charpos2tokpos(charpos, offset_mapping) for charpos in charspan]


def get_token_spans(text_df, offset_mapping, text):
    spans = []
    discourse_types = []
    discourse_eff = []
    for i, row in text_df.iterrows():
        charspan = [int(row["discourse_start"]), int(row["discourse_end"])]
        token_start_pos, token_end_pos = charspan2tokspan(charspan, offset_mapping)
        discourse_types += [row["discourse_type"]]
        discourse_eff += [row["discourse_effectiveness"]]
        spans += [[token_start_pos, token_end_pos]]
    spans = np.array(spans)

    s = np.argsort(spans[:, 0])
    spans = spans[s]
    discourse_types = np.array(discourse_types)[s]
    discourse_eff = np.array(discourse_eff)[s]
    return spans, discourse_types, discourse_eff


def add_none_spans(spans, discourse_types, discourse_eff, text_len):
    none_spans = []
    none_discourse_types = []
    none_discourse_eff = []

    # preceeding None
    if spans.min() > 0:
        none_spans += [[0, spans.min() - 1]]
        none_discourse_types += ["None"]
        none_discourse_eff += ["None"]

    # intermediate None
    for i in range(1, len(spans)):
        s = spans[i - 1, 1] + 1
        e = spans[i, 0] - 1
        if e - s > -1:
            none_spans += [[s, e]]
            none_discourse_types += ["None"]
            none_discourse_eff += ["None"]

    # following None
    if spans.max() < (text_len - 1):
        none_spans += [[spans.max() + 1, text_len - 1]]
        none_discourse_types += ["None"]
        none_discourse_eff += ["None"]

    none_spans = np.array(none_spans)
    none_discourse_types = np.array(none_discourse_types)
    none_discourse_eff = np.array(none_discourse_eff)

    # concat and resort
    if len(none_spans) > 0:
        spans = np.concatenate([spans, none_spans])
        discourse_types = np.concatenate([discourse_types, none_discourse_types])
        discourse_eff = np.concatenate([discourse_eff, none_discourse_eff])

    s = np.argsort(spans[:, 0])
    spans = spans[s]
    discourse_types = discourse_types[s]
    discourse_eff = discourse_eff[s]
    return spans, discourse_types, discourse_eff


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.ids = self.df["id"].unique()
        # if mode in ['train', 'val']:
        #     self.targets_eff = self.df["discourse_effectiveness_num"].values
        # else:
        #     self.targets_eff = np.zeros(len(self.df))

        self.data_folder = cfg.data_folder
        #         if mode == "test":
        #             self.data_folder = cfg.test_data_folder

        # try:
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        # except: self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, use_fast=False)
        self.discourse_type2id = {key: val for val, key in enumerate(cfg.discourse_types)}

        cfg.effective_types = [
            'Non-Effective',
            'Effective',
            'None',
            ]
        self.effective_type2id = {key: val for val, key in enumerate(cfg.effective_types)}

    def __getitem__(self, idx):

        id_ = self.ids[idx]
        text = self.load_text(id_)

        text_df = self.df[self.df["id"] == id_].copy()

        if self.mode == "train":
            max_length = self.cfg.max_length
        else:
            max_length = self.cfg.max_length_test

        # convert annotation to wordspans and respective classes
        token_ids, wordposition, offset_mapping = self.get_token_ids(text, max_length)
        # print(len(token_ids), len(wordposition), len(offset_mapping))
        
        target = np.ones(len(token_ids)) * self.discourse_type2id["None"]
        target2 = np.zeros_like(target)
        target_eff_ner = np.ones(len(token_ids)) * self.effective_type2id["None"]
        # target_eff = self.targets_eff[idx]

        if self.mode != "test":
            # tokenspans, discourse_types = get_token_spans(text_df, offset_mapping, text)
            tokenspans, discourse_types, discourse_eff = get_token_spans(text_df, offset_mapping, text)
            assert len(tokenspans) == len(text_df)

            # print(np.unique(discourse_eff))

            tokenspans, discourse_types, discourse_eff = add_none_spans(
                tokenspans, discourse_types, discourse_eff, len(token_ids)
            )
            # print(np.unique(discourse_eff))

            for sp, dt in zip(tokenspans, discourse_types):
                target[sp[0] : sp[1] + 1] = self.discourse_type2id[dt]
                span_len = sp[1] + 1 - sp[0]

            target2[tokenspans[1:, 0]] = 1

            for sp, dt2 in zip(tokenspans, discourse_eff):
                target_eff_ner[sp[0] : sp[1] + 1] = self.effective_type2id[dt2]

        assert len(token_ids) == len(target)
        assert len(token_ids) == len(target2)
        assert len(target_eff_ner) == len(target_eff_ner)

        if (self.mode == "train") and self.cfg.dropout_tokens:
            rands = np.random.uniform(0, 1, len(token_ids))
            rand_ids = np.argwhere(rands < self.cfg.dropout_tokens)[:, 0]
            if len(rand_ids) > 0:
                # set to the unknown token (3 for Longformer)
                token_ids = np.array(token_ids)  # is there a better way?
                token_ids[rand_ids] = self.tokenizer.unk_token_id
                token_ids = token_ids.tolist()  # more pythonic?

        # add start, end tokens
        start_token = self.tokenizer.bos_token_id
        end_token = self.tokenizer.eos_token_id
        if start_token is None:
            start_token = self.tokenizer.cls_token_id
        if start_token is None:
            start_token = self.tokenizer.eos_token_id
        if end_token is None:
            end_token = self.tokenizer.sep_token_id

        input_ids = [start_token] + token_ids + [end_token]
        mdl_wordposition = [-1] + wordposition + [-1]
        attention_mask = [1] * len(input_ids)
        target = np.pad(target, (1, 1), constant_values=self.discourse_type2id["N/A"])
        target2 = np.pad(target2, (1, 1), constant_values=0)
        
        target_eff_ner = np.pad(target_eff_ner, (1, 1), constant_values=self.effective_type2id["None"])

        # pad / truncate
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        mdl_wordposition = mdl_wordposition[:max_length]
        target = target[:max_length]
        target2 = target2[:max_length]
        target_eff_ner = target_eff_ner[:max_length]

        len_pad = max_length - len(input_ids)
        if self.tokenizer.pad_token_id is None:
            input_ids = input_ids + [self.tokenizer.eos_token_id] * len_pad
        else:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * len_pad
        attention_mask = attention_mask + [0] * len_pad
        mdl_wordposition = mdl_wordposition + [-1] * len_pad
        target = np.pad(target, (0, len_pad), constant_values=self.discourse_type2id["N/A"])
        target2 = np.pad(target2, (0, len_pad), constant_values=0)
        target_eff_ner = np.pad(target_eff_ner, (0, len_pad), constant_values=self.discourse_type2id["N/A"])

        # to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        mdl_wordposition = torch.tensor(mdl_wordposition, dtype=torch.long)
        target_class = torch.tensor(target, dtype=torch.long)
        target_sep = torch.tensor(target2, dtype=torch.long)
        # target_eff = torch.tensor(target_eff, dtype=torch.float)
        target_eff_ner = torch.tensor(target_eff_ner, dtype=torch.long)

        feature_dict = {
            "input_ids": input_ids,
            "wordposition": mdl_wordposition,
            "idx": idx,
            "attention_mask": attention_mask,
            "target": target_class,
            "target2": target_sep,
            # "target_eff": target_eff,
            "target_eff_ner": target_eff_ner,
        }
        return feature_dict

    def load_text(self, id_):

        # with open(self.data_folder + f"{id_}.txt") as f:
        #     text = f.read()
        text = self.df[self.df['id']==id_].iloc[0]['essay_text'].rstrip()

        text = re.sub(" +", " ", text.replace("\xa0", " "))
        return text

    def __len__(self):
        return len(self.ids)

    def get_token_ids(self, text, max_len):

        out = self.tokenizer(
            text,
            return_tensors=None,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len - 2,  # leave 2 for special tokens
        )
        offset_mapping = np.array(out["offset_mapping"])

        wordposition = []
        for i, (discourse_start, discourse_end) in enumerate(offset_mapping):

            if len(text[discourse_start:discourse_end].split()) == 0:
                new_pos = [-1]
            #             elif len(text[discourse_start - 1 : discourse_start].split()) != 0:  # not a new word
            #                 new_pos = [wordposition[-1]]
            else:
                new_pos = [len(text[:discourse_start].split())]  # new word

            wordposition += new_pos
            # print(i, new_pos, text[discourse_start:discourse_end])

        return out["input_ids"], wordposition, offset_mapping



#### RATER Dataset stage1 - discourse_types

class CustomDataset1(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.ids = self.df["id"].unique()

        self.data_folder = cfg.data_folder
        #         if mode == "test":
        #             self.data_folder = cfg.test_data_folder

        # self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        self.tokenizer = cfg.tokenizer
        self.discourse_type2id = {key: val for val, key in enumerate(cfg.discourse_types)}

    def __getitem__(self, idx):

        id_ = self.ids[idx]
        text = self.load_text(id_)

        text_df = self.df[self.df["id"] == id_].copy()

        if self.mode == "train":
            max_length = self.cfg.max_length
        else:
            max_length = self.cfg.max_length_test

        # convert annotation to wordspans and respective classes
        token_ids, wordposition, offset_mapping = self.get_token_ids(text, max_length)

        # print(len(token_ids), len(wordposition), len(offset_mapping))
        target = np.ones(len(token_ids)) * self.discourse_type2id["None"]
        target2 = np.zeros_like(target)

        if self.mode != "test":
            tokenspans, discourse_types = get_token_spans(text_df, offset_mapping, text)
            assert len(tokenspans) == len(text_df)

            tokenspans, discourse_types = add_none_spans(
                tokenspans, discourse_types, len(token_ids)
            )
            for sp, dt in zip(tokenspans, discourse_types):
                target[sp[0] : sp[1] + 1] = self.discourse_type2id[dt]
                span_len = sp[1] + 1 - sp[0]

            target2[tokenspans[1:, 0]] = 1

        assert len(token_ids) == len(target)
        assert len(token_ids) == len(target2)

        if (self.mode == "train") and self.cfg.dropout_tokens:
            rands = np.random.uniform(0, 1, len(token_ids))
            rand_ids = np.argwhere(rands < self.cfg.dropout_tokens)[:, 0]
            if len(rand_ids) > 0:
                # set to the unknown token (3 for Longformer)
                token_ids = np.array(token_ids)  # is there a better way?
                token_ids[rand_ids] = self.tokenizer.unk_token_id
                token_ids = token_ids.tolist()  # more pythonic?

        # add start, end tokens
        start_token = self.tokenizer.bos_token_id
        end_token = self.tokenizer.eos_token_id
        if start_token is None:
            start_token = self.tokenizer.cls_token_id
        if start_token is None:
            start_token = self.tokenizer.eos_token_id
        if end_token is None:
            end_token = self.tokenizer.sep_token_id

        input_ids = [start_token] + token_ids + [end_token]
        mdl_wordposition = [-1] + wordposition + [-1]
        attention_mask = [1] * len(input_ids)
        target = np.pad(target, (1, 1), constant_values=self.discourse_type2id["N/A"])
        target2 = np.pad(target2, (1, 1), constant_values=0)

        # pad / truncate
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        mdl_wordposition = mdl_wordposition[:max_length]
        target = target[:max_length]
        target2 = target2[:max_length]

        len_pad = max_length - len(input_ids)
        if self.tokenizer.pad_token_id is None:
            input_ids = input_ids + [self.tokenizer.eos_token_id] * len_pad
        else:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * len_pad
        attention_mask = attention_mask + [0] * len_pad
        mdl_wordposition = mdl_wordposition + [-1] * len_pad
        target = np.pad(target, (0, len_pad), constant_values=self.discourse_type2id["N/A"])
        target2 = np.pad(target2, (0, len_pad), constant_values=0)

        # to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        mdl_wordposition = torch.tensor(mdl_wordposition, dtype=torch.long)
        target_class = torch.tensor(target, dtype=torch.long)
        target_sep = torch.tensor(target2, dtype=torch.long)

        feature_dict = {
            "input_ids": input_ids,
            "wordposition": mdl_wordposition,
            "idx": idx,
            "attention_mask": attention_mask,
            "target": target_class,
            "target2": target_sep,
            #####
            "input_ids_2": input_ids,
            "attention_mask_2": attention_mask,
        }
        return feature_dict

    def load_text(self, id_):

        # with open(self.data_folder + f"{id_}.txt") as f:
        #     text = f.read()
        text = self.df[self.df['id']==id_].iloc[0]['essay_text'].rstrip()

        text = re.sub(" +", " ", text.replace("\xa0", " "))
        return text

    def __len__(self):
        return len(self.ids)

    def get_token_ids(self, text, max_len):

        out = self.tokenizer(
            text,
            return_tensors=None,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len - 2,  # leave 2 for special tokens
        )
        offset_mapping = np.array(out["offset_mapping"])

        wordposition = []
        for i, (discourse_start, discourse_end) in enumerate(offset_mapping):

            if len(text[discourse_start:discourse_end].split()) == 0:
                new_pos = [-1]
            #             elif len(text[discourse_start - 1 : discourse_start].split()) != 0:  # not a new word
            #                 new_pos = [wordposition[-1]]
            else:
                new_pos = [len(text[:discourse_start].split())]  # new word

            wordposition += new_pos
            # print(i, new_pos, text[discourse_start:discourse_end])

        return out["input_ids"], wordposition, offset_mapping




