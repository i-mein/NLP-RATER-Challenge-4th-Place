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
    for i, row in text_df.iterrows():
        charspan = [int(row["discourse_start"]), int(row["discourse_end"])]
        token_start_pos, token_end_pos = charspan2tokspan(charspan, offset_mapping)
        discourse_types += [row["discourse_type"]]
        spans += [[token_start_pos, token_end_pos]]
    spans = np.array(spans)

    s = np.argsort(spans[:, 0])
    spans = spans[s]
    discourse_types = np.array(discourse_types)[s]
    return spans, discourse_types


def add_none_spans(spans, discourse_types, text_len):
    none_spans = []
    none_discourse_types = []

    # preceeding None
    if spans.min() > 0:
        none_spans += [[0, spans.min() - 1]]
        none_discourse_types += ["None"]

    # intermediate None
    for i in range(1, len(spans)):
        s = spans[i - 1, 1] + 1
        e = spans[i, 0] - 1
        if e - s > -1:
            none_spans += [[s, e]]
            none_discourse_types += ["None"]

    # following None
    if spans.max() < (text_len - 1):
        none_spans += [[spans.max() + 1, text_len - 1]]
        none_discourse_types += ["None"]

    none_spans = np.array(none_spans)
    none_discourse_types = np.array(none_discourse_types)

    # concat and resort
    if len(none_spans) > 0:
        spans = np.concatenate([spans, none_spans])
        discourse_types = np.concatenate([discourse_types, none_discourse_types])
    s = np.argsort(spans[:, 0])
    spans = spans[s]
    discourse_types = discourse_types[s]
    return spans, discourse_types


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.ids = self.df["id"].unique().tolist()
        # print(len(self.ids), self.ids[:5])

        self.data_folder = cfg.data_folder
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        self.max_length = cfg.max_length
        self.discourse_type2id = {key: val for val, key in enumerate(cfg.discourse_types)}

    def __getitem__(self, idx):


        id_ = self.ids[idx]
        # print('id', id_)
        text = self.load_text(id_)

        text_df = self.df[self.df["id"] == id_].copy()

        # convert annotation to wordspans and respective classes
        token_ids, wordposition, offset_mapping = self.get_token_ids(text)

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

            target2[tokenspans[1:, 0]] = 1

        assert len(token_ids) == len(target)
        assert len(token_ids) == len(target2)

        # add start, end tokens
        input_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
        mdl_wordposition = [-1] + wordposition + [-1]
        attention_mask = [1] * len(input_ids)
        target = np.pad(target, (1, 1), constant_values=self.discourse_type2id["N/A"])
        target2 = np.pad(target2, (1, 1), constant_values=0)

        # pad / truncate
        input_ids = input_ids[: self.max_length]
        attention_mask = attention_mask[: self.max_length]
        mdl_wordposition = mdl_wordposition[: self.max_length]
        target = target[: self.max_length]
        target2 = target2[: self.max_length]

        len_pad = self.max_length - len(input_ids)
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
        }
        return feature_dict

    def load_text(self, id_):
        # with open(self.data_folder + f"{id_}.txt") as f:
        #     text = f.read()

        text = self.df[self.df['id'] == id_].iloc[0]["essay_text"].rstrip()    
        text = re.sub(" +", " ", text.replace("\xa0", " "))
        return text

    def __len__(self):
        return len(self.ids)

    def get_token_ids(self, text):

        out = self.tokenizer(
            text,
            return_tensors=None,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,  # leave 2 for special tokens
        )
        offset_mapping = np.array(out["offset_mapping"])

        wordposition = []
        for i, (discourse_start, discourse_end) in enumerate(offset_mapping):

            if len(text[discourse_start:discourse_end].split()) == 0:
                new_pos = [-1]
            elif len(text[discourse_start - 1 : discourse_start].split()) != 0:  # not a new word
                new_pos = [wordposition[-1]]
            else:
                new_pos = [len(text[:discourse_start].split())]  # new word

            wordposition += new_pos
            # print(i, new_pos, text[discourse_start:discourse_end])

        return out["input_ids"], wordposition, offset_mapping
