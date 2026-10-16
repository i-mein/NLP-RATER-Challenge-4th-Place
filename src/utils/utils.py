import copy
import os
import pickle
import numpy as np
import pandas as pd
import scipy
import random
import math
import time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
# from tez import enums
# from tez.callbacks import Callback
from tqdm import tqdm
import json


# ====================================================
# Utils
# ====================================================

# OUTPUT_DIR = './'
def get_logger(dir='./'):
    filename = dir+'/train'
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger



class Log:
    def __init__(self,log_path,time_key=True):
        self.path = log_path
        if time_key:
            self.path = self.path.replace('.','{}.'.format(time.strftime('_%Y%m%d%H%M%S',time.localtime(time.time()))))
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),file=open(self.path,'a+'))
        print('log path:', self.path)
        print('****************开始记录*********************',file=open(self.path,'a+'))
    
    def __call__(self,*content):
        t1 = time.strftime('%H:%M:%S',time.localtime(time.time()))
        print(*content)
        print(t1,content,file=open(self.path,'a+'))
    
    def clean(self):
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),file=open(self.path,'w'))
        print('****************开始记录*********************',file=open(self.path,'a+'))

 



def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


# Turns a dictionary into a class
class dict2class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])



def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_everything(seed: int=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def load_checkpoint(cfg, model):
    d = torch.load(cfg.architecture.pretrained_weights, map_location="cpu")

    if "model" in d:
        model_weights = d["model"]
    else:
        model_weights = d

    if (
        model.backbone.embeddings.word_embeddings.weight.shape[0]
        < model_weights["backbone.embeddings.word_embeddings.weight"].shape[0]
    ):
        print("resizing pretrained embedding weights")
        model_weights["backbone.embeddings.word_embeddings.weight"] = model_weights[
            "backbone.embeddings.word_embeddings.weight"
        ][: model.backbone.embeddings.word_embeddings.weight.shape[0]]

    try:
        model.load_state_dict(model_weights, strict=True)
    except Exception as e:
        print("removing unused pretrained layers")
        for layer_name in re.findall("size mismatch for (.*?):", str(e)):
            model_weights.pop(layer_name, None)
        model.load_state_dict(model_weights, strict=False)

    print(f"Weights loaded from: {cfg.architecture.pretrained_weights}")


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))




def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores


        

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        data = json.dumps(data)
        file.write(data)
        
        
def remove_files_from_directory(directory, verbose=False):
    """
    Removes all files and folders from directory.
    """
        
    filenames = os.listdir(directory)
    pathes = [os.path.join(directory, filename) for filename in filenames]
        
    for path in pathes:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

        if verbose:
            print(f"Removed '{path}' from '{directory}'.")
        
def make_directory(directory, overwriting=False):
    """
    Makes directory
    """
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        if overwriting:
            remove_files_from_directory(directory=directory)
    
    return directory


def apply_token_dropout(inputs, tokenizer, special_tokens_mask=None, p=0.0):
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs)
        
        if inputs.ndim == 1:
            inputs = torch.unsqueeze(inputs, dim=0)
            
    probability_matrix = torch.full(inputs.shape, p)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.clone().tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = torch.tensor(special_tokens_mask).bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), inputs.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return inputs



# ====================================================
# CPC Utils
# ====================================================


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



#### -----------------------------
###  FOLDS
#### ------------------------------

import random, os, gc
import numpy as np
import pandas as pd
from collections import Counter, defaultdict


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices





def create_folds(data, num_splits=5):
    # we create a new column called kfold and fill it with -1
    data["fold"] = -1
    
    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also just round it
    # num_bins = int(np.floor(1 + np.log2(len(data))))
    # LOGGER.info(f"kfold num_bins: {num_bins}")
    # bin targets
    # data.loc[:, "bins"] = pd.cut(data["score"], bins=num_bins, labels=False)
    
    #### initiate the kfold class from model_selection module

    # kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    kf = StratifiedGroupKFold(n_splits=num_splits)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.label.values, groups=data.essay_id)): ## 
        data.loc[v_, 'fold'] = f
    
    data['fold'] = data['fold'].astype(int)

    # drop the bins column
    # data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data






#### -----------------------------
###  FEEDBACK I
#### ------------------------------

def _prepare_training_data_helper(args, tokenizer, df, train_ids):
    training_samples = []
    for idx in tqdm(train_ids):
        filename = os.path.join(args.input, "train", idx + ".txt")
        temp_df = df[df["id"] == idx]
        max_discourse_end=temp_df.discourse_end.max() 
        #max_discourse_end=temp_df.new_end.max() 
        with open(filename, "r") as f:
            text = f.read(int(max_discourse_end))

        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded_text["input_ids"]
        input_labels = copy.deepcopy(input_ids)
        offset_mapping = encoded_text["offset_mapping"]

        for k in range(len(input_labels)):
            input_labels[k] = "O"

        sample = {
            "id": idx,
            "input_ids": input_ids,
            "text": text,
            "offset_mapping": offset_mapping,
        }

        #temp_df = df[df["id"] == idx]
        for _, row in temp_df.iterrows():
            text_labels = [0] * len(text)
            #discourse_start = int(row["discourse_start"])
            #discourse_end = int(row["discourse_end"])
            discourse_start = int(row["new_start"])
            discourse_end = int(row["new_end"])
            prediction_label = row["discourse_type"]
            text_labels[discourse_start:discourse_end] = [1] * (discourse_end - discourse_start)
            target_idx = []
            for map_idx, (offset1, offset2) in enumerate(encoded_text["offset_mapping"]):
                if sum(text_labels[offset1:offset2]) > 0:
                    if len(text[offset1:offset2].split()) > 0:
                        target_idx.append(map_idx)

            targets_start = target_idx[0]
            targets_end = target_idx[-1]
            pred_start = "B-" + prediction_label
            pred_end = "I-" + prediction_label
            input_labels[targets_start] = pred_start
            input_labels[targets_start + 1 : targets_end + 1] = [pred_end] * (targets_end - targets_start)

        sample["input_ids"] = input_ids
        sample["input_labels"] = input_labels
        training_samples.append(sample)
    return training_samples


def prepare_training_data(df, tokenizer, args, num_jobs,valid=False):
    training_samples = []
    train_ids = df["id"].unique()

    train_ids_splits = np.array_split(train_ids, num_jobs)
    if os.path.exists(f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/train_dataset_f{args.fold}.pkl') and not valid:
        training_samples = pickle.load( open( f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/train_dataset_f{args.fold}.pkl', "rb" ) )
        print('[INFO] Load train data from ', f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/train_dataset_f{args.fold}.pkl')
        return training_samples
    if os.path.exists(f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/valid_dataset_f{args.fold}.pkl') and valid :
        training_samples = pickle.load( open( f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/valid_dataset_f{args.fold}.pkl', "rb" ) )
        print('[INFO] Load train data from ', f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/valid_dataset_f{args.fold}.pkl')
        return training_samples
    else:
        
        results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
            delayed(_prepare_training_data_helper)(args, tokenizer, df, idx) for idx in train_ids_splits
        )
        for result in results:
            training_samples.extend(result)
        if valid:        
            pickle.dump( training_samples, open( f"/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/valid_dataset_f{args.fold}.pkl", "wb" ) )
        else:
            pickle.dump( training_samples, open( f"/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/train_dataset_f{args.fold}.pkl", "wb" ) )
            
        return training_samples

################################

def _prepare_training_data_helper2(args, tokenizer, df, train_ids):
    training_samples = []
    for idx in tqdm(train_ids):
        filename = os.path.join(args.input, "train", idx + ".txt")
        temp_df = df[df["id"] == idx]
        max_discourse_end=temp_df.discourse_end.max() 
        #max_discourse_end=temp_df.new_end.max() 
        with open(filename, "r") as f:
            text = f.read(int(max_discourse_end))

        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded_text["input_ids"]
        input_labels = copy.deepcopy(input_ids)
        offset_mapping = encoded_text["offset_mapping"]

        for k in range(len(input_labels)):
            input_labels[k] = "O"

        sample = {
            "id": idx,
            "input_ids": input_ids,
            "text": text,
            "offset_mapping": offset_mapping,
        }

        #temp_df = df[df["id"] == idx]
        for _, row in temp_df.iterrows():
            text_labels = [0] * len(text)
            discourse_start = int(row["discourse_start"])
            discourse_end = int(row["discourse_end"])
            #discourse_start = int(row["new_start"])
            #discourse_end = int(row["new_end"])
            prediction_label = row["discourse_type"]
            text_labels[discourse_start:discourse_end] = [1] * (discourse_end - discourse_start)
            target_idx = []
            for map_idx, (offset1, offset2) in enumerate(encoded_text["offset_mapping"]):
                if sum(text_labels[offset1:offset2]) > 0:
                    if len(text[offset1:offset2].split()) > 0:
                        target_idx.append(map_idx)

            targets_start = target_idx[0]
            targets_end = target_idx[-1]
            pred_start = "B-" + prediction_label
            pred_end = "I-" + prediction_label
            input_labels[targets_start] = pred_start
            input_labels[targets_start + 1 : targets_end + 1] = [pred_end] * (targets_end - targets_start)

        sample["input_ids"] = input_ids
        sample["input_labels"] = input_labels
        training_samples.append(sample)
    return training_samples


def prepare_training_data2(df, tokenizer, args, num_jobs,valid=False):
    training_samples = []
    train_ids = df["id"].unique()

    train_ids_splits = np.array_split(train_ids, num_jobs)
    if os.path.exists(f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/train_dataset_f{args.fold}.pkl') and not valid:
        training_samples = pickle.load( open( f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/train_dataset_f{args.fold}.pkl', "rb" ) )
        print('[INFO] Load train data from ', f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/train_dataset_f{args.fold}.pkl')
        return training_samples
    if os.path.exists(f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/valid_dataset_f{args.fold}.pkl') and valid :
        training_samples = pickle.load( open( f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/valid_dataset_f{args.fold}.pkl', "rb" ) )
        print('[INFO] Load train data from ', f'/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/valid_dataset_f{args.fold}.pkl')
        return training_samples
    else:
        results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
            delayed(_prepare_training_data_helper2)(args, tokenizer, df, idx) for idx in train_ids_splits
        )
        for result in results:
            training_samples.extend(result)
        if valid:
            pickle.dump( training_samples, open( f"/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/valid_dataset_f{args.fold}.pkl", "wb" ) )
        else:
            pickle.dump( training_samples, open( f"/mnt/md1/Work/dev_IME/Git_hacks/NLP/input/train_dataset_f{args.fold}.pkl", "wb" ) )
        return training_samples



##############################

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
    This code is from Rob Mulla's Kaggle kernel.
    """
    gt_df = gt_df[["id", "discourse_type", "predictionstring"]].reset_index(drop=True).copy()
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
        pred_subset = pred_df.loc[pred_df["class"] == discourse_type].reset_index(drop=True).copy()
        class_score = score_feedback_comp_micro(pred_subset, gt_subset)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


# class FeedbackDatasetValid:
#     def __init__(self, samples, max_len, tokenizer):
#         self.samples = samples
#         self.max_len = max_len
#         self.tokenizer = tokenizer
#         self.length = len(samples)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         input_ids = self.samples[idx]["input_ids"]
#         input_ids = [self.tokenizer.cls_token_id] + input_ids

#         if len(input_ids) > self.max_len - 1:
#             input_ids = input_ids[: self.max_len - 1]

#         # add end token id to the input_ids
#         input_ids = input_ids + [self.tokenizer.sep_token_id]
#         attention_mask = [1] * len(input_ids)

#         return {
#             "ids": input_ids,
#             "mask": attention_mask,
#         }


# class Collate:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer

#     def __call__(self, batch):
#         output = dict()
#         output["ids"] = [sample["ids"] for sample in batch]
#         output["mask"] = [sample["mask"] for sample in batch]

#         # calculate max token length of this batch
#         batch_max = max([len(ids) for ids in output["ids"]])

#         # add padding
#         if self.tokenizer.padding_side == "right":
#             output["ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["ids"]]
#             output["mask"] = [s + (batch_max - len(s)) * [0] for s in output["mask"]]
#         else:
#             output["ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["ids"]]
#             output["mask"] = [(batch_max - len(s)) * [0] + s for s in output["mask"]]

#         # convert to tensors
#         output["ids"] = torch.tensor(output["ids"], dtype=torch.long)
#         output["mask"] = torch.tensor(output["mask"], dtype=torch.long)

#         return output


# class EarlyStopping(Callback):
#     def __init__(
#         self,
#         model_path,
#         valid_df,
#         valid_samples,
#         batch_size,
#         tokenizer,
#         patience=5,
#         mode="max",
#         delta=0.001,
#         save_weights_only=True,
#         max_len=4096,
#     ):
#         self.patience = patience
#         self.counter = 0
#         self.mode = mode
#         self.best_score = None
#         self.early_stop = False
#         self.delta = delta
#         self.save_weights_only = save_weights_only
#         self.model_path = model_path
#         self.valid_samples = valid_samples
#         self.batch_size = batch_size
#         self.valid_df = valid_df
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#         if self.mode == "min":
#             self.val_score = np.Inf
#         else:
#             self.val_score = -np.Inf

#     def on_epoch_end(self, model):
#         model.eval()
#         valid_dataset = FeedbackDatasetValid(self.valid_samples, self.max_len, self.tokenizer)
#         collate = Collate(self.tokenizer)

#         preds_iter = model.predict(
#             valid_dataset,
#             batch_size=self.batch_size,
#             n_jobs=-1,
#             collate_fn=collate,
#         )

#         final_preds = []
#         final_scores = []
#         for preds in preds_iter:
#             pred_class = np.argmax(preds, axis=2)
#             pred_scrs = np.max(preds, axis=2)
#             for pred, pred_scr in zip(pred_class, pred_scrs):
#                 final_preds.append(pred.tolist())
#                 final_scores.append(pred_scr.tolist())

#         for j in range(len(self.valid_samples)):
#             tt = [id_target_map[p] for p in final_preds[j][1:]]
#             tt_score = final_scores[j][1:]
#             self.valid_samples[j]["preds"] = tt
#             self.valid_samples[j]["pred_scores"] = tt_score

#         submission = []
#         min_thresh = {
#             "Lead": 9,
#             "Position": 5,
#             "Evidence": 14,
#             "Claim": 3,
#             "Concluding Statement": 11,
#             "Counterclaim": 6,
#             "Rebuttal": 4,
#         }
#         proba_thresh = {
#             "Lead": 0.7,
#             "Position": 0.55,
#             "Evidence": 0.65,
#             "Claim": 0.55,
#             "Concluding Statement": 0.7,
#             "Counterclaim": 0.5,
#             "Rebuttal": 0.55,
#         }

#         for _, sample in enumerate(self.valid_samples):
#             preds = sample["preds"]
#             offset_mapping = sample["offset_mapping"]
#             sample_id = sample["id"]
#             sample_text = sample["text"]
#             sample_pred_scores = sample["pred_scores"]

#             # pad preds to same length as offset_mapping
#             if len(preds) < len(offset_mapping):
#                 preds = preds + ["O"] * (len(offset_mapping) - len(preds))
#                 sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))

#             idx = 0
#             phrase_preds = []
#             while idx < len(offset_mapping):
#                 start, _ = offset_mapping[idx]
#                 if preds[idx] != "O":
#                     label = preds[idx][2:]
#                 else:
#                     label = "O"
#                 phrase_scores = []
#                 phrase_scores.append(sample_pred_scores[idx])
#                 idx += 1
#                 while idx < len(offset_mapping):
#                     if label == "O":
#                         matching_label = "O"
#                     else:
#                         matching_label = f"I-{label}"
#                     if preds[idx] == matching_label:
#                         _, end = offset_mapping[idx]
#                         phrase_scores.append(sample_pred_scores[idx])
#                         idx += 1
#                     else:
#                         break
#                 if "end" in locals():
#                     phrase = sample_text[start:end]
#                     phrase_preds.append((phrase, start, end, label, phrase_scores))

#             temp_df = []
#             for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
#                 word_start = len(sample_text[:start].split())
#                 word_end = word_start + len(sample_text[start:end].split())
#                 word_end = min(word_end, len(sample_text.split()))
#                 ps = " ".join([str(x) for x in range(word_start, word_end)])
#                 if label != "O":
#                     if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
#                         temp_df.append((sample_id, label, ps))

#             temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])

#             submission.append(temp_df)

#         submission = pd.concat(submission).reset_index(drop=True)
#         submission["len"] = submission.predictionstring.apply(lambda x: len(x.split()))

#         def threshold(df):
#             df = df.copy()
#             for key, value in min_thresh.items():
#                 index = df.loc[df["class"] == key].query(f"len<{value}").index
#                 df.drop(index, inplace=True)
#             return df

#         submission = threshold(submission)

#         # drop len
#         submission = submission.drop(columns=["len"])

#         scr = score_feedback_comp(submission, self.valid_df, return_class_scores=True)
#         print(scr)
#         model.train()

#         epoch_score = scr[0]
#         if self.mode == "min":
#             score = -1.0 * epoch_score
#         else:
#             score = np.copy(epoch_score)

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(epoch_score, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
#             if self.counter >= self.patience:
#                 model.model_state = enums.ModelState.END
#         else:
#             self.best_score = score
#             self.save_checkpoint(epoch_score, model)
#             self.counter = 0

#     def save_checkpoint(self, epoch_score, model):
#         if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
#             print("Validation score improved ({} --> {}). Saving model!".format(self.val_score, epoch_score))
#             model.save(self.model_path, weights_only=self.save_weights_only)
#         self.val_score = epoch_score

############ fix for CPC codes
# import re

# for cpc in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Y']:
#     with open(f'../input/cpc-data/CPCTitleList202202/cpc-section-{cpc}_20220201.txt') as f:
#         s = f.read()
    # pattern = f'{cpc}\t\t.+'
    # result = re.findall(pattern, s)
    # pattern = "^"+pattern[:-2]
    # cpc_result = re.sub(pattern, "", result[0])
    # for context in [c for c in contexts if c[0] == cpc]:
    #     pattern = f'{context}\t\t.+'
    #     result = re.findall(pattern, s)
    #     pattern = "^"+pattern[:-2]
    #     results[context] = cpc_result + ". " + re.sub(pattern, "", result[0])
 
 

def feature_generate(df):
    new_features=[]
    # count of char
    # rmk: as it is excerpt, so lengh may not matter
    df['char_count']=df['excerpt'].apply(len)
    new_features.append('char_count')
    # count of words
    df['word_count']=df['excerpt'].apply(lambda s:len(s.split(' ')))
    new_features.append('word_count')
    # count of sentence
    # rmk: what if end with '!' '?' ...
    df['sentence_count']=df['excerpt'].apply(lambda s:len(s.split('.')))
    new_features.append('sentence_count')
    # average char per word
    df['c2w']=df['char_count']/df['word_count']
    new_features.append('c2w')
    # avg word per sentence
    df['w2s']=df['word_count']/df['sentence_count']
    new_features.append('w2s')
    # count the unique words
    df['unique_word_count'] = df['excerpt'].apply(lambda s: len(set( s.split(' ') )))
    new_features.append('unique_word_count')
    # text diversity
    df['word_diversity'] = df['unique_word_count'] / df['word_count']
    new_features.append('word_diversity')
    # word lengths
    words = df['excerpt'].apply(lambda s: s.split(' '))
    word_lengths = words.apply(lambda s: [len(f) for f in s ])
    df['longest_w_len'] = word_lengths.apply(max)
    new_features.append('longest_w_len')
    df['avg_w_len'] = word_lengths.apply(np.mean)
    new_features.append('avg_w_len')
#     # word freq
#     word_freqs = words.apply(lambda s: [word_freq(f) for f in s ])
#     df['avg_freq']=word_freqs.apply(np.mean)
#     df['min_freq']=word_freqs.apply(min)
#     new_features.append('avg_freq')
#     new_features.append('min_freq')
    return new_features
    
# extra_features=feature_generate(train_df)


##### --------------------
## 
##  **** LOSSES 
##
###### -------------------



#         