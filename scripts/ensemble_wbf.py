
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

import warnings 
warnings.filterwarnings('ignore')

print('torch.__version__:', torch.__version__)
print('transformers.__version__:', transformers.__version__)





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

label_to_threshold = {
    0 : 0.15,  #0.325, #Lead          #0.275, #Claim
    1 : 0.15,  #0.325, #Position      #0.375, #Evidence
    2 : 0.1, #0.275, #Claim         #0.325, #Lead
    3 : 0.1,  #0.35, #0.375, #Evidence      #0.325, #Position
    4 : 0.15, #0.275, #Counterclaim  #0.4, #Concluding Statement
    5 : 0.15, #0.275, #Rebuttal      #0.275, #Counterclaim
    6 : 0.2,  #Concluding Statement #0.275 #Rebuttal
}

min_thresh = {
    'Lead': 3,
    'Position': 4,
    'Evidence': 4,
    'Claim': 2,
    'Concluding Statement': 9,
    'Counterclaim': 5,
    'Rebuttal': 2,
}

proba_thresh = {
    "Lead": 0.27,
    "Position": 0.28,
    "Evidence": 0.39,
    "Claim": 0.30,
    "Concluding Statement": 0.36,
    "Counterclaim": 0.21,
    "Rebuttal": 0.20,
}

# LABEL2TYPE = ('Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement')
LABELS_TYPE = [
    'Lead', #0
    'Position', #1 
    'Claim', #2
    'Evidence', #3
    'Counterclaim', #4
    'Rebuttal', #5
    'Concluding Statement' #6 
]

TYPE2LABEL = {t:l for l, t in enumerate(LABELS_TYPE)}
LABEL2TYPE = {k:v for v,k in TYPE2LABEL.items()}


## args
ap = argparse.ArgumentParser()
ap.add_argument('--csv_paths', nargs='+', required=True)
ap.add_argument("--save_name", type=str, required=True)
ap.add_argument("--save_dir", type=str, required=True)
ap.add_argument("--iou_th", type=float, default=0.33, required=False)
ap.add_argument("--debug", type=bool, default=False, required=False)
cargs = ap.parse_args()


TEST_CSV_PATH  = '/workspace/data/test.csv'

IOU_TH = cargs.iou_th
OUTPUT = cargs.save_dir
SAVE_NAME = cargs.save_name
CSV_PATHS = cargs.csv_paths

print()
print('num subs:', len(CSV_PATHS))
print(CSV_PATHS)

# Load test data 
df_test = pd.read_csv(TEST_CSV_PATH)
test_ids = df_test['essay_id_comp'].unique().tolist()

print('test shape:', df_test.shape)
print('num essays:', df_test.essay_id_comp.nunique())


### WBF utils 

'''
Code taken and modified for 1D sequences from:
https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py
'''

def prefilter_boxes(boxes, scores, labels, weights, thr, scores_eff0, scores_eff1):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = labels[t][j]
            box_part = boxes[t][j]

            x = float(box_part[0])
            y = float(box_part[1])

            # Box data checks
            if y < x:
                warnings.warn('Y < X value in box. Swap them.')
                x, y = y, x

            score_e0 = scores_eff0[t][j]
            score_e1 = scores_eff1[t][j]

            # # [label, score, weight, model index, x, y]
            # b = [label, float(score) * weights[t], weights[t], t, x, y]

            # [label, score, weight, model index, x, y, score_eff]
            b = [label, float(score) * weights[t], weights[t], t, x, y, score_e0, score_e1]

            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box (label, score, weight, model index, x, y)
    """

    # box = np.zeros(6, dtype=np.float32)
    # box = np.zeros(7, dtype=np.float32)
    box = np.zeros(8, dtype=np.float32)
    conf = 0
    conf_list = []
    scores_eff_list0 = []
    scores_eff_list1 = []
    score_eff0 = 0
    score_eff1 = 0
    w = 0
    for b in boxes:
        # box[4:] += (b[1] * b[4:])
        box[[4,5]] += (b[1] * b[[4,5]])
        conf += b[1]
        conf_list.append(b[1])

        score_eff0 += b[-2]
        scores_eff_list0.append(b[-2])

        score_eff1 += b[-1]
        scores_eff_list1.append(b[-1])

        w += b[2]
    box[0] = boxes[0][0]
    if conf_type == 'avg':
        box[1] = conf / len(boxes)
        box[6] = score_eff0 / len(boxes)
        box[7] = score_eff1 / len(boxes)

    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
        box[6] = np.array(scores_eff_list0).max()
        box[7] = np.array(scores_eff_list1).max()

    elif conf_type in ['box_and_model_avg', 'absent_model_aware_avg']:
        box[1] = conf / len(boxes)
        box[6] = score_eff0 / len(boxes)
        box[7] = score_eff1 / len(boxes)

    box[2] = w
    box[3] = -1 # model index field is retained for consistensy but is not used.
    # box[4:] /= conf
    box[[4,5]] /= conf
 
    return box


def find_matching_box_quickly(boxes_list, new_box, match_iou):
    """ 
        Reimplementation of find_matching_box with numpy instead of loops. Gives significant speed up for larger arrays
        (~100x). This was previously the bottleneck since the function is called for every entry in the array.

        boxes_list: shape: (N, label, score, weight, model index, x, y)
        new_box: shape: (label, score, weight, model index, x, y)
    """
    def bb_iou_array(boxes, new_box):
        '''
        boxes: shape: (N, x, y)
        new_box: shape: (x, y)
        '''
        # bb interesection over union
        x_min = np.minimum(boxes[:, 0], new_box[0])
        x_max = np.maximum(boxes[:, 0], new_box[0])
        y_min = np.minimum(boxes[:, 1], new_box[1])+1
        y_max = np.maximum(boxes[:, 1], new_box[1])+1

        iou = np.maximum(0, (y_min-x_max)/(y_max-x_min))

        return iou

    if boxes_list.shape[0] == 0:
        return -1, match_iou

    # boxes = np.array(boxes_list)
    boxes = boxes_list

    # ious = bb_iou_array(boxes[:, 4:], new_box[4:])
    ious = bb_iou_array(boxes[:, 4:6], new_box[4:6])

    ious[boxes[:, 0] != new_box[0]] = -1

    best_idx = np.argmax(ious)
    best_iou = ious[best_idx]

    if best_iou <= match_iou:
        best_iou = match_iou
        best_idx = -1

    return best_idx, best_iou

def weighted_boxes_fusion(boxes_list, scores_list, labels_list, scores_eff_list0, scores_eff_list1, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 2 numbers.
     It has 3 dimensions (models_number, model_preds, 2)
     Order of boxes: x, y.
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value, 'box_and_model_avg': box and model wise hybrid weighted average, 'absent_model_aware_avg': weighted average that takes into account the absent model.
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x, y).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()

    # filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr, scores_eff_list0, scores_eff_list1)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        # weighted_boxes = np.empty((0,6)) ## [label, score, weight, model index, x, y]
        # weighted_boxes = np.empty((0,7))   ## [label, score, weight, model index, x, y, score_eff]
        weighted_boxes = np.empty((0,8))   ## [label, score, weight, model index, x, y, score_eff0, score_eff1]

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box_quickly(weighted_boxes, boxes[j], iou_thr)

            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes = np.vstack((weighted_boxes, boxes[j].copy()))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[i])
            if conf_type == 'box_and_model_avg':
                # weighted average for boxes
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weighted_boxes[i, 2]
                # identify unique model index by model index column
                _, idx = np.unique(clustered_boxes[:, 3], return_index=True)
                # rescale by unique model weights
                weighted_boxes[i, 1] = weighted_boxes[i, 1] *  clustered_boxes[idx, 2].sum() / weights.sum()
            elif conf_type == 'absent_model_aware_avg':
                # get unique model index in the cluster
                models = np.unique(clustered_boxes[:, 3]).astype(int)
                # create a mask to get unused model weights
                mask = np.ones(len(weights), dtype=bool)
                mask[models] = False
                # absent model aware weighted average
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / (weighted_boxes[i, 2] + weights[mask].sum())
            elif conf_type == 'max':
                weighted_boxes[i, 1] = weighted_boxes[i, 1] / weights.max()
            elif not allows_overflow:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * min(len(weights), len(clustered_boxes)) / weights.sum()
            else:
                weighted_boxes[i, 1] = weighted_boxes[i, 1] * len(clustered_boxes) / weights.sum()
        
        for i in range(len(new_boxes)):
            clustered_boxes = np.array(new_boxes[i])
            if len(np.unique(clustered_boxes[:, 3])) > 1:
                overall_boxes.append(weighted_boxes[i])

    overall_boxes = np.array(overall_boxes)
    # print('overall_boxes', overall_boxes.shape)
    
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    # boxes  = overall_boxes[:, 4:]
    boxes  = overall_boxes[:, 4:6]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    
    scores_eff0 = overall_boxes[:, -2]
    scores_eff1 = overall_boxes[:, -1]

    return boxes, scores, labels, scores_eff0, scores_eff1

import math
from tqdm import tqdm 

# from utils.wbf import weighted_boxes_fusion

def preprocess_for_wbf(df_list):
    boxes_list=[]
    scores_list=[]
    labels_list=[]
    scores_eff_list0=[]
    scores_eff_list1=[]
    
    for df in df_list:
        scores_list.append(df['scores'].values.tolist())
        labels_list.append(df['class'].map(rater_class_map).values.tolist()) ##
        # labels_list.append(df['discourse_type'].values.tolist())
        predictionstring = df.predictionstring.str.split().values
        
        ## add effective_score
        scores_eff_list0.append(df['score_discourse_effectiveness_0'].values.tolist())
        scores_eff_list1.append(df['score_discourse_effectiveness_1'].values.tolist())
        # scores_eff_list.append(df['scores_eff'].values.tolist())

        df_box_list = []
        for bb in predictionstring:
            df_box_list.append([int(bb[0]), int(bb[-1])])
        boxes_list.append(df_box_list)
    return boxes_list, scores_list, labels_list, scores_eff_list0, scores_eff_list1


def postprocess_for_wbf(idx, boxes_list, scores_list, labels_list, scores_eff_list0, scores_eff_list1):
    preds = []
    for box, score, label, score_eff0, score_eff1 in zip(boxes_list, scores_list, labels_list, scores_eff_list0, scores_eff_list1):
        if score > label_to_threshold[label]: 
            start = int(math.ceil(box[0]))
            end = int(box[1])
            preds.append((
                idx, 
                inv_rater_class_map[label], 
                ' '.join([str(x) for x in range(start, end+1)]), #' '.join(np.arange(start,end+1).astype(str))
                score, 
                score_eff0, 
                score_eff1,
                start, end 
                ))
        ### ++
        else:
            preds.append((
                idx, 
                inv_rater_class_map[label], 
                ' '.join([str(x) for x in range(5)]), 
                0.0, 
                1.0, 0.0, 
                0, 5
                ))
            print('No score > threshold', idx)
    return preds


def convert_to_rater_format(pred_df):

    ## convert effective scores to RATER format 
    pred_df['score_discourse_effectiveness_0'] = -100
    pred_df['score_discourse_effectiveness_1'] = -100

    for i, row in pred_df.iterrows():
        if row['class_eff'] == 'Effective':
            pred_df.loc[i, 'score_discourse_effectiveness_1'] = row['scores_eff']
            pred_df.loc[i, 'score_discourse_effectiveness_0'] = 1 - row['scores_eff']
        elif row['class_eff'] == 'Non-Effective':
            pred_df.loc[i, 'score_discourse_effectiveness_0'] = row['scores_eff']
            pred_df.loc[i, 'score_discourse_effectiveness_1'] = 1 - row['scores_eff']
        else:
            # print(row['class_eff'])
            pred_df.loc[i, 'score_discourse_effectiveness_0'] = row['scores_eff']
            pred_df.loc[i, 'score_discourse_effectiveness_1'] = 1 - row['scores_eff']

    return pred_df


## For simple post-processing 
def do_threshold(submit_df, min_thresh, proba_thresh, use=['length','probability']):
    df = submit_df.copy()
    df = df.fillna('')

    if 'length' in use:
        df['l'] = df.predictionstring.apply(lambda x: len(x.split()))
        for key, value in min_thresh.items():
            index = df.loc[df['class'] == key].query('l<%d'%value).index
            df.drop(index, inplace=True)

    if 'probability' in use:
        for key, value in proba_thresh.items():
            index = df.loc[df['class'] == key].query('scores<%f'%value).index
            df.drop(index, inplace=True)
    return df



### -----------------------
### Load submission files 
### -----------------------

all_dfs = []

pred61_ff = pd.read_csv(CSV_PATHS[0]).dropna() #0
pred61_ff = pred61_ff.rename(columns={
    'score': 'scores',
    'score_eff': 'scores_eff',
})
all_dfs.extend([pred61_ff]) 

pred66_ff = pd.read_csv(CSV_PATHS[1]).dropna() #1
pred66_ff = pred66_ff.rename(columns={
    'score': 'scores',
    'score_eff': 'scores_eff',
})
all_dfs.extend([pred66_ff]) 

ww = [2.0, 1.0]

# if len(CSV_PATHS)==3:
pred62 = pd.read_csv(CSV_PATHS[2]).dropna()
pred62 = pred62.rename(columns={
    'score': 'scores', 'score_eff': 'scores_eff',
    })
all_dfs.extend([pred62])
# ww.append([1.0]) 


ww = [2.0, 1.0, 1.0]

## summary 
for df_ in all_dfs:
    print('num ids:', df_['id'].nunique())
    print('shape:', df_.shape)

### -----------------------
### Run WBF ensemble 
### -----------------------

def generate_wbf_for_id(i, iou_thr=0.33, weights=None):
     
    new_dfs = []
    for df_ in all_dfs:
        new_dfs.append(df_[df_['id']==i])

    boxes_list, scores_list, labels_list, scores_eff_list0, scores_eff_list1 = preprocess_for_wbf(new_dfs)

    nboxes_list, nscores_list, nlabels_list, nscores_eff_list0, nscores_eff_list1 = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, scores_eff_list0, scores_eff_list1,
        weights=weights, #ww,  
        iou_thr=iou_thr, 
        conf_type='avg',  #'max',
        )
    return postprocess_for_wbf(i, nboxes_list, nscores_list, nlabels_list, nscores_eff_list0, nscores_eff_list1)


missing_ids = []
ens_preds = []
for uid in tqdm(test_ids, total=len(test_ids)):
    try:
        res = generate_wbf_for_id(uid, IOU_TH)
        if len(res)==0:
            res = generate_wbf_for_id(uid, 0.3)
            print('--> WBF with lower th', uid)
            if len(res)==0:
                res = generate_wbf_for_id(uid, 0.25)
                print('--> WBF with lower th', uid)
        ens_preds.extend(res)
    except:
        missing_ids.append(uid)
        print(uid)

wbf_preds = pd.DataFrame(ens_preds)
wbf_preds.columns = ["essay_id_comp", "class", "predictionstring", "scores", "score_discourse_effectiveness_0", "score_discourse_effectiveness_1", "start", "end"]

print('num missing_ids:', len(missing_ids))

if len(missing_ids)>0:
    df_ = all_dfs[0][all_dfs[0]['id'].isin(missing_ids)]
    df_['essay_id_comp'] = df_['id']
    wbf_preds = pd.concat([wbf_preds, df_[wbf_preds.columns]], ignore_index=True)

print('num ids:', wbf_preds.essay_id_comp.nunique())

### sanity check - fix 
print(wbf_preds['predictionstring'].isnull().sum().sum())

wbf_preds['predictionstring'] = wbf_preds['predictionstring'].apply(lambda x: np.NaN if x=="" else x)
print(wbf_preds['predictionstring'].isnull().sum())

wbf_preds = wbf_preds.dropna(subset=['predictionstring'])
print(wbf_preds['predictionstring'].isnull().sum().sum())

## Submission prep
sample_sub = pd.read_csv('/workspace/data/sample_submission.csv')
COLS = sample_sub.columns.to_list()
# sample_sub

## PP 
print('shape before pp:', wbf_preds.shape[0])
wbf_preds = do_threshold(wbf_preds, min_thresh, proba_thresh, use=['length','probability'])
print('shape after pp:', wbf_preds.shape[0])


if wbf_preds.essay_id_comp.nunique() == 10402:

    sub = wbf_preds.copy()

    ### convert discource classes to int 
    sub["discourse_type"] = sub["class"].map(rater_class_map)

    ### convert to sample submission format 
    sub = sub[COLS]
    print(sub.essay_id_comp.nunique())

    #########
    sub.to_csv(f'/workspace/results/submission_final.csv', index=False)
    print('Submission saved!')


##### Summary 
print()
print(sub.discourse_type.value_counts())


