# %%
### Inference based on FB1 1st solution
# longformer train: https://www.kaggle.com/wht1996/feedback-nn-train
# lgb train: https://www.kaggle.com/wht1996/feedback-lgb-train
# 5fold longfomer with post-processing(lb 0.697): https://www.kaggle.com/wht1996/feedback-longformer-5fold-0-697
# 5fold longfomer with lgb(lb 0.727): https://www.kaggle.com/wht1996/feedback-two-stage-lb0-727
# cv ensemble with lgb(cv 0.747): https://www.kaggle.com/wht1996/feedback-two-stage-cv0-747
# code on GitHub: https://github.com/antmachineintelligence/Feedback_1st

# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# CUDA_LAUNCH_BLOCKING=1

# %%
DEBUG = False
# DEBUG = True

# %%
import os, sys, gc 
import numpy as np
import pandas as pd
import time
import pickle
import random
from tqdm import tqdm
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score
import pdb
import copy
import math
from joblib import Parallel, delayed


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau,_LRScheduler

from transformers import AutoTokenizer, AutoModel, AutoConfig
 

import warnings
warnings.filterwarnings('ignore')

os.chdir('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3')
sys.path.append('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/')
sys.path.append('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/src/')

### CONFIG 

def get_args():
        
    args = parser.parse_args()
    # args = CFG
    args.model_name = CFG.model_name
    args.fold = 0 
    args.device = 'cuda'
    args.log_path = "log.txt"
    args.data_path = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/"
    args.train_batch_size = 4
    args.valid_batch_size = 4
    args.train_padding_side = 'random'
    args.test_padding_side = 'right'
    args.lr = 0.00001
    args.min_lr = 0.000001
    args.max_lr = 0.00001
    args.adv_lr = 0.0000
    args.adv_eps = 0.001
    args.max_grad_norm = 10
    args.load_model = False
    args.load_feat = False
    args.key_string = ''

    if 'longformer' in args.model_name:
        args.model_length=None
    elif 'bigbird' in args.model_name:
        args.model_length=None
    elif 'funnel-transformer' in args.model_name:
        args.model_length=None
    elif 'roberta' in args.model_name:
        args.model_length=512
        args.train_padding_side='random'
    elif 'albert-xxlarge-v2' in args.model_name:
        args.model_length=512
        args.train_padding_side='random'
    elif 'bert-large-NER' in args.model_name:
        args.model_length=512
        args.train_padding_side='random'
    elif 'electra' in args.model_name:
        args.model_length=512
        args.train_padding_side='random'
    elif 'gpt2' in args.model_name:
        args.model_length=1024
    elif 'distilbart' in args.model_name:
        args.model_length=1024
        args.train_padding_side='random'
    elif 'bart-large' in args.model_name:
        args.model_length=1024
        args.train_padding_side='random'
    elif 'deberta' in args.model_name:
        args.model_length=None
    else:
        args.model_length=None
    args.padding_dict = {'input_ids':0,'attention_mask':0,'labels':-100}
    args.epochs = 2 if args.debug else args.epochs
    args.key_string = args.model_name.split('/')[-1] + \
            f'_max_length_{args.max_length}' + \
            f'_fold{args.fold}' + \
            args.key_string + \
            ('_debug' if args.debug else '')
            # '_aistudio_15class' + \
            # f'_padding_{args.train_padding_side}' + \
            # (f'_adv{args.adv_lr}' if args.adv_lr>0 else '') + \
            # f'_max_lr_{args.max_lr}' + \
    
    log = Log(OUTPUT / f'infer.log', time_key=False)
    log('args:{}'.format(str(args)))
    return args, log






# %% Utils 

# Function to seed everything
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def calc_overlap(row):
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    return inter/max(len_gt,len_pred)


def get_f1_score(test_pred,test_df,log=print,slient=True):
    if not slient:
        log('test_pred.shape:',test_pred.shape,'\ttest_df.shape:',test_df.shape,)
        log('pred class:\n',test_pred['class'].value_counts())
        log('true class:\n',test_df['discourse_type'].value_counts())
    f1s = []

    for c in sorted(test_pred['class'].unique()):
        pred_df = test_pred.loc[test_pred['class']==c].copy()
        gt_df = test_df.loc[test_df['discourse_type']==c].copy()

        gt_df = gt_df[['id','discourse_type','predictionstring']].reset_index(drop=True).copy()
        pred_df = pred_df[['id','class','predictionstring']].reset_index(drop=True).copy()
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

    #     print(joined.head())
        joined['min_overlaps'] = joined.apply(calc_overlap, axis=1)
        joined['potential_TP'] = (joined['min_overlaps'] >= 0.5)

        matched_pred_ids = joined.query('potential_TP')['pred_id'].unique()
        fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in matched_pred_ids]

        matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
        fn_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

        # Get numbers of each type
        TP = len(matched_gt_ids)
        FP = len(fp_pred_ids)
        FN = len(fn_gt_ids)
        #calc microf1
        f1_score = TP / (TP + 0.5*(FP+FN))
        if not slient:
            log(f'{c:<20} f1 score:\t{f1_score}')
        f1s.append(f1_score)
    log('\nOverall f1 score \t',np.mean(f1s))

    return np.mean(f1s)




class Log:
    def __init__(self,log_path,time_key=True):
        self.path = log_path
        if time_key:
            self.path = self.path.replace('.','{}.'.format(time.strftime('_%Y%m%d%H%M%S',time.localtime(time.time()))))
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),file=open(self.path,'a+'))
        print('log path:', self.path)
        print('****************begin*********************',file=open(self.path,'a+'))
    
    def __call__(self,*content):
        t1 = time.strftime('%H:%M:%S',time.localtime(time.time()))
        print(*content)
        print(t1,content,file=open(self.path,'a+'))
    
    def clean(self):
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),file=open(self.path,'w'))
        print('****************begin*********************',file=open(self.path,'a+'))


# from utils.utils import CosineAnnealingWarmupRestarts, EarlyStopping

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr)
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr       


def mapping_to_ids(mapping,text):
    mapping[0] = 0 if mapping[0]==1 else mapping[0]
    start = len(text[:mapping[0]].split())
    end = len(text[:mapping[1]].split())
    return [str(i) for i in range(start,end)]

###----------------------------------
### Train engine helpers 
###----------------------------------

def test(model,test_loader,test_feat,test_df,args,log):
    model.eval()
    te_loss,te_accuracy = [],[]
    test_pred = []
    scaler = torch.cuda.amp.GradScaler()
    with torch.no_grad():
        for batch in test_loader:
            # with torch.cuda.amp.autocast():
            #     loss, te_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            #     loss = loss.mean()
            if args.test_padding_side in ['right','left']:
                loss, te_logits = model_predict(model, batch, model_length=args.model_length,
                                                padding_dict=args.padding_dict,padding_side=args.test_padding_side)
            elif args.test_padding_side == 'double':
                loss1, te_logits1 = model_predict(model, batch, model_length=args.model_length,
                                                padding_dict=args.padding_dict,padding_side='right')
                loss2, te_logits2 = model_predict(model, batch, model_length=args.model_length,
                                                padding_dict=args.padding_dict,padding_side='left')
                loss = (loss1+loss2)/2
                te_logits = [(l1+l2)/2 for l1,l2 in zip(te_logits1,te_logits2)]
            # compute training accuracy
            for i in range(len(te_logits)):
                pred = te_logits[i].argmax(axis=-1)
                token_label = batch['labels'][i][batch['attention_mask'][i]>0].cpu().numpy().reshape(-1)
                te_accuracy.append(accuracy_score(pred, token_label))
            te_loss.append(loss.item())
            test_pred.extend(te_logits)
    te_accuracy = np.mean(te_accuracy)
    te_loss     = np.mean(te_loss)
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    
    test_feat['pred'] = test_pred
    segment_param = {
    "Lead":                 {'min_proba':[0.40,0.30],'begin_proba':1.00,'min_sep':30,'min_length': 5},
    "Position":             {'min_proba':[0.50,0.30],'begin_proba':1.00,'min_sep':15,'min_length': 1},
    "Evidence":             {'min_proba':[0.50,0.40],'begin_proba':1.00,'min_sep': 3,'min_length':10},
    "Claim":                {'min_proba':[0.40,0.30],'begin_proba':0.30,'min_sep':10,'min_length': 1},
    "Concluding Statement": {'min_proba':[0.50,0.40],'begin_proba':1.00,'min_sep':40,'min_length':10},
    "Counterclaim":         {'min_proba':[0.50,0.20],'begin_proba':1.00,'min_sep':10,'min_length': 5},
    "Rebuttal":             {'min_proba':[0.30,0.00],'begin_proba':1.00,'min_sep': 5,'min_length': 5},
    }

    test_predictionstring = after_deal(test_feat, args.labels_to_ids, segment_param,log,acc=True)
    f1_score = get_f1_score(test_predictionstring,test_df,log)
    
    return te_loss,te_accuracy,f1_score,test_pred
   
    
def train(model,train_loader,test_loader,test_feat,test_df,args,model_path,log):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmupRestarts(optimizer = optimizer,
                 first_cycle_steps = args.num_train_steps, cycle_mult = 1,
                 max_lr = args.max_lr, min_lr = args.min_lr, warmup_steps = args.num_train_steps * 0.2,
                 gamma = 1.,last_epoch = -1
    )
    es = EarlyStopping(patience=2,max_epoch=args.epochs)
    t0 = time.time()
    scaler = torch.cuda.amp.GradScaler()
    awp = AWP(model,
          optimizer,
          adv_lr=args.adv_lr,
          adv_eps=args.adv_eps,
          start_epoch=args.num_train_steps/args.epochs,
          scaler=scaler
             )
    step = 0
    for epoch in range(100):
    
        tr_loss, tr_accuracy = [], []
        nb_tr_steps = 0

        model.train()
        for idx, batch in enumerate(train_loader):
            step += 1
            input_ids = batch['input_ids'].to(args.device, dtype = torch.long)
            attention_mask = batch['attention_mask'].to(args.device, dtype = torch.long)
            labels = batch['labels'].to(args.device, dtype = torch.long)
            
            with torch.cuda.amp.autocast():
                loss, tr_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss.mean()
            
            optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            scaler.scale(loss).backward()
            awp.attack_backward(input_ids,labels,attention_mask,step)  # 更新为对抗梯度
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            tr_loss.append(loss.item())
            nb_tr_steps += 1
            
            # compute training accuracy
            for i in range(len(tr_logits)):
                pred = tr_logits[i][attention_mask[i]>0].detach().cpu().numpy().argmax(axis=-1)
                token_label = labels[i][attention_mask[i]>0].cpu().numpy().reshape(-1)
                tr_accuracy.append(accuracy_score(pred, token_label))
            
            if idx % 200==0:
                del input_ids,attention_mask,labels,loss, tr_logits
                gc.collect()
                torch.cuda.empty_cache()
                
                tr_loss_   = np.mean(tr_accuracy)
                tr_accuracy_ = np.mean(tr_accuracy)
                
                log(f"step: \t{idx:04d}, train loss: \t{tr_loss_:.4f}, train acc: \t{tr_accuracy_:.4f}, time: \t{int(time.time()-t0)}s")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        tr_loss_   = np.mean(tr_loss)
        tr_accuracy_ = np.mean(tr_accuracy)
        te_loss, te_accuracy,f1_score, test_pred = test(model,test_loader,test_feat,test_df,args,log)
        log(f"epoch: \t{epoch:04d}, train loss: \t{tr_loss_:.4f}, train acc: \t{tr_accuracy_:.4f}, test loss: \t{te_loss:.4f}, test acc: \t{te_accuracy:.4f}, test f1: \t{f1_score:.4f}, time: \t{int(time.time()-t0)}s")
        es(f1_score,model,model_path=model_path)
        if es.early_stop:
            break
    
    return es.best_model.to(next(model.parameters()).device),test_pred

    
class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, x, y, attention_mask,epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                adv_loss, tr_logits = self.model(input_ids=x, attention_mask=attention_mask, labels=y)
            
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                # 保存原始参数
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}



def mapping_to_ids(mapping, text):
    word_start = len(text[:mapping[0]].split())
    word_end = word_start + len(text[mapping[0]:mapping[1]].split())
    word_end = min(word_end, len(text.split()))
    output = " ".join([str(x) for x in range(word_start, word_end)])
    return output

sentence_split = [';', ',', '.', '?', '!', '"']
null_symbol = ['\xa0', '\n', '\x93', ' ']
def get_sentence_split(text, pred_head=[]):
    start = []
    end = True
    for i,t in enumerate(text):
        if t == ' ':
            continue
        if i in pred_head:
            start.append(i)
            end = False
        if (t in sentence_split and text[i-1] not in null_symbol) or \
            (t in null_symbol):
            end = True
            if (t in null_symbol) and (text[i-1] not in null_symbol):
                start.append(i)
        elif end == True:
            start.append(i)
            end = False
        else:
            pass
    result = []
    for start,end in zip(start,start[1:]+[len(text)]):
#         lenght = len(text[start:end].strip())
        result.append([start,end])
    return result



def word_decode(pred, b_pred, offset_mapping, text, **kwargs):

    pred_head = []
    for i,(p,b_p,mapping) in enumerate(zip(pred,b_pred,offset_mapping)):
        if mapping == (0,0):
            continue
        if b_p >= kwargs['begin_proba'] or (abs(p-pred[i-1])>0.1):
            pred_head.append(mapping[0])
    sentence = get_sentence_split(text, pred_head)

    sentence_offset_mapping = [[]]
    sentence_pred = [[]]
    sentence_b_pred = [[]]
    sentence_idx = [[]]
    i_sentence = 0
    for i,(p,b_p,mapping) in enumerate(zip(pred,b_pred,offset_mapping)):  # token分组
        if mapping == (0,0) or mapping[0]==mapping[1]:
            continue
        if i_sentence == len(sentence):
            print(text)
            print(mapping)
        if mapping[1] <= sentence[i_sentence][1]:
            sentence_offset_mapping[i_sentence].append(mapping)
            sentence_pred[i_sentence].append(p)
            sentence_b_pred[i_sentence].append(b_p)
            sentence_idx[i_sentence].append(i)
        else:
            i_sentence += 1
            sentence_offset_mapping.append([mapping])
            sentence_pred.append([p])
            sentence_b_pred.append([b_p])
            sentence_idx.append([i])
    
    sentence_offset_mapping2 = []
    sentence_pred2 = []
    sentence_b_pred2 = []
    sentence_idx2 = []       
    for t,mapping_,pred_,b_pred_,idx_ in zip(sentence,sentence_offset_mapping,sentence_pred,sentence_b_pred,sentence_idx):
#         print(np.round(pred_,2),'\t',text[t[0]:t[1]])
        if np.mean(pred_) > kwargs['min_proba'][0]:
            sentence_offset_mapping2.append([mapping_[0][0],mapping_[-1][1]])
            sentence_pred2.append(pred_)
            sentence_b_pred2.append(b_pred_)
            sentence_idx2.append(idx_)

    sentence_offset_mapping3 = []
    sentence_idx3 = []
    for i, mapping_ in enumerate(sentence_offset_mapping2):
        sep_start = sentence_idx2[i-1][-1]+1
        sep_end = sentence_idx2[i][0]
        sep_length = sep_end-sep_start
        if i == 0:                    
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
#         elif text[mapping_[0]-1] == '\n' and sentence_b_pred2[i][0]>kwargs['line_prob']:
#             sentence_offset_mapping3.append(mapping_)
#             sentence_idx3.append(sentence_idx2[i])
        elif sentence_b_pred2[i][0] > kwargs['begin_proba']:             
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        elif sep_length >= kwargs['min_sep']:                       
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        elif sep_length/kwargs['min_sep'] - np.mean(pred[sep_start:sep_end])/kwargs['min_proba'][1] > 0:  
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        else:
            sentence_offset_mapping3[-1][1] = mapping_[1]
            sentence_idx3[-1].extend(sentence_idx2[i])

    sentence_offset_mapping4 = []
    for i,mapping in enumerate(sentence_offset_mapping3):
        word_length = len(text[mapping[0]: mapping[1]].split())
        if word_length <= kwargs['min_length']:
            continue
#         if word_length >= kwargs['min_length'][1]: 
#             continue
#         if sum(pred[sentence_idx3[i][0] : sentence_idx3[i][-1]+1]) <= kwargs['min_proba'][0]*kwargs['min_length'][1]*1.2:
#             continue
        sentence_offset_mapping4.append(mapping)
    result = [mapping_to_ids(mapping, text) for mapping in sentence_offset_mapping4]
    # return result
    return result, sentence_pred2, sentence_b_pred2
     


from src.fp1_sol01.utils import get_offset_mapping, is_head
from src.fp1_sol01.utils import get_feat
from src.fp1_sol01.utils import feat_padding


def encode2(text, tokenizer, data, labels_to_ids, labels_to_ids2):
    if 'deberta-v' in tokenizer.name_or_path:
        encoding = tokenizer.encode_plus(text)
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        encoding['offset_mapping'] = get_offset_mapping(text, tokens)
    else:
        encoding = tokenizer.encode_plus(text, return_offsets_mapping=True)
    input_ids      = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    offset_mapping = encoding['offset_mapping']
    
    unmask_length   = sum(attention_mask)
    token_label     = [0] * unmask_length

    token_label_eff = [0] * unmask_length

    if 'discourse_start' in data.columns:
        data = data.sort_values('discourse_start').reset_index(drop=True)
        
        char_label = list(
            zip(data.discourse_start.astype(int), data.discourse_end.astype(int), data.discourse_type, data.discourse_effectiveness)
            )
        

        is_first = True
        for i, mapping in enumerate(offset_mapping):
            while len(char_label) > 0 and mapping[1] > char_label[0][1]:
                char_label.pop(0)                        # 大于区间
                is_first = True
            if len(char_label) == 0:                     # char_label 为空，暂停
                break
            if mapping[1] >= char_label[0][0] and mapping != (0,0):           # 介于区间
                if is_first:
                    if is_head(text[mapping[0]:mapping[1]]):  # 是否合法的开头：剔除空值，和上句的标点符合
                        token_label[i]     = labels_to_ids['B-'+char_label[0][2]]
                        token_label_eff[i] = labels_to_ids2[char_label[0][3]]
                        is_first = False
                else:
                    token_label[i]     = labels_to_ids['I-'+char_label[0][2]]  # 后续同样token
                    token_label_eff[i] = labels_to_ids2[char_label[0][3]]
                                                         # 小于区间不操作
    return input_ids,attention_mask,token_label,token_label_eff,offset_mapping
                              


def feat_padding2(input_ids,attention_mask,token_label,eff_label,batch_length,padding_dict,padding_side):
    random_seed = None
    if padding_side == 'right':
        random_seed = 0
    elif padding_side == 'left':
        random_seed = 1
    else:
        random_seed = np.random.rand()
                
    # 剔除原有padding
    mask_index = attention_mask.nonzero().reshape(-1)
    input_ids      = input_ids.index_select(0,mask_index)
    token_label    = token_label.index_select(0,mask_index)
    attention_mask = attention_mask.index_select(0,mask_index)
    eff_label = eff_label.long().index_select(0,mask_index)
    ids_length = len(input_ids)
    
    # 减去一部分长度
    if ids_length>batch_length:
        if random_seed<=0.33:
            input_ids      = input_ids[:batch_length]
            attention_mask = attention_mask[:batch_length]
            token_label    = token_label[:batch_length]
            eff_label      = eff_label[:batch_length]
        elif random_seed >= 0.66:
            input_ids      = input_ids[-batch_length:]
            attention_mask = attention_mask[-batch_length:]
            token_label    = token_label[-batch_length:]
            eff_label      = eff_label[-batch_length:]
        else:
            sub_length = ids_length - batch_length
            strat_idx  = np.random.randint(sub_length+1)
            input_ids      = input_ids[strat_idx:strat_idx+batch_length]
            attention_mask = attention_mask[strat_idx:strat_idx+batch_length]
            token_label    = token_label[strat_idx:strat_idx+batch_length]
            eff_label      = eff_label[strat_idx:strat_idx+batch_length]

    # 加上一部分长度
    if ids_length<batch_length:
        add_length = batch_length-ids_length
        if random_seed<=0.33:
            input_ids      = F.pad(input_ids,      (0,add_length), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (0,add_length), "constant", padding_dict['attention_mask'])
            token_label    = F.pad(token_label,    (0,add_length), "constant", padding_dict['input_ids'])
            eff_label      = F.pad(eff_label,      (0,add_length), "constant", padding_dict['input_ids'])
        elif random_seed >= 0.66:
            input_ids      = F.pad(input_ids,      (add_length,0), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (add_length,0), "constant", padding_dict['attention_mask'])
            token_label    = F.pad(token_label,    (add_length,0), "constant", padding_dict['input_ids'])
            eff_label      = F.pad(eff_label,    (add_length,0), "constant", padding_dict['input_ids'])
        else:
            add_length1 = np.random.randint(add_length+1)
            add_length2 = add_length - add_length1
            input_ids      = F.pad(input_ids,      (add_length1,add_length2), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (add_length1,add_length2), "constant", padding_dict['attention_mask'])
            token_label    = F.pad(token_label,    (add_length1,add_length2), "constant", padding_dict['input_ids'])
            eff_label      = F.pad(eff_label,    (add_length1,add_length2), "constant", padding_dict['input_ids'])
    
    return input_ids,attention_mask,token_label,eff_label



class Collate2:
    def __init__(self, model_length=None,max_length=None,padding_side='right',padding_dict={}):
        self.model_length = model_length
        self.max_length   = max_length
        self.padding_side = padding_side
        self.padding_dict = padding_dict
    
    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        output["labels"] = [sample["labels"] for sample in batch]
        output["labels_eff"] = [sample["labels_eff"] for sample in batch]

        # calculate max token length of this batch
        batch_length = None
        if self.model_length is not None:
            batch_length = self.model_length
        else:
            batch_length = max([len(ids) for ids in output["input_ids"]])
            if self.max_length is not None:
                batch_length = min(batch_length,self.max_length)
        
        for i in range(len(output["input_ids"])):
            output_fill = feat_padding2(output["input_ids"][i], output["attention_mask"][i], output["labels"][i], output["labels_eff"][i],
                                batch_length, self.padding_dict, padding_side=self.padding_side)
            output["input_ids"][i], output["attention_mask"][i], output["labels"][i], output["labels_eff"][i] = output_fill

            # output_fill = feat_padding(output["input_ids"][i], output["attention_mask"][i], output["labels"][i],
            #                     batch_length, self.padding_dict, padding_side=self.padding_side)
            # output["input_ids"][i], output["attention_mask"][i], output["labels"][i] = output_fill

        # convert to tensors
        output["input_ids"]      = torch.stack(output["input_ids"])
        output["attention_mask"] = torch.stack(output["attention_mask"])
        output["labels"]         = torch.stack(output["labels"])
        output["labels_eff"]     = torch.stack(output["labels_eff"])

        return output 


class TextDataset:
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        id_ = self.data.id[index]
        input_ids = self.data.input_ids[index]
        attention_mask = self.data.attention_mask[index]
        token_label = self.data.token_label[index]
        eff_label = self.data.token_label_eff[index]
                
        item = {
            'id': id_,
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(token_label, dtype=torch.long),
            'labels_eff': torch.tensor(eff_label, dtype=torch.long),
        }
        return item


def model_predict(model, batch, model_length, max_length, padding_dict, padding_side='right', duplicate_cnt=100):
    
    # 调整序列长度
    batch_length = batch["input_ids"].shape[1]
    model_length = min(batch_length,max_length) if model_length is None else model_length
    batch_length = batch_length if batch_length>=model_length else model_length
    new_batch = []
    for i in range(len(batch["input_ids"])):
        new_batch.append(
            feat_padding2(
                batch["input_ids"][i], 
                batch["attention_mask"][i],
                batch["labels"][i],
                batch["labels_eff"][i],
                batch_length=batch_length, 
                padding_dict=padding_dict, 
                padding_side=padding_side
                )
                )
    batch["input_ids"]      = torch.stack([c[0] for c in new_batch])
    batch["attention_mask"] = torch.stack([c[1] for c in new_batch])
    batch["labels"]         = torch.stack([c[2] for c in new_batch])
    batch["labels_eff"]     = torch.stack([c[3] for c in new_batch])

    # 生成循环序列
    ids_length = batch['input_ids'].shape[1]

    loops = int(np.ceil((ids_length-duplicate_cnt)/(model_length-duplicate_cnt)))
    loops = max(loops,1)
    loops_start = [i*(model_length-duplicate_cnt) for i in range(loops)]
    loops_end   = [i*(model_length-duplicate_cnt)+model_length for i in range(loops)]
    if loops > 1:
        loops_start[-1] = ids_length-model_length
        loops_end[-1]   = ids_length

    if padding_side=='left':
        loops_start = [ids_length-idx for idx in loops_end][::-1]
        loops_end   = [idx+model_length for idx in loops_start]
    
    losses = []
    preds = None
    preds2 = None

    for i,(start,end) in enumerate(zip(loops_start,loops_end)):
        device = next(model.parameters()).device
        input_ids      = batch["input_ids"][:,start:end].to(device, dtype = torch.long)
        attention_mask = batch["attention_mask"][:,start:end].to(device, dtype = torch.long)
        labels         = batch["labels"][:,start:end].to(device, dtype = torch.long)
        labels_eff     = batch["labels_eff"][:,start:end].to(device, dtype = torch.long)
        
        model.eval()
        with torch.cuda.amp.autocast():
            # loss, logits = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
            loss, logits, logits_eff = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,labels2=labels_eff)
            loss = loss.mean()

        if i == 0:               # 首个
            preds = logits
            preds2 = logits_eff
        else:
            if i == (loops-1):   # 最后一个
                if i==1:         # 只有两个的时候
                    inter_length = loops_end[0]-loops_start[-1]
                    weight = torch.floor_divide(torch.arange(inter_length), inter_length-1).reshape(1,-1,1).to(device)
                    intersection = preds[:,start:]*(1-weight) + logits[:,:inter_length]*(weight)
                    preds = torch.cat([preds[:,:start],intersection,logits[:,inter_length:]],dim=1)
                    
                    intersection2 = preds2[:,start:]*(1-weight) + logits_eff[:,:inter_length]*(weight)
                    preds2 = torch.cat([preds2[:,:start], intersection2, logits_eff[:,inter_length:]],dim=1)

                else:            # 大于两个的时候
                    preds = torch.cat([preds[:,:start], logits],dim=1)
                    preds2 = torch.cat([preds2[:,:start], logits_eff],dim=1)
            else:                # loop 大于3，中间
                preds = torch.cat([preds, logits[:,duplicate_cnt:]],dim=1)
                preds2 = torch.cat([preds2, logits_eff[:,duplicate_cnt:]],dim=1)
        losses.append(loss)

    # # pred 矩阵转list，并且按照mask过滤
    # pred_list = []
    # for p,m in zip(preds, batch["attention_mask"]):
    #     m_index = m.nonzero().reshape(-1).to(device)
    #     pred_list.append(p.index_select(0,m_index).cpu().numpy().astype('float64'))
        
 
    pred_list = []
    pred_list2 = []
    for p,p2,m in zip(preds, preds2, batch["attention_mask"]):
        m_index = m.nonzero().reshape(-1).to(device)
        pred_list.append(p.index_select(0, m_index).cpu().numpy().astype('float64'))  
        pred_list2.append(p2.index_select(0, m_index).cpu().numpy().astype('float64'))  
        
    return torch.tensor(losses).mean(), pred_list, pred_list2



def after_deal(data_pred, labels_to_ids, segment_param, log, acc=False):
    
    y_pred = []
    accuracy = []
    for i,row in tqdm(data_pred.iterrows(),total=data_pred.shape[0]):
        attention_mask = np.array(row.attention_mask)
        offset_mapping = [mapping for mapping,mask in zip(row.offset_mapping,attention_mask) if mask > 0]
        token_label    = np.array(row.token_label)[attention_mask>0]
#         b_pred = row.pred[:,8:].sum(axis=1)
        for discourse in [
                'Claim', 
                'Evidence', 
                'Position', 
                'Concluding Statement', 
                'Lead', 
                'Counterclaim', 
                'Rebuttal'
        ]:
            b_pred = row.pred[:,labels_to_ids['B-'+discourse]].copy()
            pred   = row.pred[:,labels_to_ids['I-'+discourse]].copy()
            pred   = pred + b_pred
            pred_type = word_decode(pred, b_pred, offset_mapping, row.text,
                                    **segment_param[discourse],
                                   )
            for pred_type_ in pred_type:
                y_pred.append({'id':row['id'],'class':discourse,'predictionstring':pred_type_})
        if acc:
            pred = row.pred.argmax(axis=-1)
            sample_accuracy = accuracy_score(pred, token_label)
            accuracy.append(sample_accuracy)
    log(f'test acc: \t{np.mean(accuracy):.4f}')
    y_pred = pd.DataFrame(y_pred)
    print('y_pred.shape:',y_pred.shape)
    return y_pred

# %%
def word_decode2(pred, b_pred, e_pred, e_pred_cls, offset_mapping, text, **kwargs):

    pred_head = []
    for i,(p,b_p,mapping) in enumerate(zip(pred, b_pred, offset_mapping)):
        if mapping == (0,0):
            continue
        if b_p >= kwargs['begin_proba'] or (abs(p-pred[i-1])>0.1):
            pred_head.append(mapping[0])
    sentence = get_sentence_split(text, pred_head)
    # print(len(sentence))

    sentence_offset_mapping = [[]]
    sentence_pred = [[]]
    sentence_b_pred = [[]]
    sentence_e_pred = [[]]
    sentence_e_pred_class = [[]]
    sentence_idx = [[]]
    i_sentence = 0
    for i,(p, b_p, e_p, e_p_cls, mapping) in enumerate(zip(pred,b_pred,e_pred,e_pred_cls,offset_mapping)):  # token分组
        if mapping == (0,0) or mapping[0]==mapping[1]:
            continue
        if i_sentence == len(sentence):
            print(text)
            print(mapping)
        if mapping[1] <= sentence[i_sentence][1]:
            sentence_offset_mapping[i_sentence].append(mapping)
            sentence_pred[i_sentence].append(p)
            sentence_b_pred[i_sentence].append(b_p)
            sentence_e_pred[i_sentence].append(e_p)
            sentence_e_pred_class[i_sentence].append(e_p_cls)
            sentence_idx[i_sentence].append(i)
        else:
            i_sentence += 1
            sentence_offset_mapping.append([mapping])
            sentence_pred.append([p])
            sentence_b_pred.append([b_p])
            sentence_e_pred.append([e_p])
            sentence_e_pred_class.append([e_p_cls])
            sentence_idx.append([i])
    
    # print('len offset', len(sentence_offset_mapping))
    # print('len pred', len(sentence_pred))

    ### Loop #2 

    sentence_offset_mapping2 = []
    sentence_pred2 = []
    sentence_b_pred2 = []
    sentence_e_pred2 = []
    sentence_e_pred2_class = []
    sentence_idx2 = []       
    for t, mapping_, pred_, b_pred_, e_pred_, e_pred_cls_, idx_ in zip(
        sentence,sentence_offset_mapping,
        sentence_pred,sentence_b_pred,sentence_e_pred,sentence_e_pred_class, 
        sentence_idx
        ):
        # print(np.round(pred_,2),'\t',text[t[0]:t[1]])
        if np.mean(pred_) > kwargs['min_proba'][0]:
            sentence_offset_mapping2.append([mapping_[0][0], mapping_[-1][1]])
            sentence_pred2.append(pred_)
            sentence_b_pred2.append(b_pred_)
            sentence_e_pred2.append(e_pred_)
            sentence_e_pred2_class.append(e_pred_cls_)
            sentence_idx2.append(idx_)

    # print('#2')
    # print('len offset 2', len(sentence_offset_mapping2))
    # print('len pred', len(sentence_pred2))

    ### Loop #3 

    sentence_offset_mapping3 = []
    sentence_idx3 = []
    for i, mapping_ in enumerate(sentence_offset_mapping2):
        sep_start = sentence_idx2[i-1][-1]+1
        sep_end = sentence_idx2[i][0]
        sep_length = sep_end-sep_start
        if i == 0:                    
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        # elif text[mapping_[0]-1] == '\n' and sentence_b_pred2[i][0]>kwargs['line_prob']:
        #     sentence_offset_mapping3.append(mapping_)
        #     sentence_idx3.append(sentence_idx2[i])
        elif sentence_b_pred2[i][0] > kwargs['begin_proba']:             
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        elif sep_length >= kwargs['min_sep']:                       
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        elif sep_length/kwargs['min_sep'] - np.mean(pred[sep_start:sep_end])/kwargs['min_proba'][1] > 0:  
            sentence_offset_mapping3.append(mapping_)
            sentence_idx3.append(sentence_idx2[i])
        else:
            sentence_offset_mapping3[-1][1] = mapping_[1]
            sentence_idx3[-1].extend(sentence_idx2[i])
    
    # print('#3')
    # print('len offset 3', len(sentence_offset_mapping3))
    # print('len idx 3', len(sentence_idx3))

    sentence_pred4 = []
    sentence_pred4_e = []
    sentence_pred4_e_class = []

    sentence_offset_mapping4 = []
    for i, mapping in enumerate(sentence_offset_mapping3):
        word_length = len(text[mapping[0]: mapping[1]].split())
        if word_length <= kwargs['min_length']:
            continue
        # if word_length >= kwargs['min_length'][1]: 
        #     continue
        # if sum(pred[sentence_idx3[i][0] : sentence_idx3[i][-1]+1]) <= kwargs['min_proba'][0]*kwargs['min_length'][1]*1.2:
        #     continue
        sentence_offset_mapping4.append(mapping)
        sentence_pred4.append(sentence_pred2[i]) ## TODO check / mean(pred[sentence_idx3[i][0] : sentence_idx3[i][-1]+1])
        sentence_pred4_e.append(sentence_e_pred2[i])             # / mean(e_pred[sentence_idx3[i][0] : sentence_idx3[i][-1]+1])
        sentence_pred4_e_class.append(sentence_e_pred2_class[i]) # / mean(e_pred_cls[sentence_idx3[i][0] : sentence_idx3[i][-1]+1]) 
    
    # print('#4')
    # print('len offset 4', len(sentence_offset_mapping4))
    # print('len pred 4', len(sentence_pred4))
    
    result = [mapping_to_ids(mapping, text) for mapping in sentence_offset_mapping4]
    return result, sentence_pred4, sentence_pred4_e, sentence_pred4_e_class

    

# from src.fp1_sol01.utils import word_decode

## after_deal_helper --> pp_preds
def pp_preds(train_ids_sub, args, segment_param):

    labels_to_ids = args.labels_to_ids
    labels_to_ids2 = args.labels_to_ids2
    # segment_param = args.segment_param
    
    y_pred = []
    for i, row in train_ids_sub.iterrows():
        attention_mask = np.array(row.attention_mask)
        offset_mapping = [mapping for mapping,mask in zip(row.offset_mapping, attention_mask) if mask > 0]
        # token_label  = np.array(row.token_label)[attention_mask>0]
        # b_pred = row.pred[:,8:].sum(axis=1)
        for discourse in [
                'Claim', 
                'Evidence', 
                'Position', 
                'Concluding Statement', 
                'Lead', 
                'Counterclaim', 
                'Rebuttal'
        ]:
            # if row.pred.shape[1]!=9:
            b_pred = row.pred[:,labels_to_ids['B-'+discourse]].copy()
            pred   = row.pred[:,labels_to_ids['I-'+discourse]].copy()
            pred   = pred + b_pred

            # eff_pred = row.pred_eff[:,labels_to_ids2['B-'+discourse]].copy()
            e_pred = row.pred_eff.max(1)
            e_pred_class = row.pred_eff.argmax(1)
      
            # pred_type, _, _ = word_decode(pred, b_pred, offset_mapping, row.text,
            #                         **segment_param[discourse],
            #                        )


            pred_type, proba_disc, proba_eff, class_eff = word_decode2(pred, b_pred, e_pred, e_pred_class, offset_mapping, row.text,
                                    **segment_param[discourse],
                                   )
            
            for pred_type_, p1, p2, clseff in zip(pred_type, proba_disc, proba_eff, class_eff):
                y_pred.append({
                    'id':row['id'],
                    'class':discourse,
                    'predictionstring':pred_type_,
                    ####
                    'scores': np.mean(p1), # proba_disc
                    ######
                    'class_eff':clseff,
                    'scores_eff': np.mean(p2),
                    })
    
    y_pred = pd.DataFrame(y_pred)    
    # y_pred.columns = ['id', 'class', 'predictionstring', 'proba_disc', 'class_eff', 'proba_eff']
    ### use same cols as V2:  class, predictionstring, scores, scores_eff, class_eff
    
    y_pred['class_eff'] = y_pred['class_eff'].apply(lambda x: pd.Series(x).mode()[0])
    y_pred['discourse_effectiveness'] = y_pred['class_eff'].map(args.ids_to_labels2)
    
    return y_pred

# %%
class TextModel(nn.Module):
    def __init__(self,model_name=None,num_labels=1):
        super(TextModel,self).__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name) # 768
        self.drop_out = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.output = nn.Linear(config.hidden_size,num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        emb = self.model(input_ids,attention_mask)[0]

        preds1 = self.output(self.dropout1(emb))
        preds2 = self.output(self.dropout2(emb))
        preds3 = self.output(self.dropout3(emb))
        preds4 = self.output(self.dropout4(emb))
        preds5 = self.output(self.dropout5(emb))
        preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5
        
        logits = torch.softmax(preds, dim=-1)
        if labels is not None:
            loss = self.get_loss(preds,labels,attention_mask)
            return loss,logits
        else:
            return logits
    
    
    def _compute_kl_loss(self, p, q, pad_mask=None):
    
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

        loss = (p_loss + q_loss) / 2
        return loss

    def get_loss(self, outputs, targets, attention_mask):
        loss_fct = nn.CrossEntropyLoss()

        active_loss = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss1 = loss_fct(active_logits, true_labels)
        
        active_loss2 = ((targets[:,:-1] == targets[:,1:]) & (attention_mask[:,:1] == 1)).view(-1)
        active_logits_a = outputs[:,:-1].reshape(-1, outputs.shape[-1])
        active_logits_b = outputs[:,1:].reshape(-1, outputs.shape[-1])
        idxs2 = np.where(active_loss2.cpu().numpy() == 1)[0]
        active_logits_a = active_logits_a[idxs2]
        active_logits_b = active_logits_b[idxs2]
        
        loss2 = self._compute_kl_loss(active_logits_a,active_logits_b)
        
        loss = loss1+loss2*10
        return loss
    


# %%
class RaterModel(nn.Module):
    def __init__(self, model_name=None, num_labels=1, num_labels2=3):
        super().__init__()
        self.num_labels = num_labels
        self.num_labels_eff = num_labels2


        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name) # 768
        
        self.drop_out = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.output = nn.Linear(config.hidden_size, self.num_labels)
        self.head_eff = nn.Linear(config.hidden_size, self.num_labels_eff)
        
        self.loss_fct = nn.CrossEntropyLoss()
        self.loss_fct2 = nn.CrossEntropyLoss()
        self.loss_bce = nn.BCEWithLogitsLoss()

        if 'deberta-v2-xxlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:24].requires_grad_(False) # 冻结24/48
        if 'deberta-v2-xlarge' in model_name:
            self.model.embeddings.requires_grad_(False)
            self.model.encoder.layer[:12].requires_grad_(False) # 冻结12/24

    def forward(self, input_ids, attention_mask, labels=None, labels2=None):
        if 'gpt' in self.model.name_or_path:
            emb = self.model(input_ids)[0]
        else:
            emb = self.model(input_ids, attention_mask)[0]

        preds1 = self.output(self.dropout1(emb))
        preds2 = self.output(self.dropout2(emb))
        preds3 = self.output(self.dropout3(emb))
        preds4 = self.output(self.dropout4(emb))
        preds5 = self.output(self.dropout5(emb))
        preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5

        ### effectiveness 
        preds_e1 = self.head_eff(self.dropout1(emb))
        preds_e2 = self.head_eff(self.dropout2(emb))
        preds_e3 = self.head_eff(self.dropout3(emb))
        preds_e4 = self.head_eff(self.dropout4(emb))
        preds_e5 = self.head_eff(self.dropout5(emb))
        preds_eff = (preds_e1 + preds_e2 + preds_e3 + preds_e4 + preds_e5) / 5

        # logits1 = torch.softmax(preds[:,:,:8], dim=-1)
        # logits2 = torch.sigmoid(preds[:,:,-1:])
        # logits = torch.cat([logits1,logits2],dim=-1)

        soft_proba = torch.softmax(preds, dim=-1)
        soft_proba_eff = torch.softmax(preds_eff, dim=-1)

        if labels is not None:
            loss1 = self.get_loss(preds,labels,attention_mask)
            # loss2 = self.loss_bce(preds_eff, labels2)
            loss2 = self.loss_fct2(preds_eff.reshape(-1, self.num_labels_eff), labels2.reshape(-1))

            loss = loss1+loss2
            return loss, soft_proba, soft_proba_eff
        else:
            return soft_proba, soft_proba_eff


    def get_loss(self, outputs, targets, attention_mask):
        loss_fct = nn.CrossEntropyLoss()

        active_loss = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = loss_fct(active_logits, true_labels)
        return loss   


    def get_loss_aux_token(self, outputs, targets, attention_mask):
        loss_fct = nn.CrossEntropyLoss()
        loss_bce = nn.BCEWithLogitsLoss()
        
        active = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active.cpu().numpy() == 1)[0]
        
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)
        calss_targets = (true_labels - (true_labels>=8)*8).long()
        head_targets = (true_labels>=8).float()

        loss1 = loss_fct(active_logits[:,:8], calss_targets)
        loss2 = loss_bce(active_logits[:,8],  head_targets)

        loss = loss1+loss2
        return loss     

# %%


### ++
def get_feat_helper2(args, tokenizer, df, train_ids, text_col='full_text'):

    grp_text = df.groupby('id')[text_col].apply(lambda x: x.values[0]).reset_index()    
    training_samples = []
    for idx in tqdm(train_ids):
        # text = df[df['id']==idx][text_col].values[0].rstrip()
        text = grp_text[grp_text['id']==idx][text_col].values[0].rstrip()
        input_ids, attention_mask, token_label, token_label_eff, offset_mapping = encode2(
            text, tokenizer, df[df['id']==idx], 
            args.labels_to_ids,
            args.labels_to_ids2
            )
        training_samples.append({
            'id': idx,
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_label': token_label,
            'token_label_eff': token_label_eff,
            'offset_mapping': offset_mapping
        })
    return training_samples


def get_feat(df, tokenizer, args, data_key, text_col='full_text'):
    num_jobs = 8
    data = []
    train_ids = df["id"].unique()

    train_ids_splits = np.array_split(train_ids, num_jobs)
    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(get_feat_helper2)(args, tokenizer, df, idx, text_col) for idx in train_ids_splits
    )
    for result in results:
        data.extend(result)

    data = pd.DataFrame(sorted(data, key=lambda x: len(x['input_ids'])))
    return data

# %% [markdown]
# # Config 

save_name = "test_preds_fold1"
save_dir = "exp21"

model_name1 = '../input/longformerlarge4096/longformer-large-4096/'
model_length1 = None
# path_list1 = [
#     '../input/longformer-large-v9/longformer-large-4096_aistudio_15class_padding_right_adv0.0005_max_lr_2e-05_max_length_1500_fold0.pt',
#     '../input/longformer-large-v9/longformer-large-4096_aistudio_15class_padding_right_adv0.0005_max_lr_2e-05_max_length_1500_fold1.pt',
#     '../input/longformer-large-v9/longformer-large-4096_aistudio_15class_padding_right_adv0.0005_max_lr_2e-05_max_length_1500_fold2.pt',
#     '../input/longformer-large-v9/longformer-large-4096_aistudio_15class_padding_right_adv0.0005_max_lr_2e-05_max_length_1500_fold3.pt',
#     '../input/longformer-large-v9/longformer-large-4096_aistudio_15class_padding_right_adv0.0005_max_lr_2e-05_max_length_1500_fold4.pt',
# ]

## exp21 checkpoints
path_list1 = [
    # '/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp21/longformer-base-4096_max_length_1536_fold0_ep4.pt'
    # '/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp21/longformer-base-4096-1536-fold0.pt',
    '/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp21/longformer-base-4096-1536-fold1.pt',
    # '/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp21/longformer-base-4096-1536-fold2.pt'
]


## exp23 checkpoints

# %%
class CFG:
    exp_name = 'exp21' 
    task = "ner"
    n_folds = 5  
    trn_folds = [0] # [0,1,2,3,4]
    model_name = "allenai/longformer-base-4096"

    max_length = 1536 #768  
    stride = 128
    min_tokens = 6
    batch_size = 16 #32 #4       # 16
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
    output = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/"
    NUM_CORES = 8 #cfg["NUM_CORES"]

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
DIR = Path("/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/")
OUTPUT = Path(CFG.output + CFG.exp_name)
OUTPUT.mkdir(exist_ok=True, parents=True)




# %%
args,log = get_args()
seed_everything(args.seed)

# %%
discourse_type = ['Claim','Evidence', 'Position','Concluding Statement','Lead','Counterclaim','Rebuttal']

b_discourse_type = ['I-'+i for i in discourse_type]
i_discourse_type = ['B-'+i for i in discourse_type]

args.labels_to_ids = {k:v for v,k in enumerate(['O']+b_discourse_type+i_discourse_type)}
args.ids_to_labels = {k:v for v,k in args.labels_to_ids.items()}

# %%
args.labels_to_ids2 = {
    'O': 0,
    'Non-Effective': 1,
    'Effective': 2,
}
args.ids_to_labels2 = {k:v for v,k in args.labels_to_ids2.items()}

# %%
args.segment_param = {
    "Lead":                 {'min_proba':[0.47,0.41],'begin_proba':1.00,'min_sep':40,'min_length': 5},
    "Position":             {'min_proba':[0.45,0.40],'begin_proba':0.90,'min_sep':21,'min_length': 3},
    "Evidence":             {'min_proba':[0.50,0.40],'begin_proba':0.56,'min_sep': 2,'min_length':21},
    "Claim":                {'min_proba':[0.40,0.30],'begin_proba':0.30,'min_sep':10,'min_length': 1},
    "Concluding Statement": {'min_proba':[0.58,0.25],'begin_proba':0.93,'min_sep':50,'min_length': 5},
    "Counterclaim":         {'min_proba':[0.45,0.25],'begin_proba':0.70,'min_sep':35,'min_length': 6},
    "Rebuttal":             {'min_proba':[0.37,0.34],'begin_proba':0.70,'min_sep':45,'min_length': 5},
    }

# args.segment_param 


sample_sub = pd.read_csv(args.data_path+'sample_submission.csv')
log('sample_submission.shape:', sample_sub.shape,'\t')

tokenizer = AutoTokenizer.from_pretrained(args.model_name)


# test_feat_ori = get_feat_ori(test_df_ori, tokenizer,args, 'test_feat'+args.key_string)
# log("test_feat.shap: {}".format(test_feat_ori.shape),'\t')

 
## Test data


test_df = pd.read_csv(DIR / 'test.csv')
# test_df = test_df.rename(columns={'essay_id_comp':'id'})
test_df['id'] = test_df['essay_id_comp']
test_df['class'] = -100
test_df['predictionstring'] = -100

test_ids = test_df['id'].unique().tolist()
# len(test_ids)

if DEBUG: test_df = test_df[test_df.id.isin(test_ids[:10])]


test_feat = get_feat(test_df, tokenizer, args, 'test_feat')
print(test_feat.shape)
# test_feat.head()

test_ds = TextDataset(test_feat)

print('test_ds keys:', test_ds[0].keys())


args.test_params = {
    'batch_size': args.valid_batch_size,
    'shuffle': False, 
    'num_workers': 2,
    'pin_memory':True,
    'collate_fn': Collate2(padding_side='right',padding_dict=args.padding_dict)
    }

test_loader = DataLoader(test_ds, **args.test_params)

###--------------------------
### Inference
###--------------------------


### CREATE MODEL
# model = TextModel(model_name1, num_labels=len(args.labels_to_ids))
# model = torch.nn.DataParallel(model)

model = RaterModel(args.model_name, num_labels=len(args.labels_to_ids), num_labels2=len(args.labels_to_ids2))
model.to(args.device)


avg_preds = None
avg_preds_eff = None

for model_path in path_list1:
    print('Load weights:', model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    preds = []
    preds_eff = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # if args.test_padding_side in ['right','left']:
            #     loss, te_logits = model_predict(model, batch, model_length=model_length1,
            #                                     padding_dict=args.padding_dict,padding_side=args.test_padding_side)
            # elif args.test_padding_side == 'double':
            #     loss1, te_logits1 = model_predict(model, batch, model_length=args.model_length,
            #                                     padding_dict=args.padding_dict,padding_side='right')
            #     loss2, te_logits2 = model_predict(model, batch, model_length=args.model_length,
            #                                     padding_dict=args.padding_dict,padding_side='left')
            #     te_logits = [(l1+l2)/2 for l1,l2 in zip(te_logits1,te_logits2)]
            ##################
            if args.test_padding_side in ['right','left']:
                loss, te_logits, te_logits_eff = model_predict(
                    model, batch, model_length=model_length1, max_length=args.max_length,
                    padding_dict=args.padding_dict, padding_side=args.test_padding_side
                    )
            elif args.test_padding_side == 'double':
                loss1, te_logits1, te_logits_eff1 = model_predict(
                    model, batch, model_length=args.model_length, max_length=args.max_length,
                    padding_dict=args.padding_dict,padding_side='right'
                    )
                loss2, te_logits2, te_logits_eff2 = model_predict(
                    model, batch, model_length=args.model_length, max_length=args.max_length,
                    padding_dict=args.padding_dict,padding_side='left'
                    )

                te_logits     = [(l1+l2)/2 for l1,l2 in zip(te_logits1, te_logits2)]
                te_logits_eff = [(l1+l2)/2 for l1,l2 in zip(te_logits_eff1, te_logits_eff2)]

            
            preds.extend(te_logits)
            preds_eff.extend(te_logits_eff)
    if avg_preds is None:
        avg_preds     = [b for b in preds]
        avg_preds_eff = [b for b in preds_eff]
    else:
        avg_preds     = [a+b for a,b in zip(avg_preds, preds)]
        avg_preds_eff = [a+b for a,b in zip(avg_preds_eff, preds_eff)]
        
avg_preds     = [i/len(path_list1) for i in avg_preds]
avg_preds_eff = [i/len(path_list1) for i in avg_preds_eff]

test_feat['pred']  = avg_preds
test_feat['pred_eff'] = avg_preds_eff

segment_param = args.segment_param

## save as pkl 

# with open(OUTPUT / "test_preds_fold0.pkl", 'wb') as fin:
#     pickle.dump(test_feat, fin)

with open(OUTPUT / f"{save_name}.pkl", 'wb') as fin:
    pickle.dump(test_feat, fin)

### Post process predictions
    
sub = pp_preds(test_feat, args, args.segment_param)  

print('sub shape:', sub.shape())
print('num ids:', sub.id.nunique())


# sub.to_csv(f'{OUTPUT}/test_preds_f0.csv', index=False)
sub.to_csv(f'{OUTPUT}/{save_name}.csv', index=False)

print(sub.head())
print(sub['class_eff'].value_counts())
print(sub['discourse_effectiveness'].value_counts())

###--------------------------------------------
### submission 
###--------------------------------------------


discourse_type_to_int = {
    'Lead': 0,
    'Position': 1,
    'Claim': 2,
    'Evidence': 3, 
    'Counterclaim': 4, 
    'Rebuttal': 5, 
    'Concluding Statement': 6, 
}

def generate_submission(df_sub, save_name, out_dir=OUTPUT, verbose=False):

    # pred_df = pd.read_csv(f'{out_dir}/{fname}.csv') 
    pred_df = df_sub.copy()
    print(pred_df.shape)
    print(pred_df.id.nunique())
    # display(pred_df)

    ## convert effective scores to RATER format 
    pred_df['score_discourse_effectiveness_0'] = -100
    pred_df['score_discourse_effectiveness_1'] = -100

    for i, row in pred_df.iterrows():
        if row['discourse_effectiveness'] == 'Effective': #==1
            pred_df.loc[i, 'score_discourse_effectiveness_1'] = row['scores_eff']
            pred_df.loc[i, 'score_discourse_effectiveness_0'] = 1 - row['scores_eff']
        else:
            pred_df.loc[i, 'score_discourse_effectiveness_0'] = row['scores_eff']
            pred_df.loc[i, 'score_discourse_effectiveness_1'] = 1 - row['scores_eff']


    sample_sub = pd.read_csv(DIR / 'sample_submission.csv')

    sub = pred_df.copy()
    sub = sub.rename(columns={'id': 'essay_id_comp'})
    sub["discourse_type"] = sub["class"].map(discourse_type_to_int)

    sub = sub[sample_sub.columns]
    if sub.essay_id_comp.nunique() == 10402:
        sub.to_csv(f'{out_dir}/{save_name}.csv', index=False)
    else:
        print("check ids", sub.essay_id_comp.nunique())

    print(sub.head())
    print(sub.essay_id_comp.nunique() == 10402)
    
    ###
    if verbose:
        print(pred_df['class'].value_counts())
        print()
        print(pred_df['class_eff'].value_counts())
        print(pred_df['class_eff'].value_counts(normalize=True))
        print(sub.dtypes)

        # import seaborn as sns 
        # import matplotlib.pyplot as plt

        # sns.histplot(sub.score_discourse_effectiveness_0)
        # sns.histplot(sub.score_discourse_effectiveness_1)
        # plt.show()

    return sub


# sub = generate_submission(sub, 'sub_f0')
sub = generate_submission(sub, f'sub_{save_name}')

# %%
### Check missing ids 
missing_ids = list(set(test_ids) - set(sub.essay_id_comp))
print(missing_ids)

if len(missing_ids)>0:

    dummy_sub = pd.read_csv('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/submissions/sub_wbf_exp18_exp22_6ckp.csv')
    dummy_sub = dummy_sub[dummy_sub.essay_id_comp.isin(missing_ids)]
    dummy_sub = dummy_sub.reset_index(drop=True)
    # display(dummy_sub)

    sub = pd.concat([sub, dummy_sub], ignore_index=True)
    print(sub.essay_id_comp.nunique() == 10402)

if sub.essay_id_comp.nunique() == 10402:
    sub.to_csv(OUTPUT / f'sub_{save_name}.csv', index=False)


# assert sub.essay_id_comp.nunique() == len(test_ids), 'ERROR: Check ids..'  


# %%
# ### temp hack - add dummy preds for missing ids --> TODO debug 
# if len(missing_ids)>0:
#     dummy_df = []
#     for i, mid in enumerate(missing_ids):
#         print(i, mid)
#         tmp = {
#             'essay_id_comp': mid,
#             'predictionstring': ['0 1 2 3 4 5 6 7 8 9'],
#             'score_discourse_effectiveness_0': 0.95,
#             'score_discourse_effectiveness_1': 0.05,
#             'discourse_type': 0,
#         }
#         dummy_df.append(pd.DataFrame(tmp))
#     dummy_df = pd.concat(dummy_df, ignore_index=True) 

# if len(missing_ids)>0:
#     sub = pd.concat([sub[sample_sub.columns], dummy_df], axis=0)
#     sub = sub.reset_index(drop=True)
# else:
#     sub = sub.reset_index(drop=True)

# %%
# sub = sub.rename(columns={'id': 'essay_id_comp'})
# sub['discourse_type'] = sub['class'].map(discourse_type_to_int) 
# # ### effectiveness 
# # sub['score_discourse_effectiveness_0'] = -100
# # sub['score_discourse_effectiveness_1'] = -100
# ### SUBMIT 
# sub.to_csv(OUTPUT / 'sub_f0.csv')
