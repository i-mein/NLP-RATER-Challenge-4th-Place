
import argparse  # A library for preparing a command-line interface
import pickle  # A library for data serialization and storage
import re  # A library for processing regular expressions
import os.path as osp  # A library for path navigation in Python

import pandas as pd  # Dataframe handling library
import numpy as np  # A scientific and numeric computing library
from tqdm import tqdm   # A library to display progress bars

import torch  # The PyTorch Library for machine learning model development
import torch.nn as nn  # PyTorch sub-components
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align   # Computer vision-specific components
from transformers import AutoModelForTokenClassification, AutoTokenizer  # HuggingFace Transformers library


def parse_args():  # Command line interface to read string input
    parser = argparse.ArgumentParser()
    
    parser.add_argument('exp_name')
    parser.add_argument('model')
    parser.add_argument('checkpoint')
    parser.add_argument('-swa_checkpoint', default=None)
    parser.add_argument('-outfile')
    parser.add_argument('-test_th', default=0.5)
    return parser.parse_args()


def save_pkl(data, path):
    '''
    A function to save an object using Pickle
    
    :param data: The data object to save to a file path
    :param path: The path at which to save the target data
    
    :returns: A saved data file at the specified path
    '''
    with open(path, 'wb') as f:  # Open the data file at the specified file
        pickle.dump(data, f)   # Dump all the data into the file


def to_gpu(data):
    '''
    A helper function to move PyTorch objects, e.g., Tensors, to GPU memory
    
    :param data: The data object to transfer to GPU
    '''
    if isinstance(data, dict):  # If the object is a dictionary
        return {k: to_gpu(v) for k, v in data.items()}  # Recursively call the function on all its values
    elif isinstance(data, list): # If the object is a list
        return [to_gpu(v) for v in data]   # Move all its elements to GPU with a recursive call
    elif isinstance(data, torch.Tensor):  # Recursion base case (Torch tensor)
        return data.cuda()  # Call the cuda() method
    else:
        return data  # Otherwise, no compatible datatype is provided, return the object itself without transfer


def to_np(t):
    '''
    Convert an input object t (a torch Tensor) to a NumPy array/tensor
    '''
    if isinstance(t, torch.Tensor):  # First validate that this object is a Torch tensor
        return t.data.cpu().numpy()  # If so, move it to CPU and convert to NumPy formt
    else:
        return t  # Else do nothing


def seg_roi_align(feat, bboxes):  
    '''
    A function to pool token-level class predictions (3 + 7 discourse elements) produced by the transformer models to word-level representations.
    
    :param feat: The output features produced by the transformer. Shape [Batch (1), seq_len (#tokens), nb_classes] (In this case num_classes is 7+3)
    :param boxes: The boxes produced for token to word alignment.
    
    :returns: Word-level class predictions (+ the 3 auxiliary elements) (Shape [#words, 10])
    '''
    feat = feat.permute(0, 2, 1).unsqueeze(2) # The resulting shape is now [Batch, nb_classes, 1, seq_len]
    '''
    roi_align is used in computer vision, and accepts a tensor of shape [Batch_size, channel, height_width].
    Hence, in this code snippet, the token sequence is considered an image of shape [1, #tokens], seen across 10 feature channels (the 10 transformer labels)
    As in the rest of the code, the batch size adopted in this notebook is 1.
    Roi_align considers all tokens in the box and computes their average features (i.e., average pooling) to return word-level predictions.
    The boxes are provided as a (single-element) list of (x1, y1, x2, y2) coordinates.
    '''
    output = roi_align(feat, [bboxes], 1, aligned=True) # roi_align returns a tensor of shape Tensor[#words, nb_classes, 1, 1]
    return output.squeeze(-1).squeeze(-1)  # Resulting in a tensor of shape [#words, nb_classes]


class TextSpanDetectorPL_FP1(pl.LightningModule):
    def __init__(self,
            model_name,
            num_classes=7,
            num_classes2=2, #3, 
            dynamic_positive=True, #False,
            local_files_only=True,
            init_cfg=None, 
            cfg=None,
            train_df = None,
            valid_df = None,
            predict_df = None,
            # pretrained_path="",
            pretrained_path="/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/checkpoints/", ## None
            # with_cp=False,
            use_sigmoid=False,
            test_score_thr = 0.2, # 0.4,
            nms_thr=0.5,
            warmup=0.0,
            vb=False
        ):
        super().__init__() #init_cfg

        self.cfg = cfg
        self.num_classes  = num_classes+1+2 ## 7 disc types + obj + reg
        self.num_classes2 = num_classes2
        self.dynamic_positive = dynamic_positive
        self.warmup_ratio = warmup #0.1
        self.train_df = train_df
        self.valid_df = valid_df
        self.predict_df = predict_df
        # self.loss_weights = [1.0, 4.0]    #[1.0, 1.0]     ## run 1
        self.loss_weights = [1.0, 1.0]  ## run 2 

        self.use_sigmoid = True if num_classes2==1 else use_sigmoid
        self.test_score_thr=test_score_thr #0.5
        self.nms_thr=nms_thr
        self.skip_validation = -1
        self.val_outs = []
        self.best_score = 0.0
        self.vb=vb

        if pretrained_path != "":

            if "deberta-large" in model_name:
                sd = torch.load(pretrained_path + "tasckj/deberta_large_fold0.pth")['state_dict']
            elif "deberta-base" in model_name:
                sd = torch.load(pretrained_path + "tasckj/deberta_base_fold0.pth")['state_dict']
            elif "longformer" in model_name:
                sd = torch.load(pretrained_path + "tasckj/longformer_large_fold0.pth")['state_dict']
            else:
                print("Check weights path..")
            new_sd = OrderedDict()
            for k, v in sd.items():
                kk = k.replace("model.", "")
                new_sd[kk] = v 

            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=1 + 2 + num_classes,
                local_files_only=local_files_only)
            self.model.load_state_dict(new_sd)

            # self.backbone = AutoModelForTokenClassification.from_pretrained(model_name,
            #     num_labels=1 + 2 + num_classes,
            #     local_files_only=local_files_only)

            config_hf = AutoConfig.from_pretrained(model_name)
            

        else:
            config_hf = AutoConfig.from_pretrained(model_name)
            # config_hf.update({"max_position_embeddings": self.cfg.max_length})

            if 'deberta-v' in model_name:      
                self.backbone = DebertaV2Model(config_hf)
            elif 'deberta-' in model_name:   
                self.backbone = DebertaModel(config_hf) 
            else:
                self.backbone = AutoModel.from_pretrained(model_name) # 768

        if self.cfg.gradient_checkpointing: #with_cp:
            self.model.gradient_checkpointing_enable()
            # self.backbone.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(config_hf.hidden_size, self.num_classes)
        self.head_eff   = nn.Linear(config_hf.hidden_size, self.num_classes2)

        ## init bias
        # self.classifier.bias.data[0].fill_(bias_init_with_prob(0.02))
        # self.classifier.bias.data[3:].fill_(bias_init_with_prob(1 / self.num_classes))

        self.model.classifier.bias.data[0].fill_(bias_init_with_prob(0.02))
        self.model.classifier.bias.data[3:].fill_(bias_init_with_prob(1 / self.num_classes))

        self.head_eff.bias.data[0].fill_(bias_init_with_prob(0.5))

        # self.loss_fct = nn.CrossEntropyLoss()
        # self.loss_fct2 = nn.CrossEntropyLoss()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.tokenizer = cfg.tokenizer


    def forward(self, batch):
        batch_size = batch['input_ids'].size(0)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'
        
        ## obj_pred, reg_pred, cls_pred, logits_eff_word, logits_raw_types, logits_raw_eff
        obj_pred, reg_pred, cls_pred, logits_eff_word, _, _ = self.forward_logits(batch)
        
        obj_target, reg_target, cls_target, cls_eff_target, pos_loc = self.build_target(
            batch['gt_spans'], obj_pred, reg_pred, cls_pred, logits_eff_word)

        obj_loss, reg_loss, cls_loss, cls_eff_loss = self.get_losses(
            obj_pred,   reg_pred,   cls_pred,   logits_eff_word, 
            obj_target, reg_target, cls_target, cls_eff_target, 
            pos_loc
            )
        
        loss1 = obj_loss + reg_loss + cls_loss
        loss2 = cls_eff_loss

        # #### add NER loss (effectiveness)
        # # loss2 = self.loss_fct2(eff_pred.reshape(-1, self.num_classes2), batch['labels_eff'].reshape(-1))
        # loss2 = self.get_ner_loss(logits_eff_raw, batch['labels_eff'], batch['attention_mask']) #logits_eff_word TODO try
        # loss2 = self.get_ner_loss(logits_eff_word, batch['labels_eff'], batch['attention_mask'])  

        # ### total loss 
        #loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2
        # loss = obj_loss + reg_loss + cls_loss + cls_eff_loss
        loss = obj_loss + reg_loss + cls_loss + self.loss_weights[1]*cls_eff_loss

        out = {}
        out['loss'] = loss
        out['loss1'] = loss1
        out['loss2'] = loss2
        # out['logits'] = logits_raw
        # out['logits_eff'] = logits_eff_raw
        out['logits_eff_word'] = logits_eff_word
        return out
    

    def forward_logits(self, data):
        ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
        bs = data['input_ids'].size(0)
        assert bs == 1, f'Only batch_size=1 supported, got batch_size={bs}.'
        
        ##### original 
        # outputs = self.model(input_ids=data['input_ids'],attention_mask=data['attention_mask'])
        # logits = outputs['logits']
        
        #### modify for AutoModel
        # outputs = self.model(data['input_ids'], data['attention_mask']) 
        outputs = self.model.deberta(data['input_ids'], data['attention_mask']) 
        # outputs = self.backbone(data['input_ids'], data['attention_mask']) 
        # pooled_out = outputs[0]
        x = outputs.last_hidden_state
        # x = self.dropout(x)

        # logits_type_raw = self.classifier(x)
        logits_type_raw = self.model.classifier(x)
         
        logits_type_word = aggregate_tokens_to_words(logits_type_raw, data['word_boxes'])
        assert logits_type_word.size(0) == data['text'].split().__len__()

        logits_eff_raw = self.head_eff(x) # effectiveness
        
        logits_eff_word = aggregate_tokens_to_words(logits_eff_raw, data['word_boxes'])
        assert logits_eff_word.size(0) == data['text'].split().__len__()

        obj_pred = logits_type_word[..., 0]
        reg_pred = logits_type_word[..., 1:3]
        cls_pred = logits_type_word[..., 3:]

        return dict(
            obj_pred=obj_pred, 
            reg_pred=reg_pred, 
            cls_pred=cls_pred, 
            logits_type_raw=logits_type_raw, 
            logits_eff_raw=logits_eff_raw, 
            logits_eff_word=logits_eff_word,
            hidden_state=x,
            )
    
    ## Loss functions 
    def get_losses(
            self, obj_pred, reg_pred, cls_pred, logits_eff_word, 
            obj_target, reg_target, cls_target, cls_eff_target, pos_loc, logits_eff_disc, cls_eff_target_disc):
        
        num_total_samples = pos_loc.numel()
        assert num_total_samples > 0
        
        reg_pred   = reg_pred[pos_loc].exp()
        reg_target = reg_target[pos_loc]
        
        px1 = pos_loc - reg_pred[:, 0]
        px2 = pos_loc + reg_pred[:, 1]
        gx1 = reg_target[:, 0]
        gx2 = reg_target[:, 1]
        ix1 = torch.max(px1, gx1)
        ix2 = torch.min(px2, gx2)
        ux1 = torch.min(px1, gx1)
        ux2 = torch.max(px2, gx2)
        inter = (ix2 - ix1).clamp(min=0)
        union = (ux2 - ux1).clamp(min=0) + 1e-12
        iou = inter / union

        # reg_loss = -iou.log().sum() / num_total_samples
        reg_loss = torch.sum(-torch.log(iou + 1e-12)) / num_total_samples
        # reg_loss = torch.nansum(-torch.log(iou + 1e-12)) / num_total_samples
        
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_pred[pos_loc],
            cls_target[pos_loc] * iou.detach().reshape(-1, 1),
            reduction='sum') / num_total_samples
        
        obj_loss = F.binary_cross_entropy_with_logits(
            obj_pred, obj_target, reduction='sum') / num_total_samples
        
        # cls_loss = cls_loss_bce

        ##### ++ Jaccard loss 
        # loss_fct_j = JaccardLoss()
        # cls_loss_j = loss_fct_j(cls_pred[pos_loc], cls_target[pos_loc] * iou.detach().reshape(-1, 1)) / num_total_samples

        # ### FOCAL Loss
        # gamma = 2.0
        # alpha = 1.0
        # loss_fct1 = nn.BCEWithLogitsLoss(reduction='none')
        # cls_bce_l1 = loss_fct1(cls_pred[pos_loc], cls_target[pos_loc] * iou.detach().reshape(-1, 1))
        # probas = torch.sigmoid(cls_pred[pos_loc])
        # cls_loss_j = torch.where(cls_target[pos_loc] >= 0.5, alpha * (1. - probas)**gamma * cls_bce_l1, probas**gamma * cls_bce_l1)
        # cls_loss_j = cls_loss_j.mean()
        # cls_loss = cls_loss_bce + cls_loss_j

        ###### ## TODO check run2
        cls_eff_loss_bce = F.binary_cross_entropy_with_logits(
            logits_eff_word[pos_loc],  
            cls_eff_target[pos_loc] * iou.detach().reshape(-1, 1), 
            reduction='sum') / num_total_samples

        # loss_fct = nn.CrossEntropyLoss(reduction='sum')
        # cls_eff_loss_ce = loss_fct(logits_eff_word[pos_loc].softmax(-1), cls_eff_target[pos_loc]) / num_total_samples
        # cls_eff_loss = F.cross_entropy(logits_eff_word[pos_loc].softmax(-1), cls_eff_target[pos_loc], reduction='sum') / num_total_samples

        # loss_fct_bce = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor([3.0]).cuda())
        # cls_eff_loss_bce = loss_fct_bce(
        #     logits_eff_word[pos_loc], 
        #     cls_eff_target[pos_loc] 
        #     ) / num_total_samples
        

        loss_fct = nn.CrossEntropyLoss(reduction='mean') # 'sum'
        # cls_eff_loss_ce = loss_fct(logits_eff_disc.softmax(-1), cls_eff_target) / num_total_samples
        cls_eff_loss_ce = loss_fct(logits_eff_disc.softmax(-1), cls_eff_target_disc)  
        # cls_eff_loss = cls_eff_loss_ce

        ### add BCE loss on pos + CE Loss on pooled outs
        cls_eff_loss = cls_eff_loss_ce + cls_eff_loss_bce

        # loss_fct2 = JaccardLoss()
        # cls_eff_loss_j = loss_fct2(logits_eff_word[pos_loc], cls_eff_target[pos_loc]) / num_total_samples
        
        # ### FOCAL Loss 
        # gamma = 2.0 
        # alpha = 1.0 
        # loss_fct2 = nn.BCEWithLogitsLoss(reduction='none')
        # bce_l1 = loss_fct2(logits_eff_word[pos_loc], cls_eff_target[pos_loc])
        # probas = torch.sigmoid(logits_eff_word[pos_loc])
        # cls_eff_loss_focal = torch.where(cls_eff_target[pos_loc] >= 0.5, alpha * (1. - probas)**gamma * bce_l1, probas**gamma * bce_l1)
        # cls_eff_loss_focal = cls_eff_loss_focal.mean()
        
        # cls_eff_loss = cls_eff_loss_bce + cls_eff_loss_focal

        return obj_loss, reg_loss, cls_loss, cls_eff_loss  

    def get_ner_loss(self, outputs, targets, attention_mask):
        # loss_fct = nn.CrossEntropyLoss()
        active_loss = attention_mask.reshape(-1) == 1
        active_logits = outputs.reshape(-1, outputs.shape[-1])
        true_labels = targets.reshape(-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = self.loss_fct(active_logits, true_labels)
        return loss  
    
    
    @torch.no_grad()
    def build_target(self, gt_spans, obj_pred, reg_pred, cls_pred, cls_pred_eff):
        obj_target = torch.zeros_like(obj_pred)
        reg_target = torch.zeros_like(reg_pred)
        cls_target = torch.zeros_like(cls_pred)
        cls_eff_target = torch.zeros_like(cls_pred_eff)
        
        # first token as positive
        pos_loc = gt_spans[:, 0]
        obj_target[pos_loc] = 1
        reg_target[pos_loc, 0] = gt_spans[:, 0].half() if self.cfg.fp16 else gt_spans[:, 0].float()
        reg_target[pos_loc, 1] = gt_spans[:, 1].half() if self.cfg.fp16 else gt_spans[:, 0].float()
        cls_target[pos_loc, gt_spans[:, 2]] = 1
        
        ### ++
        if self.num_classes2==1:
            cls_eff_target[pos_loc, 0] = gt_spans[:, 3].half() if self.cfg.fp16 else gt_spans[:, 3].float()
        else:
            cls_eff_target[pos_loc, gt_spans[:, 3]] = 1

        # dynamically assign one more positive
        if self.dynamic_positive:
            
            # cls_prob = (obj_pred.sigmoid().unsqueeze(1) * cls_pred.sigmoid()).sqrt()
            cls_prob = (obj_pred.unsqueeze(1) * cls_pred).sqrt() ## for BCEwithlogits

            for start, end, label, label_eff in gt_spans:
                _cls_prob = cls_prob[start:end]
                _cls_gt = _cls_prob.new_full((_cls_prob.size(0), ),
                                             label,
                                             dtype=torch.long)
                _cls_gt = F.one_hot(
                    _cls_gt, num_classes=_cls_prob.size(1)).type_as(_cls_prob)
                
                # cls_cost = F.binary_cross_entropy(_cls_prob, _cls_gt, reduction='none').sum(-1)
                ## Try with BCELogits
                cls_cost = F.binary_cross_entropy_with_logits(_cls_prob,
                                                  _cls_gt,
                                                  reduction='none').sum(-1)


                _reg_pred = reg_pred[start:end].exp()
                _reg_loc = torch.arange(_reg_pred.size(0),
                                        device=_reg_pred.device)
                px1 = _reg_loc - _reg_pred[:, 0]
                px2 = _reg_loc + _reg_pred[:, 1]
                ix1 = torch.max(px1, _reg_loc[0])
                ix2 = torch.min(px2, _reg_loc[-1])
                ux1 = torch.min(px1, _reg_loc[0])
                ux2 = torch.max(px2, _reg_loc[-1])
                inter = (ix2 - ix1).clamp(min=0)
                union = (ux2 - ux1).clamp(min=0) + 1e-12
                iou = inter / union
                iou_cost = -torch.log(iou + 1e-12)
                cost = cls_cost + iou_cost

                pos_ind = start + cost.argmin()
                obj_target[pos_ind] = 1
                reg_target[pos_ind, 0] = start
                reg_target[pos_ind, 1] = end
                cls_target[pos_ind, label] = 1
            
                ## TODO try run2 
                cls_eff_target[pos_ind, label_eff] = 1
            
            pos_loc = (obj_target == 1).nonzero().flatten()
        
        return obj_target, reg_target, cls_target, cls_eff_target, pos_loc # 



    

    def training_step(self, batch, batch_idx):

        # out = self.shared_step(batch)
        res1 = self.forward_logits(batch)

        obj_target, reg_target, cls_target, cls_eff_target, pos_loc = self.build_target(
            batch['gt_spans'], res1['obj_pred'], res1['reg_pred'], res1['cls_pred'], res1['logits_eff_word'])
        
        # res2 = self.predict_text_span(batch, batch_idx) # start, end, score, pred_class
        # # if res2 is not None: print('[TRAIN] start/end', res2['start'], res2['end'])


        h = res1['hidden_state'].squeeze(0)  # (seq_len, hidden_size)
        if self.vb: print('[TRAIN] h', h.shape)

        # start_, end_ = res2['start'], res2['end']
        start_, end_ = batch['gt_spans'][:,0], batch['gt_spans'][:,1] 

        # logits_eff = []
        pooled_outs = []
        for st_, en_ in zip(start_, end_):
            # active_logits_eff_word = res1['logits_eff_word'][st_:en_].mean()
            # active_logits_eff_word = self.head_eff(h[st_:en_])                   # len_span x 2 
            # active_logits_eff_word = active_logits_eff_word.mean(0).unsqueeze(0) # 1x2

            pooled_outs.append(h[st_:en_].mean(0).unsqueeze(0))
            # logits_eff.append(active_logits_eff_word)
        
        if self.vb: 
            print('[TRAIN] len pooled_outs', len(pooled_outs))

        # logits_eff = torch.cat(logits_eff) 
        pooled_outs = torch.cat(pooled_outs)
        logits_eff = self.head_eff(pooled_outs)  
        if self.vb: print('[TRAIN] logits_eff', logits_eff.shape)

        eff_target = torch.tensor(batch['gt_spans'][:,-1])
        if self.vb: print('[TRAIN] eff_target', eff_target.shape)
        # print('[TRAIN] cls_eff_target word-level', cls_eff_target.shape)


        obj_loss, reg_loss, cls_loss, cls_eff_loss = self.get_losses(
            obj_pred=res1['obj_pred'],
            reg_pred=res1['reg_pred'],
            cls_pred=res1['cls_pred'],
            obj_target=obj_target,
            reg_target=reg_target,
            cls_target=cls_target,
            pos_loc=pos_loc,
            ###
            logits_eff_word=res1['logits_eff_word'], ## change: num words x num_classes2 (Nd x 2) 
            cls_eff_target=cls_eff_target,           ## change: num words x 1 
            ##
            logits_eff_disc=logits_eff,    ##  num_disc x num_classes2 (Nd x 2) 
            cls_eff_target_disc=eff_target, 
            )

        loss1 = obj_loss + reg_loss + cls_loss
        loss2 = cls_eff_loss
        loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2


        self.log("train_loss", loss, on_step=True, logger=True, prog_bar=True)
        self.log("train_obj_loss", obj_loss, on_step=True, logger=True, prog_bar=True)
        self.log("train_reg_loss", reg_loss, on_step=True, logger=True, prog_bar=True)
        self.log("train_cls_loss", cls_loss, on_step=True, logger=True, prog_bar=True)
        self.log("train_eff_loss", cls_eff_loss, on_step=True, logger=True, prog_bar=True)
        self.log("train_loss1", loss1, on_step=True, logger=True, prog_bar=True)
        # self.log("train_f1", f1, on_step=False, on_epoch=True, logger=True, prog_bar=False)
        
        return {
            'loss': loss, 
            'obj_loss': obj_loss,
            'reg_loss': reg_loss,
            'cls_loss': cls_loss,
            'seg_loss': loss1,
            'eff_loss': cls_eff_loss }
    

    def train_epoch_end(self,outputs):
        tr_avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tr_obj_loss = torch.stack([x['obj_loss'] for x in outputs]).mean()
        tr_reg_loss = torch.stack([x['reg_loss'] for x in outputs]).mean()
        tr_cls_loss = torch.stack([x['cls_loss'] for x in outputs]).mean()

        print(f'epoch {trainer.current_epoch} training losses: {tr_avg_loss:.4f}, obj:{tr_obj_loss:.4f}, reg:{tr_reg_loss:.4f}, cls:{tr_cls_loss:.4f}')
        return {'train_loss_epoch': tr_avg_loss} 


    def validation_step(self, batch, batch_idx):
        # if self.skip_validation > 0: return {}

        # out = self.shared_step(batch)
        res1 = self.forward_logits(batch)

        obj_target, reg_target, cls_target, cls_eff_target, pos_loc = self.build_target(
            batch['gt_spans'], res1['obj_pred'], res1['reg_pred'], res1['cls_pred'], res1['logits_eff_word'])
        
        # res2 = self.predict_text_span(batch, batch_idx) # start, end, score, pred_class
        # # if res2 is not None: print('[TRAIN] start/end', res2['start'], res2['end'])

        h = res1['hidden_state'].squeeze(0)  # (seq_len, hidden_size)
        if self.vb: print('[VAL] h', h.shape)

        # start_, end_ = res2['start'], res2['end']
        start_, end_ = batch['gt_spans'][:,0], batch['gt_spans'][:,1] 

        # logits_eff = []
        # for st_, en_ in zip(start_, end_):
        #     active_logits_eff_word = self.head_eff(h[st_:en_])                   # len_span x 2 
        #     active_logits_eff_word = active_logits_eff_word.mean(0).unsqueeze(0) # 1x2
        #     logits_eff.append(active_logits_eff_word)
        # logits_eff = torch.cat(logits_eff) 
        # logits_eff = torch.stack(logits_eff)
        
        pooled_outs = []
        for st_, en_ in zip(start_, end_):
            pooled_outs.append(h[st_:en_].mean(0).unsqueeze(0))        
        if self.vb: print('[VAL] len pooled_outs', len(pooled_outs))

        pooled_outs = torch.cat(pooled_outs)
        if self.vb: print('[VAL] pooled_outs', pooled_outs.shape)

        logits_eff = self.head_eff(pooled_outs)  

        eff_target = torch.tensor(batch['gt_spans'][:,-1])

        if self.vb: print('[VAL] logits_eff', logits_eff.shape)
        if self.vb: print('[VAL] eff_target', eff_target.shape)
        # print('[TRAIN] cls_eff_target word-level', cls_eff_target.shape)

        obj_loss, reg_loss, cls_loss, cls_eff_loss = self.get_losses(
            obj_pred=res1['obj_pred'],
            reg_pred=res1['reg_pred'],
            cls_pred=res1['cls_pred'],
            obj_target=obj_target,
            reg_target=reg_target,
            cls_target=cls_target,
            pos_loc=pos_loc,
            ###
            logits_eff_word=res1['logits_eff_word'], ## change: num words x num_classes2 (Nd x 2) 
            cls_eff_target=cls_eff_target,           ## change: num words x 1 
            ##
            logits_eff_disc=logits_eff,    ##  num_disc x num_classes2 (Nd x 2) 
            cls_eff_target_disc=eff_target, 
            )

        loss1 = obj_loss + reg_loss + cls_loss
        loss2 = cls_eff_loss
        loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2

        self.log("val_loss", loss, on_step=True, logger=True, prog_bar=True)
        self.log("val_obj_loss", obj_loss, on_step=True, logger=True, prog_bar=True)
        self.log("val_reg_loss", reg_loss, on_step=True, logger=True, prog_bar=True)
        self.log("val_cls_loss", cls_loss, on_step=True, logger=True, prog_bar=True)
        self.log("val_eff_loss", cls_eff_loss, on_step=True, logger=True, prog_bar=True)

        self.val_outs.append({
            "val_loss": loss, 
            "val_obj_loss": obj_loss,
            "val_reg_loss": reg_loss,
            "val_cls_loss": cls_loss,
            "val_eff_loss": cls_eff_loss,
            ###
            # "logits": res1['logits'], 
            # "logits_eff": res1['logits_eff_raw'], 
            'hidden_state': res1['hidden_state'].squeeze(0), 
            'obj_pred': res1['obj_pred'],
            'reg_pred': res1['reg_pred'],
            'cls_pred': res1['cls_pred'],
            ###
            'input_ids': batch['input_ids'], 
            'attention_mask': batch['attention_mask'], 
            'word_boxes': batch['word_boxes'], 
            'text': batch['text'], 
            'text_id': batch['text_id'], 
            })
        
        return {
            'loss': loss, 
            'obj_loss': obj_loss,
            'reg_loss': reg_loss,
            'cls_loss': cls_loss,
            'seg_loss': loss1,
            'eff_loss': cls_eff_loss }


    def on_validation_epoch_start(self):
        print('\n')


    def on_validation_epoch_end(self):

        outputs = self.val_outs
        # print('len outputs', len(outputs))
        
        ######################################################
            # out = self.predict_step(batch, i)
        ######################################################
        # val_loss = 0
        preds = []
        for i, o in enumerate(outputs):
            ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
            # batch = {'input_ids': o['input_ids'], 'attention_mask': o['attention_mask'], 'word_boxes': o['word_boxes'], 'text_id': o['text_id'], 'text': o['text']}
            # obj_pred, reg_pred, cls_pred, logits_eff_word, outputs = self.forward_logits(batch)
            # res1 = self.forward_logits(batch)

            obj_pred = o['obj_pred'].sigmoid() # res1['obj_pred'].sigmoid()
            reg_pred = o['reg_pred'].exp()
            cls_pred = o['cls_pred'].sigmoid()

            obj_scores = obj_pred
            cls_scores, cls_labels = cls_pred.max(-1)
            pr_scores = (obj_scores * cls_scores)**0.5
            
            pos_inds = pr_scores > self.test_score_thr
            
            if pos_inds.sum() == 0:
                return None

            pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
            pos_loc = pos_inds.nonzero().flatten()
            start = pos_loc - reg_pred[pos_inds, 0]
            end = pos_loc + reg_pred[pos_inds, 1]

            min_idx, max_idx = 0, obj_pred.numel() - 1
            start = start.clamp(min=min_idx, max=max_idx).round().long()
            end = end.clamp(min=min_idx, max=max_idx).round().long()

            # nms
            keep = span_nms(start, end, pr_score)
            start = start[keep]
            end = end[keep]
            pr_score = pr_score[keep]
            pr_label = pr_label[keep]

            # ### incl effectiveness preds --> TODO try both 
            # if self.use_sigmoid:
            #     logits_eff_word = logits_eff_word.sigmoid()         ## run 1 (2 classes)
            #     eff_proba, eff_cls = logits_eff_word.max(-1) 

            #     # eff_proba = eff_proba[keep]
            #     eff_proba = logits_eff_word[keep]
            #     eff_cls   = eff_cls[keep]

            #     # eff_proba_0 = eff_proba_1 = eff_proba
            #     eff_proba_1 = eff_proba
            #     eff_proba_0 = 1.0 - eff_proba_1

            # else:
            #     ### use softmax 
            #     logits_eff_word = torch.softmax(logits_eff_word, -1)  ## run 2 (2 classes)

            #     _, eff_cls = logits_eff_word.max(-1) 
            #     eff_proba_1 = logits_eff_word[keep, 1] ## positive class
            #     eff_proba_0 = logits_eff_word[keep, 0] ## negative class
            #     # eff_proba = eff_proba[keep]
            #     eff_proba = eff_proba_1
            #     eff_cls   = eff_cls[keep]

            # # eff_proba, eff_cls = logits_eff_word[start:end].max(-1) 

            h = o['hidden_state'].squeeze(0)  # (seq_len, hidden_size)

            # logits_eff = []
            # for st_, en_ in zip(start, end):
            #     # active_logits_eff_word = res1['logits_eff_word'][st_:en_].mean()
            #     active_logits_eff_word = self.head_eff(h[st_:en_])                   # len_span x 2 
            #     active_logits_eff_word = active_logits_eff_word.mean(0).unsqueeze(0) # 1x2
            #     logits_eff.append(active_logits_eff_word)
            # logits_eff = torch.cat(logits_eff) 

            pooled_outs = []
            for st_, en_ in zip(start, end):
                pooled_outs.append(h[st_:en_].mean(0).unsqueeze(0))        
            if self.vb: 
                print('[VAL] len pooled_outs list', len(pooled_outs))
                for j in range(len()):
                    print(f'[VAL] pooled_out list {j}-th', pooled_outs[j].shape )

            pooled_outs = torch.cat(pooled_outs)
            if self.vb: 
                print('[VAL] pooled_outs before head', pooled_outs.shape)
            logits_eff = self.head_eff(pooled_outs)  

            logits_eff_pred = logits_eff.softmax(-1)
            _, eff_cls = logits_eff_pred.max(-1) 
            eff_proba_0 = logits_eff_pred[:, 0] ## negative class
            eff_proba_1 = logits_eff_pred[:, 1] ## positive class

            # ## TODO in train try calibrate pred by obj 
            # eff_proba_0 = (obj_scores[pos_inds][keep] * logits_eff_pred[:, 0])**0.5
            # eff_proba_1 = (obj_scores[pos_inds][keep] * logits_eff_pred[:, 1])**0.5

            del h 
            torch.cuda.empty_cache()
            gc.collect()

            out = dict(
                text_id=o['text_id'],
                start=to_np(start),
                end=to_np(end),
                score=to_np(pr_score),
                label=to_np(pr_label),
                ##### ++ 
                label_eff=to_np(eff_cls),
                score_eff_0=to_np(eff_proba_0),
                score_eff_1=to_np(eff_proba_1),
                # score_eff=to_np(eff_proba_1),
                )
            ###########################
            test_ps = self.result_to_ps(out) # print('test_ps', test_ps)            
            preds.append(test_ps)  

        ### pred_df
        preds = pd.concat(preds)
        # print('pred_df', preds)

        # avg_f1 = get_f1_score(preds, self.valid_df, slient=False)
        avg_f1 = get_f1_score_v1(preds, self.valid_df, slient=False)
        # metrics = self.compute_metrics(logits , targets, attention_mask=mask)["f1"]
        
        ### reset placeholder 
        self.val_outs = []   
        gc.collect() 

        if avg_f1>self.best_score:
            self.best_score = avg_f1 
        
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # val_loss = val_loss / len(outputs)

        # self.log(f'val_loss_epoch', val_loss, on_epoch=True)
        self.log(f'val_f1_avg', avg_f1, on_epoch=True)

        # print(f'epoch {trainer.current_epoch} validation loss: {val_loss:.4f}')
        print(f'epoch {trainer.current_epoch} validation f1: {avg_f1} (best f1: {self.best_score:.4f})')

        return {'val_loss_epoch': val_loss, 'val_f1': avg_f1}
    

    ###  predict loop
    def predict_text_span(self, batch, batch_idx):
        ######################################################
        ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
        # obj_pred, reg_pred, cls_pred, logits_eff_word, _ = self.forward_logits(batch)
        res1 = self.forward_logits(batch)
        
        obj_pred = res1['obj_pred'].sigmoid()
        reg_pred = res1['reg_pred'].exp()
        cls_pred = res1['cls_pred'].sigmoid()

        # print('[PREDICT] obj_pred', obj_pred.shape) # Num words 

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        pr_scores = (obj_scores * cls_scores)**0.5
        
        pos_inds = pr_scores > self.test_score_thr ## TODO try thres per class 
         

        if pos_inds.sum() == 0:
            return None

        pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score, nms_thr=self.nms_thr)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]

        return dict(
            text_id=batch['text_id'],
            start=to_np(start),
            end=to_np(end),
            score=to_np(pr_score),
            label=to_np(pr_label),
            )


    ###  predict loop 
    def predict_step(self, batch, batch_idx):

        ### ------------------------
        # res2 = self.predict_text_span(batch, batch_idx)
        ### -------------------------
        ######################################################
        ## 'input_ids', 'attention_mask', 'word_boxes', 'text'
        # obj_pred, reg_pred, cls_pred, logits_eff_word, _ = self.forward_logits(batch)
        res1 = self.forward_logits(batch)
        
        obj_pred = res1['obj_pred'].sigmoid()
        reg_pred = res1['reg_pred'].exp()
        cls_pred = res1['cls_pred'].sigmoid()

        obj_scores = obj_pred
        cls_scores, cls_labels = cls_pred.max(-1)
        pr_scores = (obj_scores * cls_scores)**0.5
        
        pos_inds = pr_scores > self.test_score_thr ## TODO try thres per class 

        if pos_inds.sum() == 0:
            return None

        pr_score, pr_label = pr_scores[pos_inds], cls_labels[pos_inds]
        pos_loc = pos_inds.nonzero().flatten()
        start = pos_loc - reg_pred[pos_inds, 0]
        end = pos_loc + reg_pred[pos_inds, 1]

        min_idx, max_idx = 0, obj_pred.numel() - 1
        start = start.clamp(min=min_idx, max=max_idx).round().long()
        end = end.clamp(min=min_idx, max=max_idx).round().long()

        # nms
        keep = span_nms(start, end, pr_score, nms_thr=self.nms_thr)
        start = start[keep]
        end = end[keep]
        pr_score = pr_score[keep]
        pr_label = pr_label[keep]

        # ### incl effectiveness preds (method A) ##  
        # if self.use_sigmoid:
        #     logits_eff_word = logits_eff_word.sigmoid()         ## run 1 (2 classes)
        #     eff_proba, eff_cls = logits_eff_word.max(-1) 

        #     eff_proba = eff_proba[keep]
        #     eff_cls   = eff_cls[keep]

        #     # eff_proba_0 = eff_proba_1 = eff_proba
        #     eff_proba_0 = logits_eff_word[keep, 0] ## negative class
        #     eff_proba_1 = logits_eff_word[keep, 1] ## positive class

        # else:
        #     ### use softmax 
        #     logits_eff_word = torch.softmax(logits_eff_word, -1)  ## run 2 (2 classes)

        #     _, eff_cls = logits_eff_word.max(-1) 
        #     eff_proba_1 = logits_eff_word[keep, 1] ## positive class
        #     eff_proba_0 = logits_eff_word[keep, 0] ## negative class
        #     # eff_proba = eff_proba[keep]
        #     eff_proba = eff_proba_1
        #     eff_cls   = eff_cls[keep]
        ###########################################################
        # (method B) 
        # # eff_proba, eff_cls = logits_eff_word[start:end].max(-1) 
        # eff_proba = []
        # eff_cls = []
        # for st_, en_ in zip(start, end):
        #     ## apply first mean then max
        #     active_logits_eff_word = logits_eff_word[st_:en_].mean()
        #     eff_proba_, eff_cls_ = active_logits_eff_word.max(-1) 

        #     eff_proba.append(eff_proba_.detach().cpu().numpy())
        #     eff_cls.append(eff_cls_.detach().cpu().numpy())
        ############################################################
        # (method C) NEW

        h = res1['hidden_state'].squeeze(0)  # (seq_len, hidden_size)

        pooled_outs = []
        for st_, en_ in zip(start, end):
            pooled_outs.append(h[st_:en_].mean(0).unsqueeze(0))        

        pooled_outs = torch.cat(pooled_outs)
        logits_eff = self.head_eff(pooled_outs)  

        logits_eff_pred = logits_eff.softmax(-1)
        _, eff_cls = logits_eff_pred.max(-1) 
        
        eff_proba_0 = logits_eff_pred[:, 0] ## negative class
        eff_proba_1 = logits_eff_pred[:, 1] ## positive class

        # eff_proba_0 = (obj_scores[pos_inds][keep] * logits_eff_pred[:, 0])**0.5
        # eff_proba_1 = (obj_scores[pos_inds][keep] * logits_eff_pred[:, 1])**0.5


        del h 
        torch.cuda.empty_cache()
        gc.collect()


        return dict(
            text_id=batch['text_id'],
            start=to_np(start),
            end=to_np(end),
            score=to_np(pr_score),
            label=to_np(pr_label),
            # ###
            label_eff=to_np(eff_cls),
            score_eff_0=to_np(eff_proba_0),
            score_eff_1=to_np(eff_proba_1),
            score_eff=to_np(eff_proba_1),
            )
    
    

    def train_dataloader(self):
        return self._train_dataloader 
    

    def validation_dataloader(self):
        return self._valid_dataloader

    
    def raw_predict(self, batch):  
        
        # data = to_gpu(data)  # Step 1: Map the input data to GPU
        batch_size = data['input_ids'].size(0)   
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'
        
        # outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        # logits = outputs['logits']  

        outputs = self.model.deberta(data['input_ids'], data['attention_mask']) 
        logits = outputs.last_hidden_state


        logits = seg_roi_align(logits, batch['boxes'])  # Align token-level outputs to word-level outputs using region-of-interest alignment.
        obj_pred = logits[..., 0].sigmoid()  # Object prediction: A prediction scalar to decide whether an object, i.e., a discourse element, exists
        reg_pred = logits[..., 1:3].exp()  # Region prediction: Two predictions to dictate where the potential discourse element lies 
        #These two numbers are positive and can reach unbounded values (output of an exponential function)
        
        cls_pred = logits[..., 3:].sigmoid()  # 7 Probability predictions for classes to decide, if an object exists, what class it has.
        
        # return batch['text_id'], to_np(obj_pred), to_np(reg_pred), to_np(cls_pred)  
        return dict(
            text_id=batch['text_id'],
            obj_pred=to_np(obj_pred),
            reg_pred=to_np(reg_pred),
            cls_pred=to_np(cls_pred),
            )


class CustomCollator(object):
    def __init__(self, tokenizer, model_name): #model
        self.pad_token_id = tokenizer.pad_token_id
        config_model = config = AutoConfig.from_pretrained(model_name)
        # if hasattr(model.config, 'attention_window'):
        if hasattr(config_model, 'attention_window'):
            # For longformer
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/longformer/modeling_longformer.py#L1548
            self.attention_window = (
                config_model.attention_window if isinstance(
                    config_model.attention_window, int)
                else max(config_model.attention_window)
                )
        else:
            self.attention_window = None

    def __call__(self, samples):
        batch_size = len(samples)
        assert batch_size == 1, f'Only batch_size=1 supported, got batch_size={batch_size}.'

        sample = samples[0]

        max_seq_length = len(sample['input_ids'])
        if self.attention_window is not None:
            attention_window = self.attention_window
            padded_length = (attention_window -
                             max_seq_length % attention_window
                             ) % attention_window + max_seq_length
        else:
            padded_length = max_seq_length

        input_shape = (1, padded_length)
        input_ids = torch.full(input_shape,
                               self.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros(input_shape, dtype=torch.long)
        # labels_eff = torch.zeros(input_shape, dtype=torch.long)

        seq_length = len(sample['input_ids'])
        input_ids[0, :seq_length] = sample['input_ids']
        attention_mask[0, :seq_length] = sample['attention_mask']
        # labels_eff[0, :seq_length] = sample['labels_eff'][:seq_length]

        text_id = sample['text_id']
        text = sample['text']
        word_boxes = sample['word_boxes']
        gt_spans = sample['gt_spans']
        
        ## ++
        span_list = [sample['span_list']]
        discourse_types = [sample['discourse_types']]
 
        return dict(
            text_id=text_id,
            text=text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_boxes=word_boxes,
            gt_spans=gt_spans,
            ## ++
            span_list=span_list,
            discourse_types=discourse_types,
            )
    
class SpanDetectorDatasetTest(Dataset):
    def __init__(self, df, tokenizer,):
        
        self.df = df.copy() #pd.read_csv(csv_file)
        self.samples = list(self.df.groupby('id'))
        self.tokenizer = tokenizer
        print(f'Loaded {len(self)} samples.')


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text_id, df = self.samples[index]
        
        # text_path = os.path.join(self.text_dir, f'{text_id}.txt')
        # with open(text_path) as f:
        #     text = f.read().rstrip()
        text = df['essay_text'].values[0].rstrip()

        tokens = self.tokenizer(text, return_offsets_mapping=True)
        
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.LongTensor(tokens['attention_mask'])
        offset_mapping = np.array(tokens['offset_mapping'])
        offset_mapping = self.strip_offset_mapping(text, offset_mapping)
        num_tokens = len(input_ids)

        # token slices of words
        woff = self.get_word_offsets(text)
        toff = offset_mapping
        wx1, wx2 = woff.T
        tx1, tx2 = toff.T
        ix1 = np.maximum(wx1[..., None], tx1[None, ...])
        ix2 = np.minimum(wx2[..., None], tx2[None, ...])
        ux1 = np.minimum(wx1[..., None], tx1[None, ...])
        ux2 = np.maximum(wx2[..., None], tx2[None, ...])
        ious = (ix2 - ix1).clip(min=0) / (ux2 - ux1 + 1e-12)
        assert (ious > 0).any(-1).all()

        word_boxes = []
        for row in ious:
            inds = row.nonzero()[0]
            word_boxes.append([inds[0], 0, inds[-1] + 1, 1])
        word_boxes = torch.FloatTensor(word_boxes)


        return dict(
            text=text,
            text_id=text_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_boxes=word_boxes)
    

    def strip_offset_mapping(self, text, offset_mapping):
        ret = []
        for start, end in offset_mapping:
            match = list(re.finditer('\S+', text[start:end]))
            if len(match) == 0:
                ret.append((start, end))
            else:
                span_start, span_end = match[0].span()
                ret.append((start + span_start, start + span_end))
        return np.array(ret)

    def get_word_offsets(self, text):
        matches = re.finditer("\S+", text)
        spans = []
        words = []
        for match in matches:
            span = match.span()
            word = match.group()
            spans.append(span)
            words.append(word)
        assert tuple(words) == tuple(text.split())
        return np.array(spans)
    
### RUN INFERENCE
args = parse_args()   



class CFG:
    exp_name = 'exp36_base' 
    task = "ner"
    n_folds = 5  
    trn_folds = [0]  
    model_name = "microsoft/deberta-base"
    pretrained_detector_path = '/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/checkpoints/tasckj' #'none' # 'taskj'
    max_length = 1536 #768  
    stride = 128
    min_tokens = 6
    # TRAINING HYPERPARAMS
    batch_size = 1 
    epochs = 1 #4 #1
    gradient_checkpointing = True 
    scheduler = "Linear"
    LR = 6e-5 #2e-5    # 4e-5    ## 3e-05
    WD = 0.01
    WARMUP = 0.1
    fp16 = False 
    train = False
    infer = True
    debug = False # set True for debugging
    wandb = False # False
    wb_project = "NLP-Rater"
    output = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/"
    seed = 42


DIR = Path("/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/")
OUTPUT = Path(CFG.output + args.exp_name)


TEST_TH = args.test_th #0.5 
MDL_PATH = args.checkpoint
MDL_SWA = args.swa_checkpoint
CFG.use_swa = True if MDL_SWA is not None else False

# MDL_PATH = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/weights_f2/checkpoint-epoch=0-val_loss=5.8498.ckpt" ## CV 0.6743
# MDL_SWA = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/swa_f0.pt"
# MDL_SWA = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/swa_f1.pt"
# MDL_SWA = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/swa_f2.pt"
# MDL_SWA = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp36_base/swa_f3.pt"

trainer = pl.Trainer(
    max_epochs=1,  
    accelerator='gpu', #'cpu', 
    precision = 16 if CFG.fp16 else 32,
)
CFG.tokenizer = AutoTokenizer.from_pretrained(OUTPUT / 'tokenizer')


# if CFG.infer:

test = pd.read_csv('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/input/test.csv')
test.columns = ['id', 'essay_text']
test['essay_id'] = test['id'] 

print(test.shape)
# display(test.head(2))

test_ds = SpanDetectorDatasetTest(test, tokenizer=CFG.tokenizer)
# print(test_ds[0].keys())

test_loader = DataLoader(test_ds,
                        batch_size=1,
                        shuffle=False,
                        num_workers=2,
                        collate_fn=CustomCollatorTest(CFG.tokenizer, CFG.model_name)
                        )

### ---------------
### MODEL 
### ---------------
litmodel = TextSpanDetectorPL_FP1.load_from_checkpoint(MDL_PATH,
        model_name=CFG.model_name,
        cfg=CFG, test_score_thr = TEST_TH 
        )
print('Infer model:', MDL_PATH.split('/')[-2], MDL_PATH.split('/')[-1])
print('test th:', litmodel.test_score_thr)

if CFG.use_swa and MDL_SWA is not None:
    litmodel.load_state_dict(torch.load(MDL_SWA)['state_dict'])
    print('--> SWA model loaded', MDL_SWA)

# res = trainer.predict(model, test_loader, return_predictions=True)  
litmodel.eval()

results = []   
for data in tqdm(dl):   
    data = to_gpu(data)
    with torch.no_grad():   
        # result = detector.raw_predict(data)
        result = litmodel.raw_predict(data)      
    results.append(result)  # And append the result

print(len(results))
save_path = OUTPUT / args.outfile
save_pkl(results, save_path)
print('--> raw preds saved at', save_path)
