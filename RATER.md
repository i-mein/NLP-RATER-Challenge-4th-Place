# Rater Challenge Learning Agency Lab

Description:


Wandb:  




## TODO

- tune awp params - OK
- pretrained models from Feedback v1
- pseudo label dataset Feedback v1 single model per fold --> fb1_pseudo_single_cv_<xxx>
- pseudo label dataset Feedback v1 ensemble --> fb1_pseudo_ens_cv_<xxx>
- pretrain with pseudo 1-2 epochs + finetune on comp data
- finetune with pseudo + comp data

- add aux targets, eg sum_sore, mean_score
- Multi-out model (6 heads)

- SVR emb finetuned models
- SVR emb finetuned models + PCA
- SVR emb + meta feats




## experiments 

| EXP | backbone                       | pool | model              | max_len | CV      | LB    | Notes                   |
|-----|--------------------------------|------|--------------------|---------|---------|-------|-------------------------|
| r01 | ms/deberta-v3-base             | cls  | FB1Model           | 1024    | 0.x     |       | NERSegLoss              |
| r02 |    | cls  | FeedbackModel      | 1024    | 0.x     |       | AuxLoss                 |
| r03 |           | cls  | FeedbackModel      | 2560    |         |       | resolve_encod           |
| r04 |   |      |                    | 2048    |         |       |                         |
| r05 |                                |      |                    |         |         |       |                         |
| r06 |            |      |                    | 1560    | 0.x |       |   |
| r07 |                                |      |                    |         |         |       |          |
| r08 |                                |      |                    |         |         |       |          |
| r09 |                                |      |                    |         |         |       |          |
 


## Submissions (baseline)

LOGS 

- V1 LB 0.4428
    - Token_clf : FB1 Models 7th ['cfg_ch_10c', 'cfg_ch_34d'] --> sub_01 (missing: 3 essay_ids)
    -  Effective : FB2 Model 5th single fold0 --> sub_02 


- V2 LB 0.4458 
    - Token_clf : FB1 Models 7th ['cfg_ch_10c', 'cfg_ch_34d'] --> sub_01 (missing: 3 essay_ids)
    - Effective : FB2 Model 5th single x 5 folds --> sub_02 

- V3 LB 0.411
    - V2 + NN stacker / BUG?

- V4 LB 0.4x
    - V2 + NN stacker 


- V5 LB 0.433
    - V2 + LGB stacker 


- V6 LB 
    - Token_clf : FB1 Model 1st Longformer 
    - Effective : FB2 Model 5th single x 5 folds


- V7 LB 
    - Token_clf : FB1 Model 1st Longformer + LGB
    - Effective : FB2 Model 5th single x 5 folds


## Backbones 

- microsoft/deberta-xlarge
- microsoft/deberta-v2-xlarge

- deberta-small/large 
- deberta-V2-small/xlarge 
- deberta-V3-small/large 
- lsanochkin/deberta-l 

- funnel 
- funnel-transformer/medium
- funnel-transformer/large
- funnel-transformer/xlarge 

- allenai/longformer-large-4096

- google/electra-large-discriminator
- google/bigbird-roberta-large
- facebook/bart-base   ### OOM
- facebook/bart-large  ### OOM
- facebook/muppet-roberta-large
- albert-xxlarge-v2
- microsoft/mpnet-base


## Papers
- xx
