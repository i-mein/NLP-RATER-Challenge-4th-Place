# Rater Challenge 

Predict effectiveness and discourse elements with a unique model architecture 

My Solution for the RATER Competition
This repository contains my solution for the RATER competition.


### Challenges
- Long essay texts 
- multi-task:  i) regression ii) text segmentation (NER)
- xxx
and.. 
- High competition candidates with strong baselines from previous 


### Next steps 

- read 2-3 previous solutions to get familiar with the problem/dataset & check/run code
- understand the metric 
- ⁠implement metric in torch/numpy --> crucial for evaluating our models/ideas 
- EDA, preprocess ideas
- pipeline για train + infer 
- MLOps gia exp racking + metrics (wandb, to be discussed)
- git repo
- set attack plan for next 2-3 weeks 
- find/study relevant papers/code 



-------------------


# Winning solutions 

## Feedback I

From here you can check any winning solution (focus on top 10)
https://www.kaggle.com/competitions/feedback-prize-2021/leaderboard


- 1st place: https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177
     - 2 stage solution: predict token --> sentence pred with LGBM 
     - GBDT models (LGBM)
     - git code: https://github.com/antmachineintelligence/Feedback_1st
     - longformer train: https://www.kaggle.com/wht1996/feedback-nn-train
     - lgb train: https://www.kaggle.com/wht1996/feedback-lgb-train
     - longfomer with post-processing(lb 0.697): https://www.kaggle.com/wht1996/feedback-longformer-5fold-0-697
     - ongfomer with lgb(lb 0.727): https://www.kaggle.com/wht1996/feedback-two-stage-lb0-727
     - cv ensemble (cv 0.747): https://www.kaggle.com/wht1996/feedback-two-stage-cv0-747



- 2nd place: https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389
    - WBF post-process 
    - training tips 
    - deberta, funnel, longformer, yoso 
    - Train code: https://github.com/ubamba98/feedback-prize
    - Infer code: https://www.kaggle.com/cdeotte/2nd-place-solution-cv741-public727-private740



- 3rd place:
    - deberta, longformer + GBDT models --> text span preds
    - deberta with sliding window for long texts 
    - Train code: https://github.com/Shujun-He/3rd_Solution_Feedback_Prize_Evaluating_Student_Writing
    - Infer code: https://www.kaggle.com/code/aerdem4/xgb-lgb-feedback-prize-cv-0-7322/notebook



- 4th place:



- 5th place:



## Feedback II

https://www.kaggle.com/competitions/feedback-prize-effectiveness/leaderboard 


- 1st place: https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536
     - infer: https://www.kaggle.com/code/ybabakhin/team-hydrogen-1st-place
     - train code: https://github.com/ybabakhin/kaggle-feedback-effectiveness-1st-place-solution
     - Efficiency solution post: https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347537
     - Efficiency code: https://www.kaggle.com/code/philippsinger/team-hydrogen-efficiency-prize-1st-place


- 2nd place:



- 3rd place:




- 4th place:





- 5th place:



