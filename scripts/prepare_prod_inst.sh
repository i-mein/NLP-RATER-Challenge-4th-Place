## 1. ssh to VM instance 
# $ ssh root@149.36.0.242 -p 28042 -i /Users/ime/.ssh/CompetitorKey

## 2. create directory to save the code
# cd home
# mkdir final_v1
# cd final_v1
# mkdir data
# mkdir inference
# mkdir utils
# chmod +x ./scripts/*.sh 

## 2b. Install dependencies 

# pip3 cache purge
# TMPDIR=/workspace/pip_cache/  pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
# TMPDIR=/workspace/pip_cache/  pip install -r requirements.txt 


## 3. upload to VM instance the code, data and checkpoints 

# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/scripts/ root@149.36.0.242:/workspace/
# scp -P 28042 -i ./CompetitorKey requirements.txt root@149.36.0.242:/workspace/

### checkpoints 
# kaggle datasets download -d tascj0/feedback-checkpoints -f deberta_large_fold0.pth --unzip
# kaggle datasets download -d tascj0/feedback-checkpoints -f deberta_xlarge_fold0.pth --unzip

# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/hf_models root@149.36.0.242:/workspace/checkpoints/hf_hub
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/checkpoints/debertalstm root@149.36.0.242:/workspace/checkpoints/hf_pretrained/debertalstm


### exp18
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f0 root@149.36.0.242:/workspace/checkpoints/exp18
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f1 root@149.36.0.242:/workspace/checkpoints/exp18
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f2 root@149.36.0.242:/workspace/checkpoints/exp18
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f3 root@149.36.0.242:/workspace/checkpoints/exp18
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp18/deberta-large-f4 root@149.36.0.242:/workspace/checkpoints/exp18


### exp61 
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp61_fix/fold-1/model_seed100_fold-1_epoch4.pth root@149.36.0.242:/workspace/checkpoints/exp61
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp61_fix/fold-1/swa_full_ep34.pt root@149.36.0.242:/workspace/checkpoints/exp61
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp61_full2/fold0/model_seed66_fold0_epoch4.pth root@149.36.0.242:/workspace/checkpoints/exp61


### exp61 fullfit 2
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp61_full2/fold-1/model_seed66_fold-1_epoch4.pth root@149.36.0.242:/workspace/checkpoints/exp61_V2



### exp65  
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp65_V9/fold0/model_seed100_fold0_epoch3.pth root@149.36.0.242:/workspace/checkpoints/exp65_V9
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp65_V9/fold0/model_seed100_fold0_epoch4.pth root@149.36.0.242:/workspace/checkpoints/exp65_V9

# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/scripts/inference_exp65V9.py root@149.36.0.242:/workspace/src/inference/exp65_V9/
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/scripts/inference_exp65V9.sh root@149.36.0.242:/workspace/src/inference/exp65_V9/

### exp66 
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp66_xl/fold-1/model_seed100_fold-1_epoch3.pth root@149.36.0.242:/workspace/checkpoints/exp66


### WBF trials 
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp61_fix/test_preds_f-1_ep34_th0.65.csv root@149.36.0.242:/workspace/results/exp61/test_preds.csv
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp66_xl/test_preds_f-1_ep3_th0.65.csv  root@149.36.0.242:/workspace/results/exp66/test_preds.csv
# scp -P 28042 -i ./CompetitorKey -r /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp61_full2/test_preds_f0_ep4.csv  root@149.36.0.242:/workspace/results/exp61_V2/test_preds.csv

# scp -P 28042 -i ./CompetitorKey -r root@149.36.0.242:/workspace/results/submission_final.csv  /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/submissions/sub_wbf_final.csv  


## INSTRUCTIONS INFERENCE 





## INSTRUCTIONS TRAINING 



