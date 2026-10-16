cd scripts

### LB 0.6029
CUDA_VISIBLE_DEVICES=0 python inference_exp61.py  --model_paths /workspace/checkpoints/exp61/swa_full_ep34.pt \
--model_name deberta-large \
--save_name submission \
--save_dir /workspace/results/exp61/ \
--fold -1 --epoch 34 --test_th 0.65


### exp61 seed#2
CUDA_VISIBLE_DEVICES=0 python inference_exp61.py  --model_paths /workspace/checkpoints/exp61/model_seed66_fold0_epoch4.pth \
--model_name deberta-large \
--save_name submission \
--save_dir /workspace/results/exp61_V2/ \
--fold 0 --epoch 4 --test_th 0.60


### exp66 xl
CUDA_VISIBLE_DEVICES=0 python inference_exp66.py  --model_paths /workspace/checkpoints/exp66/model_seed100_fold-1_epoch3.pth \
--model_name deberta-xlarge \
--save_name submission \
--save_dir /workspace/results/exp66/ \
--fold -1 --epoch 3 --test_th 0.65


### WBF ens 
CUDA_VISIBLE_DEVICES=0 python ensemble_wbf.py  --csv_paths /workspace/results/exp61/test_preds.csv /workspace/results/exp66/test_preds.csv /workspace/results/exp61_V2/test_preds.csv \
--save_name submission --save_dir /workspace/results/ \
--iou_th 0.20



# ### PP [TODO]
# CUDA_VISIBLE_DEVICES=0 python post_process.py --csv_path /workspace/results/exp61/submission.csv \
# --save_name submission_pp \
# --save_dir /workspace/results/

# mv /workspace/results/submission_pp.csv /workspace/results/submission_final.csv
### END 