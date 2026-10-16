cd scripts

### LB 0.6029
CUDA_VISIBLE_DEVICES=0 python inference_exp61.py  --model_paths /workspace/checkpoints/exp61/swa_full_ep34.pt \
--model_name deberta-large \
--save_name submission \
--save_dir /workspace/results/exp61/ \
--fold -1 --epoch 34 --test_th 0.65

mv /workspace/results/exp61/submission.csv /workspace/results/submission_final.csv

# ### PP [TODO]
# CUDA_VISIBLE_DEVICES=0 python post_process.py --csv_path /workspace/results/exp61/submission.csv \
# --save_name submission_pp \
# --save_dir /workspace/results/

# mv /workspace/results/submission_pp.csv /workspace/results/submission_final.csv
### END 