cd scripts

### LB 0.5918
CUDA_VISIBLE_DEVICES=0 python inference_exp66.py  --model_paths /workspace/checkpoints/exp66/model_seed100_fold-1_epoch3.pth \
--model_name deberta-xlarge \
--save_name submission \
--save_dir /workspace/results/exp66/ \
--fold -1 --epoch 3 --test_th 0.65


# # ### LB 0.5889
# CUDA_VISIBLE_DEVICES=0 python inference_exp66.py  --model_paths /workspace/checkpoints/exp66/model_seed100_fold-1_epoch3.pth \
# --model_name deberta-xlarge \
# --save_name submission \
# --save_dir /workspace/results/exp66/ \
# --fold -1 --epoch 3 --test_th 0.60


# CUDA_VISIBLE_DEVICES=0 python inference_exp66.py  --model_paths /workspace/checkpoints/exp66/model_seed100_fold-1_epoch2.pth \
# --model_name deberta-xlarge \
# --save_name submission \
# --save_dir /workspace/results/exp66/ \
# --fold -1 --epoch 2 --test_th 0.60


