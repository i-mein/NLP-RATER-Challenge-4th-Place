cd scripts

### LB 0.5993
# CUDA_VISIBLE_DEVICES=0 python inference_exp61.py  --model_paths /workspace/checkpoints/exp61/model_seed100_fold-1_epoch4.pth \
# --model_name deberta-large \
# --save_name submission \
# --save_dir /workspace/results/exp61/ \
# --fold -1 --epoch 4 --test_th 0.6

### LB 0.6018
# CUDA_VISIBLE_DEVICES=0 python inference_exp61.py  --model_paths /workspace/checkpoints/exp61/model_seed100_fold-1_epoch4.pth \
# --model_name deberta-large \
# --save_name submission \
# --save_dir /workspace/results/exp61/ \
# --fold -1 --epoch 4 --test_th 0.65

# ### LB 0.6004
# CUDA_VISIBLE_DEVICES=0 python inference_exp61.py  --model_paths /workspace/checkpoints/exp61/model_seed100_fold-1_epoch4.pth \
# --model_name deberta-large \
# --save_name submission \
# --save_dir /workspace/results/exp61/ \
# --fold -1 --epoch 4 --test_th 0.70

### LB 0.6029
CUDA_VISIBLE_DEVICES=0 python inference_exp61.py  --model_paths /workspace/checkpoints/exp61/swa_full_ep34.pt \
--model_name deberta-large \
--save_name submission \
--save_dir /workspace/results/exp61/ \
--fold -1 --epoch 34 --test_th 0.65



