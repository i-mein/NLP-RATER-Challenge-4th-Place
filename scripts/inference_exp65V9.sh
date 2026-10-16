cd scripts
# CUDA_VISIBLE_DEVICES=0 python inference_exp65V9.py  --model_paths /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp65_V9/fold0/model_seed100_fold0_epoch3.pth \
# --model_name deberta-large \
# --save_name exp65_f0 \
# --save_dir /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp65_V9/ \
# --fold 0 --epoch 3 --test_th 0.6


CUDA_VISIBLE_DEVICES=0 python inference_exp65V9.py  --model_paths /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp65_V9/fold0/model_seed100_fold0_epoch3.pth \
--model_name deberta-large \
--save_name exp65_f0 \
--save_dir /mnt/md1/Work/dev_IME/Git_hacks/Feedback3/output/exp65_V9/ \
--fold 0 --epoch 3 --test_th 0.5



