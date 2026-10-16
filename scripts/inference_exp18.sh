cd scripts 

### LB ?
CUDA_VISIBLE_DEVICES=0 python inference_exp18.py  --fold 0  --test_th 0.50 --save_dir /workspace/results/exp18/

# CUDA_VISIBLE_DEVICES=0 python inference_exp18.py  --fold 1  --test_th 0.55 --save_dir /workspace/results/exp18/
# CUDA_VISIBLE_DEVICES=0 python inference_exp18.py  --fold 2  --test_th 0.55 --save_dir /workspace/results/exp18/
# CUDA_VISIBLE_DEVICES=0 python inference_exp18.py  --fold 3  --test_th 0.55 --save_dir /workspace/results/exp18/
# CUDA_VISIBLE_DEVICES=0 python inference_exp18.py  --fold 4  --test_th 0.55 --save_dir /workspace/results/exp18/
# CUDA_VISIBLE_DEVICES=0 python inference_exp18.py  --fold -1 --test_th 0.55 --save_dir /workspace/results/exp18/       

