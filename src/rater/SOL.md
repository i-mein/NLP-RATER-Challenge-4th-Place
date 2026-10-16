# RATER Competition 



# Solution description 

The overview of my solution is described in `solution.pdf`


# Code structure

```
│
├── README.md             
│
├── data/                 # competition dataset
│   ├── train.csv    
│   ├── test.csv    
│   ├── sample_submission.csv       
│   ├── train_fold10_sgkf.csv  
│   ├── train_sgkf_fold5.csv  
│
├── checkpoints/           # model weights 
│   ├── exp61/             # deberta-large 
│   ├── exp65/             # deberta-large 
│   ├── exp65_V9/          # deberta-large 
│   ├── exp66/             # deberta-xlarge 
│   ├── tascj/             # FP1 weights (deberta-large, deberta-xlarge) 
│   ├── hf_pretrained/     # HF weights (deberta-large, deberta-xlarge) 
│
├── scripts/                     # Inference code (.py, .sh)  
│   └── inference_exp61.py       # Inference for exp61 (.py)
│   └── inference_exp61.sh       # Run inference for exp61 (.sh)
│   └── inference_exp65_V9.py    # Inference for exp65 (Not used in final solution) 
│   └── inference_exp65_V9.sh    # Run inference for exp65 (Not used in final solution) 
│   └── inference_exp66.py       # Inference for exp66 (.py)
│   └── inference_exp66.sh       # Run inference for exp66 (.sh)
│   └── ensemble_wbf.py          # Ensemble with WBF  
│   └── inference_final_V2.sh    # Run inference for final submission 
│   └── ...
│
├── src/                    # Source code for use in this project
│   ├── __init__.py         # Makes src a Python module
│   └── losses/             # Losses (jaccard)
│   └── models/             # Model architectures 
│   └── train/              # Training exp [TODO]
│   └── inference/          # Inference code (Moved to /scripts for convenience)
│   └── utils/              # Utility files
│
├── results/            # Output results (.pkl, log files, predictions, ..)
│   └── exp61/
│       └── submission.csv   
│       └── test_preds.csv   
│       └── test_raw.pkl 
│       └── ...
│   └── exp66/
│       └── submission.csv   
│       └── test_preds.csv   
│       └── test_raw.pkl 
│       └── ...
│   └── submission_final.csv  # 
│
├── nbs/                 # (Optional) Jupyter notebooks [TODO]
│   └── exp18-train-hf.nb
│   └── exp61-train.nb
│   └── ...
│
├── configs/             # (Optional) .yaml config files for training experiments [TODO]
│
├── docs/                # Project documentation
│   └──  solution.pdf
│
├── requirements.txt   # The dependencies files for reproducing the environment
├── .gitignore         # (Optional) Specifies intentionally untracked files to ignore
└── .env               # (Optional) Environment variables file (should be ignored by git)
├── LICENSE            # (Optional) License information

```


# Inference Instructions 

```bash
rm -r results/*.csv

cd scripts 

./inference_final_V2.sh 
```

The intermediate outputs and the submission file are saved in the folder `results`, e.g. 
```bash
$ ls results/exp61

test_raw.pkl  submission.csv  test_preds.csv
```

The final submission file: `results/submission_final.csv`

Verify submission file:
```bash
head results/submission_final.csv
```


# Training Instructions 

To training code to reproduce the models is inside the inference script. 

1. Simply modifying the flags in configs `CFG.train=True` and `CFG.train_folds=-1` 
will start training run for 5 epochs using full dataset. 
e.g. for `deberta-large (exp61)`  

```bash
scripts/inference_exp61.py 
```

OR
```bash
src/train/exp61_train.py  
```


2. Average weights of last 2 epochs using the following code snippet:

```bash

def average_checkpoints_pt(input_ckpts, save_dir="./", save_name="swa"):

    print('num checkpoints:', len(input_ckpts))
    swa_dict = torch.load(input_ckpts[0], map_location='cpu') 

    swa_n = 1
    for ckpt in input_ckpts[1:]:
        print('\nLoad weights from', ckpt)
        new_dict = torch.load(ckpt, map_location='cpu') 

        swa_n += 1
        for k, v in new_dict.items():
            if v.dtype == torch.int64:
                continue
            swa_dict[k] += (new_dict[k] - swa_dict[k]) / swa_n

    torch.save(swa_dict, f'save_dir/{save_name}.pt')

```

example:

```bash

MDL_PATHS = [
  '/workspace/results/exp61/fold-1/model_seed100_fold-1_epoch3.pth', 
  '/workspace/results/exp61/fold-1/model_seed100_fold-1_epoch4.pth',
  ]

average_checkpoints_pt(MDL_PATHS, save_dir='/workspace/results/exp61', save_name=f'swa')
```
