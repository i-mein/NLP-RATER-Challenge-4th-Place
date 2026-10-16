# IME Experiments 



```
branch: ime-fb1-4th/
│
├── README.md             
│
├── config/              # .yaml config files for running experiments at scale
│
├── input/               # (should be ignored by git)
│   ├── train_fold10_sgkf.py    # Makes src a Python module
│
├── nbs/                 # Jupyter notebooks for exploratory data analysis and experimentation
│   └── exp01-train-hf.nb
│   └── exp02-train-torch.nb
│   └── ...
│
├── src/                    # Source code for use in this project
│   ├── __init__.py         # Makes src a Python module
│   │
│   └── comp_metric.py      # Feedback competition metric 
│   └── configs.py          # Global configs 
│   └── dataset.py          # Dataset classes 
│   └── models.py           # Model architectures 
│   └── training.py         # Train engine
│   └── inference.py        # Infer engine
│   └── wbf.py              # Weighted Box Fusion (WBF)
│   └── post_processing.py  # Weighted Box Fusion (WBF)
│   └── utils.py            # Utility files
│
├── output/            # Output results (model weights, .h5, .pkl, log files, predictions, ..)
│   └── exp01/
│       └── tokenizer/
│       └── checkpoints/
│           └── <MODEL_NAME>_fold0.pt
│           └── <MODEL_NAME>_fold1.pt
│           └── ...
│       └── config.pth
│       └── ...
│   └── exp02/
│   └── ...
│
├── docs/              # Project documentation
│   ├── data_dictionary.md
│   └── project_report.md
│
├── requirements.txt   # The dependencies file for reproducing the analysis environment
├── .gitignore         # Specifies intentionally untracked files to ignore
└── .env               # (Optional) Environment variables file (should be ignored by git)
├── LICENSE            # (Optional) License information

```

## Modelling 

- FP-02 multitask model --> see exp18 nb: `./nbs/18-FP1-02-HF-train.ipynb`
  - `src/models/hf_debertabilstmV2`
  - `src/models/hf_longofrmerbilstmV2`
  - `src/trainer_hf.py --config exp.yaml`
  - Inference: `src/generate_preds_hf.py` 



## Next steps 

- Implement multitask model/post-process based on FP 1st sol
- Implement multitask model/post-process based on FP 2nd sol --> OK IME
- Implement multitask model/post-process based on FP 3rd sol --> WiP IME needs debug
- Implement multitask model/post-process based on FP 7th sol --> WiP 

- Work on 2 stage approach(es)
- Select token clf model/approach and add LLM embeddings for predict effectiveness  

