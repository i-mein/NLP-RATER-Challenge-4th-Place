import os, sys, gc 
import pandas as pd 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--csv_path", type=str, required=True)
ap.add_argument("--save_name", type=str, required=True)
ap.add_argument("--save_dir", type=str, required=False)
# ap.add_argument("--debug", type=bool, default=False, required=False)
cargs = ap.parse_args()

CSV_PATH = cargs.csv_path
SAVE_DIR = cargs.save_dir
SAVE_NAME = cargs.save_name

min_thresh = {
    'Lead': 3,
    'Position': 4,
    'Evidence': 4,
    'Claim': 2,
    'Concluding Statement': 9,
    'Counterclaim': 5,
    'Rebuttal': 2,
}


proba_thresh = {
    "Lead": 0.27,
    "Position": 0.28,
    "Evidence": 0.39,
    "Claim": 0.30,
    "Concluding Statement": 0.36,
    "Counterclaim": 0.21,
    "Rebuttal": 0.20,
}

def do_threshold(submit_df, min_thresh, proba_thresh, use=['length','probability']):
    df = submit_df.copy()
    df = df.fillna('')

    if 'length' in use:
        df['l'] = df.predictionstring.apply(lambda x: len(x.split()))
        for key, value in min_thresh.items():
            index = df.loc[df['class'] == key].query('l<%d'%value).index
            df.drop(index, inplace=True)

    if 'probability' in use:
        for key, value in proba_thresh.items():
            index = df.loc[df['class'] == key].query('score<%f'%value).index
            df.drop(index, inplace=True)
    return df


sub = pd.read_csv(CSV_PATH)
print('Load submission from:', CSV_PATH, sub.shape)

if 'score' in sub.columns:
    sub_pp = do_threshold(sub, min_thresh, proba_thresh, use=['length','probability'])
else:
    sub_pp = do_threshold(sub, min_thresh, proba_thresh, use=['length'])

sub_pp['essay_id_comp'] = sub_pp['id']
sub_pp.to_csv(os.path.join(SAVE_DIR, SAVE_NAME, '.csv'), index=False)

