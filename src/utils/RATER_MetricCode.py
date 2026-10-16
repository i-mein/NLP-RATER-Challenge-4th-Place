import os
import gc
import sys
import json
import time
import torch
import joblib
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
# import plotly.express as px


# ==============================================================================================
# Parameters
# ==============================================================================================
LABEL2TYPE = ('Lead', 'Position', 'Claim', 'Evidence', 'Counterclaim', 'Rebuttal', 'Concluding Statement')
TYPE2LABEL = {t: l for l, t in enumerate(LABEL2TYPE)}


# ==============================================================================================
# Pre-Processing
# ==============================================================================================
## Overlap Removal
def modified(pred_string, diff_words):
    return len(pred_string) > len(diff_words)

def get_original(modified, pred_string):
    if modified:
        return pred_string
    else:
        return "--"

def remove_overlaps(pred_df, by_discourse=False, verbose=False):
    pred_df['predictionstring'] = [[int(x) for x in pred.split(' ')] for pred in pred_df['predictionstring']]  # Map to integers. Keep as list for better vectorization.
    pred_df["nb_words"] = pred_df["predictionstring"].map(lambda x: len(x))
    pred_df["start_word"] = pred_df["predictionstring"].map(lambda x: min(x))  # Some Dataframes already have this information, but computing in case they don't in future
    # pred_df["end_word"] = pred_df["predictionstring"].map(lambda x: max(x))

    if by_discourse:  # If only removing overlaps within the same discourse element
        grouped_df = pred_df.groupby(["id", "discourse_type"])  # Group by essay and discourse type
    else:
        grouped_df = pred_df.groupby("id") # Group by essay only
    
    processed_groups = []
    removed_groups = []  # Keep track of cleaned up segments
    nb_groups = len(grouped_df)
    pbar = tqdm(total=nb_groups)
    segments_affected = 0
    for name, grp in grouped_df:
        pbar.update(1)
        grp_sorted = grp.sort_values(["start_word",'nb_words'],ascending=True)
        
        grp_sorted["cum_words"] = grp_sorted["predictionstring"].cumsum().apply(set)  # Compute cumulative words seen so far
        grp_sorted["cum_words"] = np.concatenate((np.array([{}]), grp_sorted["cum_words"].iloc[0:-1])) # Shift by 1
        # Need to shift to consider cumulation up to but excluding current segment.
        grp_sorted["set_diff_words"] = (grp_sorted["predictionstring"].map(set) - grp_sorted["cum_words"].map(set)).apply(list)  # Compute set difference with current segment in a vectorized way
        # Reference: https://stackoverflow.com/questions/28457149/how-to-map-a-function-using-multiple-columns-in-pandas -> ListComp faster than apply for some reason.
        grp_sorted["modified"] = [modified(pred, diff_w) for pred, diff_w in zip(grp_sorted["predictionstring"], grp_sorted["set_diff_words"])]  # Check if set difference is shorter (i.e., there is an overlap) in a vect way
        grp_sorted["noncontiguous"] = grp_sorted["set_diff_words"].map(lambda x: np.any(np.diff(x) > 1))  # Non-contiguous segments post overlap removal will be deleted
        grp_sorted["nonzero"] = grp_sorted["set_diff_words"].map(lambda x: len(x) > 0)  # Same for empty segments
        grp_sorted['original_predstring'] = [get_original(mod, pred) for mod, pred in zip(grp_sorted["modified"], grp_sorted["predictionstring"])]
        grp_sorted["predictionstring"] = grp_sorted["set_diff_words"].map(lambda x: " ".join([str(y) for y in x]))
        # Now filter out
        grp_processed = grp_sorted[((~grp_sorted["noncontiguous"]) & grp_sorted["nonzero"])]
        processed_groups.append(grp_processed)
        grp_removed = grp_sorted[((grp_sorted["noncontiguous"]) | ~grp_sorted["nonzero"])]
        removed_groups.append(grp_removed)
        
        if verbose and len(grp_removed) > 0:
            print("Discarded segments")
            print(grp_removed)
    # End of all group for loops
    new_segments = pd.concat(processed_groups)[["id", "discourse_type", "score_discourse_effectiveness", "score_discourse_effectiveness_0", 
                                               "score_discourse_effectiveness_1", 
                                               "predictionstring", "modified", "original_predstring"]].to_csv("processed.csv")
    removed_segments = pd.concat(removed_groups)[["id", "discourse_type", "score_discourse_effectiveness", "score_discourse_effectiveness_0", 
                                               "score_discourse_effectiveness_1", 
                                                 "predictionstring", "modified", "original_predstring"]].to_csv("removed.csv")
    if verbose:
        print(str(segments_affected) + " segments have been modified/removed by pre-processing")
    pbar.close()

# ====================================================================================================== #

## Normalization
def normalize_probs(df_sub):
    prob_sum = df_sub["score_discourse_effectiveness_0"] + df_sub["score_discourse_effectiveness_1"]
    df_sub["score_discourse_effectiveness_0"] = df_sub["score_discourse_effectiveness_0"] / prob_sum
    df_sub["score_discourse_effectiveness_1"] = df_sub["score_discourse_effectiveness_1"] / prob_sum
    df_sub["score_discourse_effectiveness"] = df_sub["score_discourse_effectiveness"] / prob_sum
    normalized_df_sub = df_sub.to_csv("normalized.csv")
    
    
# ==============================================================================================
# Metric
# ==============================================================================================

## The final function, compute_scores(), takes in:
##      df_sub - a submission file of predictions
##      gt_df - a file containing the ground truths
##      ids_selected - a list of the ids

# ====================================================================================================== #

def _tp_score_effectiveness(col=""):
    def row_wise(row):
        gt_eff = row.discourse_effectiveness 
        return row[f'score_discourse_effectiveness_{gt_eff}']
    return row_wise

# ====================================================================================================== #

def calc_overlap(set_pred, set_gt,threshold=0.5):  # Note: If we are to change the score/acceptance criterion, the logic should be changed here
    """
    Calculates if the overlap between prediction and
    ground truth is enough fora potential True positive
    """
    # Length of each and intersection
    try:
        len_gt = len(set_gt)  # Computes the number of words in the ground truth segment
        len_pred = len(set_pred)  # Computes the number of words in the prediction segment
        inter = len(set_gt & set_pred)  # Computes the length of the intersection
        union = len(set_gt | set_pred)
        overlap_1 = inter / len_gt  # Check the ratio of the interseections to both ground truth and prediction
        overlap_2 = inter/ len_pred
        return overlap_1 >= threshold and overlap_2 >= threshold, inter / union # max(overlap_1,overlap_2)  # If both thresholds satisfied, return True. Criteria for best overlap
    except:  # at least one of the input is NaN
        return False, 0  # Otherwise, discard any overlap and return 0.

# ====================================================================================================== #

def score_feedback_comp_micro(pred_df, gt_df, discourse_type, threshold=0.5, weight_tp_segment=0.5, return_segments=True):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df.loc[gt_df['discourse_type'] == discourse_type, 
                      ['id', 'predictionstring','discourse_effectiveness', "discourse_type"]].reset_index(drop=True)  # Get all the ground truth segments of a given discourse type along with their effectiveness label (0, 1, 2)
    # Kept discourse_type for more readability vis-a-vis return_segment, but this is not necessary for the computation per se. 
        
    pred_df = pred_df.loc[pred_df['discourse_type'] == discourse_type,
                      ['id', 'predictionstring','score_discourse_effectiveness_0',
                                               'score_discourse_effectiveness_1', 'discourse_type']].reset_index(drop=True)  # Get the prediction segments along with the predicted effectiveness distributions (three positive scalars summing to 1)
    
    pred_df['pred_id'] = pred_df.index  # Set prediction and ground truth IDs as dataframe index
    gt_df['gt_id'] = gt_df.index
    pred_df['predictionstring'] = [set([int(x) for x in pred.split(' ')]) for pred in pred_df['predictionstring']]  # Split the predictionstring of both ground truth and prediction segments up into a list of word indices for later set operations
    gt_df['predictionstring'] = [set([int(x) for x in pred.split(' ')]) for pred in gt_df['predictionstring']]  # Turned into integers for better CSV readability in case of use of return_segments.
    
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on='id',
                           right_on='id',
                           how='outer',
                           suffixes=('_pred','_gt')
                          )  # Outer join all prediction and ground truth segments to perform pair-wise comparison
    
    overlaps = [calc_overlap(*args,threshold=threshold) for args in zip(joined.predictionstring_pred, 
                                                     joined.predictionstring_gt)]  # Calculate the overlap of each segment - prediction pair using the calc_overlap function above (This returns a Boolean for threshold acceptance and the overlap value (currently max of two overlaps))
    
    joined['discourse_effectiveness'] = joined['discourse_effectiveness'].fillna(0).astype(int)
    joined['overlaps'] = np.asarray([x[0] for x in overlaps])*1  # Convert the overlap Boolean into a 0/1 value and assign the result as a  column in the outer join dataframe.
    joined['overlaps_scores'] = np.asarray([x[1] for x in overlaps])*1  # Add the actual overlap scores to the join dataframe
    joined['effectiveness_TP_score'] = joined.apply(_tp_score_effectiveness(' '),axis=1)  # Call a helper function to fetch the predicted probability, for each prediction, on the correct effectiveness class of the ground truth segment, and add this as a column
    # One - Eff not really needed
    
    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    # we don't need to compute the match to compute the score
    
    joined = joined.sort_values(["overlaps",'overlaps_scores'],ascending=False).reset_index(drop=True).groupby('gt_id').head(1)  # 1) Sort predictions in descending order by their overlap scores, 2) Group them by the ground truth segment they target, 3) only keep the best prediction for each segment. 
    TP = joined[joined.overlaps==1]['gt_id'].nunique() # Then, from the best predictions per group which satisfy the overlap threshold, count the number of satisfied ground truth segments. This returns the number of true positives
    
            
    TP_weighted = weight_tp_segment * TP + (1-weight_tp_segment)*(joined[joined.overlaps==1]['effectiveness_TP_score'].sum())  # In our weighted setup, we don't simply want the number of such segments. We want to assign a partial fraction of the true positive to effectiveness, and thus we compute an effectiveness component.
    
    # NEW: Introduce the double weighting scheme
    TP_overlap_coeff = joined[joined.overlaps==1]['overlaps_scores'].sum()
    TP_double_weighted =  weight_tp_segment * TP_overlap_coeff + (1-weight_tp_segment)*(joined[joined.overlaps==1]['effectiveness_TP_score'].sum())
    
    # Therefore, in effect TP_weighted <= TP.
    # Consistently with the original metrics, we assign the lost fraction due to the effectiveness coefficient as a false negative. Therefore, TP + FN remains the same in both weighted and unweighted cases.
    # To avoid rewarding the model for effectiveness prediction in non-overlapping segments, we do NOT apply the weighted system for rejected segments. Otherwise, a wholly non-overlapping segment (a false positive) with an otherwise perfect match to any segment goes unpunished.
    # Hence, FP remains the same as in the unweighted metric. and thus TP + FP weighted drops by how much TP drops only.
    
    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    TPandFP = len(pred_df)
    TPandFN = len(gt_df)
    
    TP_gap = TP - TP_weighted  # Compute the amount of TP lost due to the weighting of effectiveness.
    
    # CHANGE:
    # print("Discourse Type: " + str(discourse_type))
    # x = joined[(joined.overlaps==1) & (joined.discourse_effectiveness == 1)]  # probability given to matching gt segments
    # print(x['effectiveness_TP_score'].mean())
                    
    # print(1 - (TP_gap / ((1 - weight_tp_segment) * TP)) * (len(gt_df) / len(gt_df[gt_df["discourse_effectiveness"]==1])))
    
    TPandFP_weighted = TPandFP - TP_gap  # Append it to the unweighted TP+FP component
    
    # NEW: Double Coeff Weighting
    TP_double_gap = TP - TP_double_weighted
    TPandFP_double_weighted = TPandFP - TP_double_gap
    
    #Calculate microf1
    f1_score_fb1 = 2*TP / (TPandFP + TPandFN)  # Compute F1 scores in the unweighted case
    new_f1_score = 2*TP_weighted / (TPandFP_weighted + TPandFN)  # Analogously compute the weighted case. 
    # NEW
    double_weighted_f1_score = 2*TP_double_weighted / (TPandFP_double_weighted + TPandFN)
    '''
    Notice that, given an unweighted F1 score = a/b in the unweighted case, the weighted F1 score would equal (a - 2g)/() b - g), where 0 <= g <= a/2 is the TP loss due to the efficiency/overlap components. 
    Observe that (a - 2g) / (b - g) <= (a / b) for 0 <= g <= a/2, and thus weighted F1 <= normal F1 .
    '''
    if return_segments:
        TP_pred_ids = joined[joined.overlaps==1]['pred_id'].unique()
        TP_gt_ids = joined[joined.overlaps==1]['gt_id'].unique()
        nb_pred_ids = len(TP_pred_ids)
        nb_gt_ids = len(TP_gt_ids)  # With overlaps and threshold <=0.5, it's possible that nb_pred >= nb_gt (i.e., one prediction is the best match to more than one GT)
        
        all_pred_ids = pred_df['pred_id'] # .unique()  # No need for .unique()
        FP_pred_ids = np.setdiff1d(all_pred_ids, TP_pred_ids)
        
        all_gt_ids = gt_df['gt_id']  # .unique()  # No need for .unique()
        FN_gt_ids = np.setdiff1d(all_gt_ids, TP_gt_ids)
        
        if nb_pred_ids != nb_gt_ids:  # Analysis of ambiguous metric edge case (implementation-specific)
            # Find the predictions who match more than 1 GT
            finding_overmatch = joined[joined.overlaps==1].groupby('pred_id').filter(lambda x: len(x) > 1)  # Now group by prediction ID and only print out overmatching
            print(finding_overmatch[["id", "predictionstring_gt", "predictionstring_pred"]])
        
        TP_preds = pred_df.iloc[TP_pred_ids, :]  # No need to return the index, it's arbitrary anyway
        TP_gts = gt_df.iloc[TP_gt_ids, :]
        FP_preds = pred_df.iloc[FP_pred_ids, :]
        FN_gts = gt_df.iloc[FN_gt_ids, :]
        
        return f1_score_fb1,new_f1_score, double_weighted_f1_score, TP_preds, TP_gts, FP_preds, FN_gts
    else:    
        return f1_score_fb1,new_f1_score, double_weighted_f1_score

# ====================================================================================================== #

def score_feedback_comp(pred_df, gt_df, threshold=0.5, weight_tp_segment=0.5, return_class_scores=False, return_segments=False):
    class_scores_fb1 = {}
    new_class_scores = {}
    double_weighted_class_scores = {}  # NEW
    
    tp_preds = []  # To store the segments in case they are requested
    tp_gts = []
    fp_preds = []
    fn_gts = []
    
    for discourse_type in gt_df.discourse_type.unique():
        if return_segments:
            s_fb1,s, s_double, tp_pred, tp_gt, fp_pred, fn_gt = score_feedback_comp_micro(pred_df, gt_df, discourse_type,threshold,
                                                                                          weight_tp_segment, return_segments=return_segments)
            tp_preds.append(tp_pred)
            tp_gts.append(tp_gt)
            fp_preds.append(fp_pred)
            fn_gts.append(fn_gt)
            # TODO: Add logic to get FP, FN, render files (with a custom file name), and save them.
        else:
            s_fb1,s, s_double = score_feedback_comp_micro(pred_df, gt_df, discourse_type,threshold, weight_tp_segment, return_segments=return_segments)
            
        class_scores_fb1[discourse_type] = s_fb1
        new_class_scores[discourse_type] = s
        double_weighted_class_scores[discourse_type] = s_double  # NEW
    
    # To get breakdown. CHANGE
    # print(class_scores_fb1)
    # print(new_class_scores)
    # print(double_weighted_class_scores)
    
    f1_fb1 = np.mean([v for v in class_scores_fb1.values()])
    new_f1 = np.mean([v for v in new_class_scores.values()])
    double_weighted_f1 = np.mean([v for v in double_weighted_class_scores.values()])  # NEW
    if return_segments:
        tp_preds = pd.concat(tp_preds).to_csv("TP_pred.csv")
        tp_gts = pd.concat(tp_gts).to_csv("TP_gt.csv")
        fp_preds = pd.concat(fp_preds).to_csv("FP_pred.csv")
        fn_gts = pd.concat(fn_gts).to_csv("FN_gt.csv")
        
        # print("FP:"+str(len(fp_preds)))
        # print("TP:"+str(len(tp_preds)))
        # print("FN:"+str(len(fn_gts)))
        
    if return_class_scores:
        return f1_fb1, class_scores_fb1, new_f1, new_class_scores, double_weighted_class_scores  # NEW
    return f1_fb1, new_f1, double_weighted_f1  # NEW

# ====================================================================================================== #

def vanilla_micro(pred_df, gt_df, discourse_type, threshold=0.5,weight_tp_segment=0.5):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df.loc[gt_df['discourse_type'] == discourse_type, 
                      ['id', 'predictionstring','discourse_effectiveness']].reset_index(drop=True)  # Get all the ground truth segments of a given discourse type along with their effectiveness label (0, 1, 2)
    pred_df = pred_df.loc[pred_df['discourse_type'] == discourse_type,
                      ['id', 'predictionstring','score_discourse_effectiveness_0',
                                               'score_discourse_effectiveness_1']].reset_index(drop=True)  # Get the prediction segments along with the predicted effectiveness distributions (three positive scalars summing to 1)
    pred_df['pred_id'] = pred_df.index  # Set prediction and ground truth IDs as dataframe index
    gt_df['gt_id'] = gt_df.index
    pred_df['predictionstring'] = [set(pred.split(' ')) for pred in pred_df['predictionstring']]  # Split the predictionstring of both ground truth and prediction segments up into a list of word indices for later set operations
    gt_df['predictionstring'] = [set(pred.split(' ')) for pred in gt_df['predictionstring']]
    
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on='id',
                           right_on='id',
                           how='outer',
                           suffixes=('_pred','_gt')
                          )  # Outer join all prediction and ground truth segments to perform pair-wise comparison
    
    overlaps = [calc_overlap(*args,threshold=threshold) for args in zip(joined.predictionstring_pred, 
                                                     joined.predictionstring_gt)]  # Calculate the overlap of each segment - prediction pair using the calc_overlap function above (This returns a Boolean for threshold acceptance and the overlap value (currently max of two overlaps))
    
    joined['discourse_effectiveness'] = joined['discourse_effectiveness'].fillna(0).astype(int)
    joined['overlaps'] = np.asarray([x[0] for x in overlaps])*1  # Convert the overlap Boolean into a 0/1 value and assign the result as a  column in the outer join dataframe.
    joined['overlaps_scores'] = np.asarray([x[1] for x in overlaps])*1  # Add the actual overlap scores to the join dataframe
    joined['effectiveness_TP_score'] = joined.apply(_tp_score_effectiveness(' '),axis=1)  # Call a helper function to fetch the predicted probability, for each prediction, on the correct effectiveness class of the ground truth segment, and add this as a column
    # One - Eff not really needed
    
    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    # we don't need to compute the match to compute the score
    
    joined = joined.sort_values(["overlaps",'overlaps_scores'],ascending=False).reset_index(drop=True).groupby('gt_id').head(1)  # 1) Sort predictions in descending order by their overlap scores, 2) Group them by the ground truth segment they target, 3) only keep the best prediction for each segment. 
    
    TP = joined[joined.overlaps==1]['gt_id'].nunique() # Then, from the best predictions per group which satisfy the overlap threshold, count the number of satisfied ground truth segments. This returns the number of true positives
    
    # Therefore, in effect TP_weighted <= TP.
    # Consistently with the original metrics, we assign the lost fraction due to the effectiveness coefficient as a false negative. Therefore, TP + FN remains the same in both weighted and unweighted cases.
    # To avoid rewarding the model for effectiveness prediction in non-overlapping segments, we do NOT apply the weighted system for rejected segments. Otherwise, a wholly non-overlapping segment (a false positive) with an otherwise perfect match to any segment goes unpunished.
    # Hence, FP remains the same as in the unweighted metric. and thus TP + FP weighted drops by how much TP drops only.
    
    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    TPandFP = len(pred_df)
    TPandFN = len(gt_df)
    
    
    df_eff = joined[joined.overlaps==1].reset_index()
    df_eff['discourse_effectiveness_pred'] = df_eff[['score_discourse_effectiveness_0',
                                               'score_discourse_effectiveness_1']].idxmax(axis=1).str.split('_',expand=True)[3].astype(int)
    
    TP_effec = (df_eff['discourse_effectiveness']==df_eff['discourse_effectiveness_pred']).sum()
    
    return TP,TPandFP,TPandFN,TP_effec

# # ====================================================================================================== #

def vanilla_accuracy(pred_df, gt_df, threshold=0.5, weight_tp_segment=0.5, return_class_scores=False, verbose=False):  # NOTE: This is strange
    TP,TPandFP,TPandFN,TP_EFFEC = [],[],[],[]
    for discourse_type in gt_df.discourse_type.unique():
        tp,tp_fp,tp_fn,tp_eff = vanilla_micro(pred_df, gt_df, discourse_type,threshold,weight_tp_segment)
        TP.append(tp)
        TPandFP.append(tp_fp)
        TPandFN.append(tp_fn)
        TP_EFFEC.append(tp_eff)
        
    
    TP = sum(TP)
    TPandFP = sum(TPandFP)
    TPandFN = sum(TPandFN)
    TP_EFFEC = sum(TP_EFFEC)
    
    acc_type = (TP/TPandFN)*100
    acc_eff = (TP_EFFEC/TP)*100
    
    print(f"{acc_type:.2f}% of all discourse elements were accurately predicted")
    print(f"{acc_eff:.2f}% of these elements were also accurately predicted for effectiveness")
    print(f"TP={TP}, TP_EFFEC={TP_EFFEC}")

    
# # ====================================================================================================== #

def compute_scores(df_sub, gt_df, ids_selected, threshold=0.5, return_segments=False, return_scores=False, compute_acc=False, verbose=False):
    if "Lead" in df_sub['discourse_type'].unique():
        df_sub['discourse_type'] = df_sub['discourse_type'].map(TYPE2LABEL)
    

    if return_scores:
        ## TODO 
        pass
    
    else:
        fb1_score, new_f1_score, double_weighted_f1_score = score_feedback_comp(df_sub[df_sub.id.isin(ids_selected)].reset_index(drop=True),
                        gt_df[gt_df.id.isin(ids_selected)].reset_index(drop=True),
                        threshold=threshold,
                        weight_tp_segment=0.5,
                        return_class_scores=return_scores, return_segments=return_segments)
        
        if verbose: print(f"F1 FB1 : {fb1_score:.4f} , Single Weighted F1 : {new_f1_score:.4f}, Double Weighted F1 : {double_weighted_f1_score:.4f}\n")  # NEW
        
    if compute_acc:
        vanilla_accuracy(df_sub[df_sub.id.isin(ids_selected)].reset_index(drop=True),
                        gt_df[gt_df.id.isin(ids_selected)].reset_index(drop=True),
                        threshold=threshold,
                        weight_tp_segment=0.5,
                        return_class_scores=return_scores
                        )
    
    return dict(
        F1_FB1 = fb1_score,
        F1_single_w = new_f1_score,
        F1_double_w = double_weighted_f1_score,
    )
    
