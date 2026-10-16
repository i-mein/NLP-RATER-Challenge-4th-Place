import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


DEBUG = True


def calc_overlap(set_pred, set_gt):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    from: https://www.kaggle.com/cpmpml/faster-metric-computation
    """
    # Length of each and intersection
    try:
        len_gt = len(set_gt)
        len_pred = len(set_pred)
        inter = len(set_gt & set_pred)
        overlap_1 = inter / len_gt
        overlap_2 = inter / len_pred
        return (overlap_1, overlap_2)
    except:  # at least one of the input is NaN
        return (0, 0)


def score_feedback_comp_micro(pred_df, gt_df, discourse_type, verbose=False):
    """
    A function that scores for the kaggle
        Student Writing Competition
        from: https://www.kaggle.com/cpmpml/faster-metric-computation

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df.loc[
        gt_df["discourse_type"] == discourse_type, ["id", "predictionstring"]
    ].reset_index(drop=True)
    pred_df = pred_df.loc[
        pred_df["class"] == discourse_type, ["id", "predictionstring"]
    ].reset_index(drop=True)
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    pred_df["predictionstring"] = [set(pred.split(" ")) for pred in pred_df["predictionstring"]]
    gt_df["predictionstring"] = [set(pred.split(" ")) for pred in gt_df["predictionstring"]]

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df, left_on="id", right_on="id", how="outer", suffixes=("_pred", "_gt")
    )
    overlaps = [
        calc_overlap(*args)
        for args in zip(joined.predictionstring_pred, joined.predictionstring_gt)
    ]

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["potential_TP"] = [(overlap[0] >= 0.5 and overlap[1] >= 0.5) for overlap in overlaps]
    joined["max_overlap"] = [max(*overlap) for overlap in overlaps]
    joined_tp = joined.query("potential_TP").reset_index(drop=True)
    tp_pred_ids = (
        joined_tp.sort_values("max_overlap", ascending=False)
        .groupby(["id", "gt_id"])["pred_id"]
        .first()
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = set(joined["pred_id"].unique()) - set(tp_pred_ids)

    matched_gt_ids = joined_tp["gt_id"].unique()
    unmatched_gt_ids = set(joined["gt_id"].unique()) - set(matched_gt_ids)

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    if verbose:
        print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


def score_feedback_comp(pred_df, gt_df, return_class_scores=False, verbose=False):
    # from: https://www.kaggle.com/cpmpml/faster-metric-computation
    class_scores = {}
    for discourse_type in gt_df.discourse_type.unique():
        class_score = score_feedback_comp_micro(pred_df, gt_df, discourse_type, verbose=verbose)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1


def calc_metric(cfg, preds, val_df, pre="val", logger=None):
    if isinstance(preds, dict):
        pred_df = preds["pred_df"]
    else:
        pred_df = preds.copy()

    # COMPUTE F1 SCORE
    f1 = []
    CLASSES = val_df["discourse_type"].unique()
    
    for class_ in CLASSES:
        pred_df_oneclass = pred_df.loc[pred_df["class"] == class_].copy()
        gt_df_oneclass = val_df.loc[val_df["discourse_type"] == class_].copy()
        f1_score_per_class = score_feedback_comp(pred_df_oneclass, gt_df_oneclass)

        logger.info(f"{pre} score for {class_}: {f1_score_per_class:.6}")

        f1.append(f1_score_per_class)

    logger.info(f"{pre} Overall score: {np.mean(f1):.6}")
    avg_f1 = np.mean(f1)

    # Calculate additional metrics (assume perfect classifier or assume perfect seperator)
    if isinstance(preds, dict):
        pred_df = preds["perf_class_df"]
        # COMPUTE F1 SCORE perfect classifier
        f1 = []
        for class_ in CLASSES:
            pred_df_oneclass = pred_df.loc[pred_df["class"] == class_].copy()
            gt_df_oneclass = val_df.loc[val_df["discourse_type"] == class_].copy()
            f1_score_per_class = score_feedback_comp(pred_df_oneclass, gt_df_oneclass)

            logger.info(f"{pre} perf_class_score for {class_}: {f1_score_per_class:.6}")

            f1.append(f1_score_per_class)

        logger.info(f"{pre} Overall perf_class_score: {np.mean(f1):.6}")

        pred_df = preds["perf_sep_df"]
        # COMPUTE F1 SCORE perfect seperator
        f1 = []
        for class_ in CLASSES:
            pred_df_oneclass = pred_df.loc[pred_df["class"] == class_].copy()
            gt_df_oneclass = val_df.loc[val_df["discourse_type"] == class_].copy()
            f1_score_per_class = score_feedback_comp(pred_df_oneclass, gt_df_oneclass)

            logger.info(f"{pre} perf_sep_score for {class_}: {f1_score_per_class:.6}")

            f1.append(f1_score_per_class)

        logger.info(f"{pre} Overall perf_sep_score: {np.mean(f1):.6}")
    ######
    return avg_f1 #f1 

### RATER 

def remove_overlaps(segments):
    """
    Removes overlaps in predicted segments.

    Args:
    segments (list of tuples): Each tuple represents a segment and contains two elements - the start and end of the segment.

    Returns:
    list of tuples: The processed segments with overlaps removed.
    """
    # Sort segments based on their starting position
    segments.sort(key=lambda x: x[0])

    used_words = set()
    processed_segments = []

    for segment in segments:
        start, end = segment

        # Check for overlap with used words
        overlap = set(range(start, end+1)) & used_words

        # If there's an overlap, remove the used words from the current segment
        if overlap:
            new_segment = set(range(start, end+1)) - overlap

            # If the resulting segment is valid, add it to the list of processed segments
            if len(new_segment) > 1:
                processed_segments.append((min(new_segment), max(new_segment)))

            # Add the new words to the set of used words
            used_words.update(new_segment)
        else:
            # If there's no overlap, add the segment as is to the list of processed segments
            processed_segments.append(segment)

            # Add the words in the current segment to the set of used words
            used_words.update(range(start, end+1))

    return processed_segments


def calculate_score_new(predictions, ground_truth, w, verbose=False):
    """
    Calculates the score for predicted segments based on overlap quality and effectiveness predictions.

    Args:
    predictions (list of tuples): Each tuple represents a predicted segment and contains three elements - the start and end of the segment, and the predicted effectiveness.
    ground_truth (list of tuples): Each tuple represents a ground truth segment and contains three elements - the start and end of the segment, and the actual effectiveness.
    w (float): The segmentation weight.

    Returns:
    float: The overall macro-F1 score for the segmentation task.
    """
    TP = 0
    FP = 0
    FN = 0

    for prediction in predictions:
        start_p, end_p, effectiveness_p = prediction
        overlap = [gt for gt in ground_truth if set(range(start_p, end_p+1)) & set(range(gt[0], gt[1]+1))]

        if overlap:
            # Calculate the IoU score
            IoU = len(set(range(start_p, end_p+1)) & set(range(overlap[0][0], overlap[0][1]+1))) / len(set(range(start_p, end_p+1)) | set(range(overlap[0][0], overlap[0][1]+1)))

            # Check if the IoU score is above the threshold
            if IoU > 0.51:
                # Calculate the true positive fraction
                TP += w * IoU + (1 - w) * effectiveness_p

                # Calculate the false negative proportion
                FN += (1 - w) * (1 - effectiveness_p)
            else:
                FP += 1
        else:
            FP += 1

    for gt in ground_truth:
        start_gt, end_gt, effectiveness_gt = gt
        overlap = [p for p in predictions if set(range(start_gt, end_gt+1)) & set(range(p[0], p[1]+1))]

        if not overlap:
            FN += 1

    # Calculate the precision, recall, and F1 score
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    if verbose:
        print(f"TP: {TP}, FP: {FP}, FN: {FN}")
        print(f"pre: {precision}, rec: {recall}")

    return F1





####### FP1 metric modified 

def calc_overlap(row):
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    return inter/max(len_gt,len_pred)


def get_f1_score(pred_df, test_df, log=print, slient=True):
    
    test_pred = pred_df.copy()
    
    if not slient:
        log('test_pred.shape:',test_pred.shape,'\ttest_df.shape:',test_df.shape,)
        log('pred class:\n',test_pred['class'].value_counts())
        log('true class:\n',test_df['discourse_type'].value_counts())
    

    test_pred['class'] = test_pred['discourse_type']

    f1s = []
    aucs = []
    logloss_scores = []
    total_scores = []
    for c in sorted(test_pred['class'].unique()):
        pred_df = test_pred.loc[test_pred['class']==c].copy()
        gt_df = test_df.loc[test_df['discourse_type']==c].copy()

        gt_df = gt_df[['id','discourse_type','predictionstring','discourse_effectiveness_int']].reset_index(drop=True).copy()
        pred_df = pred_df[['id','class','predictionstring', 'score_eff']].reset_index(drop=True).copy()
        
        pred_df['pred_id'] = pred_df.index
        gt_df['gt_id'] = gt_df.index
        
        # Step 1. all ground truths and predictions for a given class are compared.
        joined = pred_df.merge(gt_df,
                               left_on=['id','class'],
                               right_on=['id','discourse_type'],
                               how='outer',
                               suffixes=('_pred','_gt')
                              )
        joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
        joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

        # print(joined.head())
        joined['min_overlaps'] = joined.apply(calc_overlap, axis=1)
        joined['potential_TP'] = (joined['min_overlaps'] >= 0.5)

        matched_pred_ids = joined.query('potential_TP')['pred_id'].unique()
        fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in matched_pred_ids]

        matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
        fn_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

        # Get numbers of each type
        TP = len(matched_gt_ids)
        FP = len(fp_pred_ids)
        FN = len(fn_gt_ids)
        #calc microf1
        f1_score = TP / (TP + 0.5*(FP+FN))
        

        ###########
        ll_score = log_loss(
            joined[joined.potential_TP==True]['discourse_effectiveness_int'],
            joined[joined.potential_TP==True]['score_eff'],
            labels=[0,1]
        )
        logloss_scores.append(ll_score)

        auc_sc = roc_auc_score(
            joined[joined.potential_TP==True]['discourse_effectiveness_int'],
            joined[joined.potential_TP==True]['score_eff']
        )
        aucs.append(auc_sc)

        # #### try this
        # joined3 = joined[['potential_TP', 'discourse_effectiveness_int', 'score_eff']].dropna()
        # auc_sc2 = roc_auc_score(joined3['discourse_effectiveness_int'], joined3['score_eff'],)

        tot_score = 0.5*f1_score + 0.5*ll_score
        total_scores.append(tot_score)

        if not slient:
            print(f'{c:<20} f1 score:\t{f1_score}')
            print(f'{c:<20} logloss (TP) score:{ll_score:.4f} \t auc (TP) score: {auc_sc:.4f}')
            print(f'{c:<20} total score:\t{tot_score}')
            print()
        f1s.append(f1_score)
    
    print('\nOverall f1 score',  np.mean(f1s))
    print('Overall total score:\t', np.mean(total_scores))
    print()
    print('Overall logloss (TP) score:\t', np.mean(logloss_scores))
    print('Overall auc (TP) score:\t', np.mean(aucs))
    return np.mean(f1s)






