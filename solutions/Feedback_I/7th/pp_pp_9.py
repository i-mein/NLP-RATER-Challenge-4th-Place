import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

map_clip_05p = {
    "Lead": 3,
    "Position": 2,
    "Evidence": 15,  # 11 may be better for evidence
    "Claim": 1,
    "Concluding Statement": 5,
    "Counterclaim": 3,
    "Rebuttal": 3,
}

CLASS_THRES = [0.2, 0.3, 0.10, 0.51, 0.3, 0.25, 0.10, 0.3, 0.3]
CLASS_THRES2 = [0.45, 0.9, 0.35, 0.80, 0.70, 0.60, 0.35, 0.9, 0.9]


def join_class(df, class_):
    dfa = df[df["class"] != class_][["id", "class", "predictionstring"]].copy()
    dfb = df[df["class"] == class_][["id", "predictionstring"]].copy()

    dfb["predictionstring"] = (
        dfb[["id", "predictionstring"]]
        .groupby(["id"])["predictionstring"]
        .transform(lambda x: " ".join(x))
    )
    dfb = dfb.drop_duplicates().copy()
    dfb["class"] = class_

    df_pp = pd.concat([dfa, dfb])
    return df_pp


def jn(pst, start, end):
    return " ".join([str(x) for x in pst[start:end] if x != -1])


def link_evidence(oof):
    thresh = 1
    idu = oof["id"].unique()
    # idc = idu[1]
    eoof = oof[oof["class"] == "Evidence"]
    neoof = oof[oof["class"] != "Evidence"]
    for thresh2 in range(26, 27, 1):
        retval = []
        for idv in idu:
            for c in [
                "Lead",
                "Position",
                "Evidence",
                "Claim",
                "Concluding Statement",
                "Counterclaim",
                "Rebuttal",
            ]:
                q = eoof[(eoof["id"] == idv) & (eoof["class"] == c)]
                if len(q) == 0:
                    continue
                pst = []
                for i, r in q.iterrows():
                    pst = pst + [-1] + [int(x) for x in r["predictionstring"].split()]
                start = 1
                end = 1
                for i in range(2, len(pst)):
                    cur = pst[i]
                    end = i
                    # if pst[start] == 205:
                    #   print(cur, pst[start], cur - pst[start])
                    if (cur == -1 and c != "Evidence") or (
                        (cur == -1)
                        and (
                            (pst[i + 1] > pst[end - 1] + thresh)
                            or (pst[i + 1] - pst[start] > thresh2)
                        )
                    ):
                        retval.append((idv, c, jn(pst, start, end)))
                        start = i + 1
                v = (idv, c, jn(pst, start, end + 1))
                # print(v)
                retval.append(v)
        roof = pd.DataFrame(retval, columns=["id", "class", "predictionstring"])
        roof = roof.merge(neoof, how="outer")
        return roof


def post_process_pipeline(cfg, val_data, val_df):

    print("postprocessing")
    class_preds = val_data["class_preds"].float().cpu()
    sep_preds = val_data["sep_preds"].float().sigmoid().cpu()
    wordposition = val_data["wordposition"].cpu()
    attention_mask = val_data["attention_mask"].cpu()
    target_class = val_data["target_class"].cpu()
    target_sep = val_data["target_sep"].cpu()

    class_preds[:, :, 8][attention_mask == 0] = 5000

    val_ids = val_df.id.unique()
    nms_kernel = 5
    nms_padding = (nms_kernel - 1) // 2

    predictionstring = []
    predictionstring_perf_sep = []
    predictionstring_perf_class = []
    predictionstring_perf_both = []

    for i, val_id in tqdm(enumerate(val_ids)):
        p = class_preds[i].softmax(1)
        p_sig = class_preds[i].sigmoid()
        p2 = target_class[i]

        x = sep_preds[i]
        x2 = target_sep[i]
        x_pooled = F.max_pool1d(
            x.permute(1, 0), kernel_size=nms_kernel, stride=1, padding=nms_padding
        ).permute(1, 0)
        x[x != x_pooled] = 0
        sep_indices = torch.where(x[:, 0] > 0.40)[0].cpu()
        perf_sep_indices = torch.where(x2 == 1)[0].cpu()
        class_spans = torch.tensor_split(p, sep_indices)
        class_spans_sig = torch.tensor_split(p_sig, sep_indices)
        class_spans_perf_sep = torch.tensor_split(p, perf_sep_indices)
        class_spans_perf_class = torch.tensor_split(p2, sep_indices)
        class_spans_perf_perf = torch.tensor_split(p2, perf_sep_indices)
        word_indices = torch.tensor_split(wordposition[i], sep_indices)
        word_indices_perf_sep = torch.tensor_split(wordposition[i], perf_sep_indices)

        for class_span, class_span_sig, word_inds in zip(
            class_spans, class_spans_sig, word_indices
        ):
            mask = class_span.argmax(1) < 8
            masked_class_span = class_span[mask]
            if len(masked_class_span) > 0:
                for class_id in range(7):
                    argmax_class_id = class_span[mask].mean(0).argmax()
                    if (
                        (class_id == argmax_class_id)
                        and (class_span_sig[mask].mean(0)[class_id] > CLASS_THRES[class_id])
                    ) or (class_span_sig[mask].mean(0)[class_id] > CLASS_THRES2[class_id]):
                        pred = [
                            val_ids[i],
                            cfg.discourse_types[class_id],
                            " ".join(
                                [
                                    str(w.item())
                                    for w in torch.unique(word_inds[mask]).cpu()
                                    if w != -1
                                ]
                            ),
                        ]
                        predictionstring += [pred]

        for class_span, word_inds in zip(class_spans_perf_class, word_indices):
            mask = class_span < 8
            masked_class_span = class_span[mask]
            if len(masked_class_span) > 0:
                # print(pd.value_counts(pd.Series(masked_class_span.numpy())))
                class_id = pd.value_counts(pd.Series(masked_class_span.cpu().numpy())).index[0]
                if class_id < 7:
                    pred = [
                        val_ids[i],
                        cfg.discourse_types[class_id],
                        " ".join(
                            [str(w.item()) for w in torch.unique(word_inds[mask]).cpu() if w != -1]
                        ),
                    ]
                    predictionstring_perf_class += [pred]

        for class_span, word_inds in zip(class_spans_perf_sep, word_indices_perf_sep):
            mask = class_span.argmax(1) < 8
            masked_class_span = class_span[mask]
            if len(masked_class_span) > 0:
                class_id = class_span[mask].mean(0).argmax()
                if class_id < 7:
                    pred = [
                        val_ids[i],
                        cfg.discourse_types[class_id],
                        " ".join(
                            [str(w.item()) for w in torch.unique(word_inds[mask]).cpu() if w != -1]
                        ),
                    ]
                    predictionstring_perf_sep += [pred]

        for class_span, word_inds in zip(class_spans_perf_perf, word_indices_perf_sep):
            mask = class_span < 8
            masked_class_span = class_span[mask]
            if len(masked_class_span) > 0:
                class_id = pd.value_counts(pd.Series(masked_class_span.cpu().numpy())).index[0]
                if class_id < 7:
                    pred = [
                        val_ids[i],
                        cfg.discourse_types[class_id],
                        " ".join(
                            [str(w.item()) for w in torch.unique(word_inds[mask]).cpu() if w != -1]
                        ),
                    ]
                    predictionstring_perf_both += [pred]

    oof = pd.DataFrame(predictionstring)
    oof.columns = ["id", "class", "predictionstring"]

    perf_sep_df = pd.DataFrame(predictionstring_perf_sep)
    perf_sep_df.columns = ["id", "class", "predictionstring"]

    perf_class_df = pd.DataFrame(predictionstring_perf_class)
    perf_class_df.columns = ["id", "class", "predictionstring"]

    debug_df = pd.DataFrame(predictionstring_perf_both)
    debug_df.columns = ["id", "class", "predictionstring"]

    # join classes
    for class_ in ["Lead", "Position", "Concluding Statement"]:
        oof = join_class(oof, class_)
        perf_sep_df = join_class(perf_sep_df, class_)
        perf_class_df = join_class(perf_class_df, class_)

    CLASSES = val_df["discourse_type"].unique()

    all_pred_strings = (
        oof["predictionstring"].str.split(expand=True).fillna(0).astype(int).to_numpy()
    )
    oof["startwordpos"] = all_pred_strings[:, 0]
    oof["endwordpos"] = all_pred_strings.max(axis=1)
    oof["word_count"] = oof["endwordpos"] - oof["startwordpos"] + 1

    new_oof = []
    for c in CLASSES:
        new_oof.append(oof.loc[(oof["class"] == c) & (oof["word_count"] > map_clip_05p[c])].copy())
    oof = pd.concat(new_oof)

    oof = link_evidence(oof)

    ret = {
        "pred_df": oof.copy(),
        "perf_class_df": perf_class_df.copy(),
        "perf_sep_df": perf_sep_df.copy(),
        "debug_df": debug_df.copy(),
    }
    return ret
