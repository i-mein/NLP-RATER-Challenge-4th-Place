import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm


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


def post_process_pipeline(cfg, val_data, val_df):

    print("postprocessing")
    class_preds = val_data["class_preds"].float()  # .cpu()
    sep_preds = val_data["sep_preds"].float().sigmoid()  # .cpu()
    wordposition = val_data["wordposition"]  # .cpu()

    val_ids = val_df.id.unique()
    nms_kernel = 5
    nms_padding = (nms_kernel - 1) // 2

    predictionstring = []

    for i, val_id in tqdm(enumerate(val_ids)):
        p = class_preds[i].softmax(1)

        x = sep_preds[i]
        x_pooled = F.max_pool1d(
            x.permute(1, 0), kernel_size=nms_kernel, stride=1, padding=nms_padding
        ).permute(1, 0)
        x[x != x_pooled] = 0
        sep_indices = torch.where(x[:, 0] > 0.35)[0].cpu()
        class_spans = torch.tensor_split(p, sep_indices)
        word_indices = torch.tensor_split(wordposition[i], sep_indices)

        for class_span, word_inds in zip(class_spans, word_indices):
            mask = class_span.argmax(1) < 7
            masked_class_span = class_span[mask]
            if len(masked_class_span) > 0:
                class_id = class_span[mask].mean(0).argmax()
            else:
                class_id = 7
            pred = [
                val_ids[i],
                cfg.discourse_types[class_id],
                " ".join([str(w.item()) for w in torch.unique(word_inds[mask]).cpu() if w != -1]),
            ]
            predictionstring += [pred]

    oof = pd.DataFrame(predictionstring)
    oof.columns = ["id", "class", "predictionstring"]

    # join classes
    for class_ in ["Lead", "Position", "Concluding Statement"]:
        oof = join_class(oof, class_)

    return oof.copy()
