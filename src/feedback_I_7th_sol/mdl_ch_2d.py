from torch import nn
import torch
from transformers import LongformerModel, LongformerConfig


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg

        if self.cfg.offline_inference:
            config = LongformerConfig.from_pretrained(cfg.backbone)
            self.backbone = LongformerModel._from_config(config, add_pooling_layer=False)
        else:
            self.backbone = LongformerModel.from_pretrained(cfg.backbone, add_pooling_layer=False)
            if self.cfg.gradient_checkpointing:
                self.backbone.gradient_checkpointing_enable()
        self.n_classes = len(self.cfg.discourse_types)

        hidden_size = self.backbone.config.hidden_size
        self.classification_head = nn.Linear(hidden_size, self.n_classes)
        self.sep_head = nn.Linear(hidden_size, 1)

        self.class_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.sep_loss_fn = nn.BCEWithLogitsLoss()

        self.return_preds = True

    def forward(self, batch):

        x = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[
            0
        ]  # torch.Size([4, 1656, 768])
        class_preds = self.classification_head(x)  # torch.Size([4, 1656, 9])
        sep_preds = self.sep_head(x)  # torch.Size([4, 1656, 1])

        output = {}
        if "target" in batch.keys():
            class_loss = self.class_loss_fn(class_preds.permute(0, 2, 1), batch["target"])
            sep_loss = self.sep_loss_fn(sep_preds.squeeze(2), batch["target2"].float())

            cl = []
            for i in range(class_preds.shape[0]):
                sep_indices = torch.where(batch["target2"][i] > 0.5)[0].cpu()
                loss_spans = torch.tensor_split(class_loss[i], sep_indices)
                cl += [s.mean() for s in loss_spans]
            class_loss = torch.stack(cl).mean()

            loss = self.cfg.loss_weights[0] * class_loss + self.cfg.loss_weights[1] * sep_loss
            output.update({"loss": loss, "class_loss": class_loss, "sep_loss": sep_loss})

        if (not self.training) & self.return_preds:

            len_pad = self.cfg.max_length_test - batch["input_ids"].shape[1]
            # print(len_pad)
            output["class_preds"] = torch.nn.functional.pad(
                class_preds, (0, 0, 0, len_pad), mode="constant", value=0
            )
            output["sep_preds"] = torch.nn.functional.pad(
                sep_preds, (0, 0, 0, len_pad), mode="constant", value=0
            )
            output["wordposition"] = torch.nn.functional.pad(
                batch["wordposition"], (0, len_pad), mode="constant", value=-1
            )
            output["attention_mask"] = torch.nn.functional.pad(
                batch["attention_mask"], (0, len_pad), mode="constant", value=0
            )

        return output
