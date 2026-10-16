import torch
from torch import nn
from transformers import AutoModel, AutoConfig, DebertaV2Model
import torch.utils.checkpoint
import gc, importlib


class Net(nn.Module):
    def __init__(self, cfg, use_fp1_weights=False, vb=False):
        super(Net, self).__init__()

        self.cfg = cfg
        self.vb = vb 

        if use_fp1_weights:
            fp1_weights='/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/checkpoints/cfg_ch_32c/checkpoint_last_seed453209.pth'
            cfg.pretrained_weights = fp1_weights

            Net = importlib.import_module('model_ch_8_deberta').Net
            mdl = Net(cfg)


            self.backbone = mdl.backbone
            if self.cfg.gradient_checkpointing:
                self.backbone.gradient_checkpointing_enable()

            self.classification_head = mdl.classification_head
            self.sep_head = mdl.sep_head

            del mdl 
            gc.collect()

        else:

            if self.cfg.offline_inference:
                deberta_config = AutoConfig.from_pretrained(cfg.backbone)
                deberta_config.update({"max_position_embeddings": self.cfg.max_length})
                self.backbone = DebertaV2Model(deberta_config)
            else:
                if cfg.transformer_config != "":
                    deberta_config = AutoConfig.from_pretrained(cfg.transformer_config)
                else:
                    deberta_config = AutoConfig.from_pretrained(cfg.backbone)
                deberta_config.update({"max_position_embeddings": self.cfg.max_length})
                sd = AutoModel.from_pretrained(cfg.backbone).state_dict()
                sd.update({"embeddings.position_ids": torch.arange(self.cfg.max_length).unsqueeze(0)})
                self.backbone = DebertaV2Model(deberta_config)
                self.backbone.load_state_dict(sd)
                if self.cfg.gradient_checkpointing:
                    self.backbone.gradient_checkpointing_enable()

        self.n_classes = len(self.cfg.discourse_types)

        hidden_size = self.backbone.config.hidden_size
        self.dropout_layer = nn.Dropout(p=cfg.dropout)

        # self.classification_head = nn.Linear(hidden_size, self.n_classes)
        # self.sep_head = nn.Linear(hidden_size, 1)
        ## self.flow_head = nn.Linear(hidden_size, 1)
        self.head_eff = nn.Linear(hidden_size, 1)

        class_weights = (
            torch.tensor(cfg.class_loss_weights) / torch.tensor(cfg.class_loss_weights).mean()
        )

        self.class_loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        self.single_class_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.sep_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_fn_bce = nn.BCEWithLogitsLoss()
        # self.flow_loss_fn = nn.MSELoss(reduction="none")

        self.return_preds = False
        self.class_bias = torch.tensor(cfg.class_bias).to(cfg.device) #.cuda()

    def forward(self, batch):

        # torch.Size([4, 1656, 768])
        x = self.backbone(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state

        x = self.dropout_layer(x)

        class_preds = self.classification_head(x)  # torch.Size([4, 1656, 9])
        class_preds += self.class_bias.unsqueeze(0).unsqueeze(0)

        sep_preds = self.sep_head(x)  # torch.Size([2, 1300, 1])
        sep_preds += self.cfg.sep_bias

        # pool x along axis 1
        x_pool = x.mean(axis=1)
        logit_eff = self.head_eff(x_pool)

        output = {}
        if "target" in batch.keys():
            class_loss = self.class_loss_fn(class_preds.permute(0, 2, 1), batch["target"])
            sep_loss = self.sep_loss_fn(sep_preds, batch["target2"][:, :, None].float()).mean(2)

            class_loss *= batch["attention_mask"]
            sep_loss *= batch["attention_mask"]
            #             flow_loss *= batch["attention_mask"]

            class_loss = class_loss.sum() / batch["attention_mask"].sum()
            sep_loss = sep_loss.sum() / batch["attention_mask"].sum()
            #             flow_loss = flow_loss.sum() / batch["attention_mask"].sum()

            output.update(
                {
                    "class_loss": class_loss,
                    "sep_loss": sep_loss,
                }
            )

            class_loss2 = []
            for i, class_ in enumerate(self.cfg.discourse_types):
                single_class_target = (batch["target"] == i) * 1.0

                single_class_loss = self.single_class_loss_fn(
                    class_preds[:, :, i], single_class_target
                )
                single_class_loss *= batch["attention_mask"]
                single_class_loss = single_class_loss.sum() / batch["attention_mask"].sum()

                class_loss2.append(single_class_loss)

                output.update({f"class_loss_{class_}": single_class_loss})

            class_loss2 = torch.stack(class_loss2).mean()

            eff_loss = self.loss_fn_bce(logit_eff, batch["target_eff"])

            loss = (
                0.5*(self.cfg.loss_weights[0] * (0.5 * class_loss + 0.5 * class_loss2)
                + self.cfg.loss_weights[1] * sep_loss) +0.5*eff_loss
            )
            output.update(
                {
                    "loss": loss,
                    "class_loss": class_loss,
                    "class_loss_single": class_loss2,
                    "sep_loss": sep_loss,
                    "eff_loss": eff_loss,
                }
            )

        # if not self.training:
        output["class_preds"] = class_preds
        output["sep_preds"] = sep_preds
        output["target_class"] = batch["target"]
        output["target_sep"] = batch["target2"]
        output["wordposition"] = batch["wordposition"]
        output["attention_mask"] = batch["attention_mask"]
        ###
        output["target_eff"] = batch["target_eff"]
        output["logit_eff"] = batch["logit_eff"]

        return output
    
    
