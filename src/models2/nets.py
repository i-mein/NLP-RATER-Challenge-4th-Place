import os, gc
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule
import wandb


from dataset.dataset import *
from utils.utils import *
from utils.awp import *


### ------------------------
###  *** FEEDBACK MODELS ***
### ------------------------


default_no_decay_parameters = ("LayerNorm.weight", "LayerNorm.bias", "bias")

def get_decay_module_parameters(module, no_decay_parameters=default_no_decay_parameters, recurse=True):
    for name, parameter in list(module.named_parameters(recurse=recurse)):
        if name not in no_decay_parameters:
            yield parameter

            
def get_no_decay_module_parameters(module, no_decay_parameters=default_no_decay_parameters, recurse=True):
    for name, parameter in list(module.named_parameters(recurse=recurse)):
        if name in no_decay_parameters:
            yield parameter
            

def init_weights(module, **kwargs):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, **kwargs)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, **kwargs)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def glorot_uniform(parameter):
    nn.init.xavier_uniform_(parameter.data, gain=1.0)




### Pooler / Heads

def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings



######################################################################################
#### Loss
######################################################################################


class MCRMSE(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target, return_column_wise=False):
        """
        Mean Column-wise Root Mean Squared Error
        https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/348985
        """
        
        column_wise_mse = torch.mean(torch.square(target - input), dim=0)
        mcrmse = torch.mean(torch.sqrt(column_wise_mse), dim=0)
        
        return mcrmse, column_wise_mse if return_column_wise else mcrmse
    

######################################################################################
#### Lightning models
######################################################################################


class FeedbackPrize3Model(pl.LightningModule):
    def __init__(
        self, 
        pretrained_model_name_or_path="microsoft/deberta-v3-small", 
        token_embeddings_size=None,
        gradient_checkpointing=False,
        num_labels=6,
        config_path=None,
        pretrained=True,
        label2id={},
        config_updates={}, 
        optimizer={}, 
        scheduler={}):
        super().__init__()
        
        if config_path is None:
            self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        else:
            self.config = AutoConfig.from_pretrained(config_path)
            
        self.config.output_hidden_states = True
        self.config.update(config_updates)
            
        if pretrained:
            self.backbone = AutoModel.from_pretrained(pretrained_model_name_or_path, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)
        
        if token_embeddings_size is not None:
            self.backbone.resize_token_embeddings(token_embeddings_size)
            print(f"Token Embeddings size: {token_embeddings_size}")
        
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            print(f"Gradient Checkpointing: {self.backbone.is_gradient_checkpointing}")
        
        self.pooling = MeanPooling()
        self.head = nn.Linear(in_features=self.config.hidden_size, out_features=num_labels)
        init_weights(self.head, std=self.config.initializer_range)

        # # defining attention network for attention scores 
        # self.attention = nn.Sequential(
        #     nn.Linear(self.config.hidden_size, 512),
        #     nn.Tanh(),
        #     nn.Linear(512, 1),
        #     nn.Softmax(dim=1)
        # )
        # # self._init_weights(self.attention)
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.label2id = label2id
        
        self.loss_fn = MCRMSE()

        # self.awp = AWP(model, 
        #     criterion, 
        #     optimizer,
        #     CFG.apex,
        #     adv_lr=CFG.awp_lr, 
        #     adv_eps=CFG.awp_eps)
        
        self.save_hyperparameters()
        
    def forward(self, input_ids, attention_mask=None, **kwargs):

        #transformer_out = self.transformer(input_ids=ids, attention_mask=mask)
        #sequence_output = transformer_out.last_hidden_state[:, 0, :]

        backbone_outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            **kwargs
        )
        ##### MeanPool
        #hidden_states = backbone_outputs.hidden_states
        #hidden_state = hidden_states[-1]
        #features = self.pooling(hidden_state=hidden_state, attention_mask=attention_mask)
        
        #### CLS Pool
        features = backbone_outputs.last_hidden_state[:, 0, :]

        #### attention based sentence representation
        # weights = self.attention(last_hidden_states)
        # attention_feature = torch.sum(weights * last_hidden_states, dim=1)

        #### Concat them
        # combine_feature = torch.cat([mean_feature, attention_feature, cls_token_feature], dim = -1)


        #### Head
        outputs = self.head(features)
        return outputs
        
    def training_step(self, batch, batch_index):
        input_ids = batch["input_ids"].to(torch.int32)
        attention_mask = batch["attention_mask"].to(torch.int32)
        labels = batch["labels"].to(torch.float32)
        
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = F.smooth_l1_loss(outputs, labels)
        mcrmse_loss_, column_wise_mse_loss_ = self.loss_fn(outputs, labels, return_column_wise=True)

        ##### AWP adversial attack for perturbation
        if self.config['nth_awp_start_epoch'] <= batch_index: #epoch:
                loss = self.awp.attack_backward(input_ids, labels)
                # scaler.scale(loss).backward()
                self.awp._restore()
        # losses.update(loss.item(), batch_size)
        
        logs_dict = {
            "train/loss": loss,
            "train/mcrmse_loss": mcrmse_loss_,
        }
        
        for label, id in self.label2id.items():
            log_name = f"train/{label}_mse_loss"
            logs_dict[log_name] = column_wise_mse_loss_[id]
        
        self.log_dict(logs_dict, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_index):
        input_ids = batch["input_ids"].to(torch.int32)
        attention_mask = batch["attention_mask"].to(torch.int32)
        labels = batch["labels"].to(torch.float32)
        
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = F.smooth_l1_loss(outputs, labels)
        mcrmse_loss_, column_wise_mse_loss_ = self.loss_fn(outputs, labels, return_column_wise=True)
        
        logs_dict = {
            "validation/loss": loss,
            "validation/mcrmse_loss": mcrmse_loss_,
        }
        
        for label, id in self.label2id.items():
            log_name = f"validation/{label}_mse_loss"
            logs_dict[log_name] = column_wise_mse_loss_[id]
        
        self.log_dict(logs_dict, on_step=False, on_epoch=True, prog_bar=False)
        
        return {"predictions": outputs, "labels": labels}
    
    def predict_step(self, batch, batch_index):
        input_ids = batch["input_ids"].to(torch.int32)
        attention_mask = batch["attention_mask"].to(torch.int32)
        
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
    
        return outputs
    
    def configure_optimizers(self):
        optimizers, schedulers = [], []
        
        model_parameters = [
            {"params": get_decay_module_parameters(self), "weight_decay": 1e-2},
            {"params": get_no_decay_module_parameters(self), "weight_decay": 0.0},
        ]
        ### TODO try AdamP
        optimizer = AdamW(params=model_parameters, **self.optimizer)
        optimizers.append(optimizer)
        
        if self.scheduler != {}:
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, **self.scheduler)
            schedulers.append({"scheduler": scheduler, "interval": "step"})
            
        return optimizers, schedulers
    
    def on_train_start(self):
        metrics_summary = {
            "validation/mcrmse_loss": {"summary": "min"},
        }
        
        for metric, summary in metrics_summary.items():
            wandb.define_metric(metric, **summary)    
            
    def validation_epoch_end(self, outputs):
        outputs = torch.cat([output["predictions"] for output in outputs], dim=0)
        outputs = outputs.to("cpu").numpy()
        
        for label, id in self.label2id.items():
            prediction_column = f"{label}_prediction"
            valid_fold[prediction_column] = outputs[:, id]
        
        validation_fold_path = os.path.join(fold_directory, "oof.csv")
        valid_fold.to_csv(validation_fold_path, index=False)




######################################################################################
#### Torch models
######################################################################################

#### FB3
# see: https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train

class FeedbackModel_01(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model_name, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model_name, config=self.config)
        else:
            self.model = AutoModel(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 6)
        self._init_weights(self.fc)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    # def forward(self, inputs):
    #     outputs = self.model(**inputs)
    #     last_hidden_states = outputs[0]
    #     pool_feats = self.pool(last_hidden_states, inputs['attention_mask'])

    #     # feature = self.feature(feature)
    #     output = self.fc(pool_feats)
    #     return output


    def forward(self, inputs):
        outputs = self.model(**inputs)
        
        ##### MeanPooling
        if self.cfg.pooling_type=="mean":
            last_hidden_states = outputs[0]
            pool_feats = self.pool(last_hidden_states, inputs['attention_mask'])
        ##### attention based sentence representation
        elif self.cfg.pooling_type=="attention":
            weights = self.attention(last_hidden_states)
            pool_feats = torch.sum(weights * last_hidden_states, dim=1)
        
        ##### CLS Token representation
        else:
            pool_feats = outputs.last_hidden_state[:, 0, :] # only cls token
            # pool_feats = outputs[0][:,0]

        output = self.fc(pool_feats)
        return output




## --------------------------------------

class FeedbackModel_01_cls(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False, vb=False):
        super().__init__()
        self.cfg = cfg
        self.vb = vb
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
            # self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=1)
        else:
            self.model = AutoModel.from_config(self.config)
 
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.2)
        self.dp3 = nn.Dropout(0.3)
        self.dp4 = nn.Dropout(0.4)
        self.dp5 = nn.Dropout(0.5)

        self._init_weights(self.fc)
        # self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        


    def forward(self, inputs):
        outputs = self.model(**inputs)
        feature = outputs[0][:,0]
        # if self.vb: print('mdl out:', outputs.shape)
        # last_hidden_states = outputs[0]
        # weights = self.attention(last_hidden_states)
        # if self.vb: print('attn weights:', weights.shape)  ## B x max_len x 1
        # feature = torch.sum(weights * last_hidden_states, dim=1)
        if self.vb: print('features:', feature.shape)  ## B x 1024
        # logits = self.fc(self.fc_dropout(feature))

        ##### multi dropout
        l1 = self.fc(self.dp1(feature))
        l2 = self.fc(self.dp2(feature))
        l3 = self.fc(self.dp3(feature))
        l4 = self.fc(self.dp4(feature))
        l5 = self.fc(self.dp5(feature))

        logits = (l1+l2+l3+l4+l5)/5

        if self.vb: print('output:', logits.shape)
        return logits



class FeedbackModel_01_att(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False, vb=False):
        super().__init__()
        self.cfg = cfg
        self.vb = vb
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
            # self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=1)
        else:
            self.model = AutoModel.from_config(self.config)
 
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.2)
        self.dp3 = nn.Dropout(0.3)
        self.dp4 = nn.Dropout(0.4)
        self.dp5 = nn.Dropout(0.5)

        self._init_weights(self.fc)
        self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        if self.vb: print('inputs:', inputs.keys())
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs):
        # feature = self.feature(inputs)
        outputs = self.model(**inputs)
        # if self.vb: print('mdl out:', outputs.shape)
        last_hidden_states = outputs[0]
        weights = self.attention(last_hidden_states)
        if self.vb: print('attn weights:', weights.shape)  ## B x max_len x 1
        feature = torch.sum(weights * last_hidden_states, dim=1)
        if self.vb: print('features:', feature.shape)  ## B x 1024
        # logits = self.fc(self.fc_dropout(feature))

        ##### multi dropout
        l1 = self.fc(self.dp1(feature))
        l2 = self.fc(self.dp2(feature))
        l3 = self.fc(self.dp3(feature))
        l4 = self.fc(self.dp4(feature))
        l5 = self.fc(self.dp5(feature))

        logits = (l1+l2+l3+l4+l5)/5

        if self.vb: print('output:', logits.shape)
        return logits


class TransformerHead(nn.Module):
    def __init__(self, in_features, max_length, num_layers=1, nhead=8, num_targets=1):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=in_features, nhead=nhead),
            num_layers=num_layers
            )
        self.row_fc = nn.Linear(in_features, 1)
        self.out_features = max_length

    def forward(self, x):
        out = self.transformer(x)
        out = self.row_fc(out).squeeze(-1)
        return out



class Model_tfm(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False, vb=False):
        super().__init__()
        self.cfg = cfg
        self.vb = vb
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.dropout0 = nn.Dropout(cfg.reg_dropout)

        self.attention = TransformerHead(
            in_features=self.config.hidden_size, 
            max_length=cfg.max_len, num_layers=1, nhead=8, 
            num_targets=1
            )

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.2)
        self.dp3 = nn.Dropout(0.3)
        self.dp4 = nn.Dropout(0.4)
        self.dp5 = nn.Dropout(0.5)

        weight_data = torch.linspace(-5, 5, 1+self.config.num_hidden_layers)
        weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights = nn.Parameter(weight_data, requires_grad=True)
        
        self.fc = nn.Linear(self.attention.out_features, self.cfg.target_size)
        # self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        # self.fc = nn.Linear(self.config.hidden_size+(2*cfg.emb_dim), self.cfg.target_size)
        # self.embedding_alpha = nn.Embedding(num_embeddings=9, embedding_dim=cfg.emb_dim)
        # self.embedding_num   = nn.Embedding(num_embeddings=100, embedding_dim=cfg.emb_dim)

        # self._init_weights(self.lstm)
        # self._glorot_uniform(self.fc)
        self._init_weights(self.fc)
        self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _glorot_uniform(self, parameter):
        nn.init.xavier_uniform_(parameter.data, gain=1.0)
        

    def get_features(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs["last_hidden_state"]
        features = last_hidden_states[:, 0, :]
        return features


    def forward(self, inputs):
        outputs = self.model(**inputs, output_hidden_states=True)
        if self.vb: print('backbone out:', outputs.keys()) 
        # attention_mask = inputs['attention_mask']
        # print(attention_mask.shape)
        outputs = outputs[0] ## outputs["last_hidden_state"]  ## outputs[0]
        # if self.vb: print('last_hidden out:', last_hidden_states.shape)  ## B x 192 x 1024
        ### last_hidden_states = last_hidden_states[:, 0, :]    ## B x 1024

        # x = torch.stack(outputs.hidden_states)
        # w = F.softmax(self.weights, 0)
        # w = self.dropout0(w)
        # out = (w * x).sum(0)

        ##### attention 
        # feature, hc = self.lstm(outputs)
        # feature = self.lstm(out)[0]
        feature = self.attention(outputs)       ## B x 144
        if self.vb: print('features:', feature.shape)       ## B x 192 x 1024
        # feature = feature[:, 0, :]  ### TODO or -1 ??
        # if self.vb: print('features last:', feature.shape)  ## B x 1024
   
        ##### multi dropout
        l1 = self.fc(self.dp1(feature))
        l2 = self.fc(self.dp2(feature))
        l3 = self.fc(self.dp3(feature))
        l4 = self.fc(self.dp4(feature))
        l5 = self.fc(self.dp5(feature))

        logits = (l1+l2+l3+l4+l5)/5

        if self.vb: print('output:', logits.shape)   ## B x 1
        return logits





class USPModel_r14(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False, vb=False):
        super().__init__()
        self.cfg = cfg
        self.vb = vb
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.dropout0 = nn.Dropout(cfg.reg_dropout)

        self.lstm = nn.LSTM(
            self.config.hidden_size, (self.config.hidden_size) // 2, 
            dropout=self.config.hidden_dropout_prob, 
            batch_first=True, 
            num_layers=1,
            bidirectional=True ## False
            )

        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.2)
        self.dp3 = nn.Dropout(0.3)
        self.dp4 = nn.Dropout(0.4)
        self.dp5 = nn.Dropout(0.5)

        weight_data = torch.linspace(-5, 5, 1+self.config.num_hidden_layers)
        weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights = nn.Parameter(weight_data, requires_grad=True)
        
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        # self.fc = nn.Linear(self.config.hidden_size+(2*cfg.emb_dim), self.cfg.target_size)
        # self.embedding_alpha = nn.Embedding(num_embeddings=9, embedding_dim=cfg.emb_dim)
        # self.embedding_num   = nn.Embedding(num_embeddings=100, embedding_dim=cfg.emb_dim)

        self._init_weights(self.lstm)
        # self._glorot_uniform(self.fc)
        self._init_weights(self.fc)
        # self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _glorot_uniform(self, parameter):
        nn.init.xavier_uniform_(parameter.data, gain=1.0)
        

    def get_features(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs["last_hidden_state"]
        features = last_hidden_states[:, 0, :]
        return features


    def forward(self, inputs):
        outputs = self.model(**inputs, output_hidden_states=True)
        if self.vb: print('backbone out:', outputs.keys()) 
        # attention_mask = inputs['attention_mask']
        # print(attention_mask.shape)
        # last_hidden_states = outputs["last_hidden_state"]  ## outputs[0]
        # if self.vb: print('last_hidden out:', last_hidden_states.shape)  ## B x 192 x 1024
        ### last_hidden_states = last_hidden_states[:, 0, :]    ## B x 1024

        x = torch.stack(outputs.hidden_states)
        w = F.softmax(self.weights, 0)
        w = self.dropout0(w)
        out = (w * x).sum(0)

        ##### lstm 
        # feature, hc = self.lstm(last_hidden_states)
        feature = self.lstm(out)[0]
        if self.vb: print('features:', feature.shape)       ## B x 192 x 1024
        feature = feature[:, 0, :]  ### TODO or -1 ??
        if self.vb: print('features last:', feature.shape)  ## B x 1024
        
        # alpha_outputs = self.embedding_alpha(inp_alpha) 
        # num_outputs   = self.embedding_num(inp_num)         
        # if self.vb: print('emb_outputs:', alpha_outputs.shape,num_outputs.shape) ## B x 8 (emb_dim)
        
        # outputs = torch.cat([feature, alpha_outputs, num_outputs], axis=1) 
        # if self.vb: print('concat:', outputs.shape)  ## B x 1040

        ##### multi dropout
        l1 = self.fc(self.dp1(feature))
        l2 = self.fc(self.dp2(feature))
        l3 = self.fc(self.dp3(feature))
        l4 = self.fc(self.dp4(feature))
        l5 = self.fc(self.dp5(feature))

        logits = (l1+l2+l3+l4+l5)/5

        if self.vb: print('output:', logits.shape)   ## B x 1
        return logits




