from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaModel
from transformers.modeling_outputs import TokenClassifierOutput
from torch import nn
import torch

from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput
from transformers import AutoTokenizer, AutoModelForTokenClassification


@dataclass
class TokenClassifierOutput2(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits_eff: torch.FloatTensor = None
    loss1: Optional[torch.FloatTensor] = None
    loss2: Optional[torch.FloatTensor] = None



class DebertaForTokenClassificationLSTM2(DebertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 15 # CFG.num_labels #config.num_labels # 15 
        self.num_labels_eff = 5 # CFG.num_labels_eff # 5 

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, num_layers=2, dropout=config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.head_eff = nn.Linear(config.hidden_size, self.num_labels_eff)
        
        self.loss_fct = nn.CrossEntropyLoss()
        self.loss_fct2 = nn.CrossEntropyLoss()
        self.loss_weights = [1.0, 1.0]

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        labels_eff=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels  for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        Labels2 for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)

        logits = self.classifier(lstm_output)
        # print('logits', logits.shape)
        # print('logits', np.unique(labels.detach().cpu().numpy()))

        logits_eff = self.head_eff(lstm_output)
        # print('logits_eff', logits_eff.shape)
        # print('logits_eff', np.unique(labels_eff.detach().cpu().numpy()))

        loss = None
        loss1 = None
        loss2 = None
        if labels is not None and labels_eff is not None:
            # loss_fct = nn.CrossEntropyLoss()
            loss1 = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss2 = self.loss_fct2(logits_eff.view(-1, self.num_labels_eff), labels_eff.view(-1))

            loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2 

        if not return_dict:
            # output = (logits,) + outputs[1:]
            output = (logits,) + (logits_eff,) + outputs[1:]
  
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput2(
            loss=loss, 
            logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
            logits_eff=logits_eff,
            loss1=loss1, 
            loss2=loss2, 
        )




class DebertaForTokenClassificationLSTM2Test(DebertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 15 # CFG.num_labels #config.num_labels # 15 
        self.num_labels_eff = 5 # CFG.num_labels_eff # 5 

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, num_layers=2, dropout=config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.head_eff = nn.Linear(config.hidden_size, self.num_labels_eff)
        
        self.loss_fct = nn.CrossEntropyLoss()
        self.loss_fct2 = nn.CrossEntropyLoss()
        self.loss_weights = [1.0, 1.0]

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        labels_eff=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels  for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        Labels2 for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)

        logits = self.classifier(lstm_output)
        # print('logits', logits.shape)
        # print('logits', np.unique(labels.detach().cpu().numpy()))

        logits_eff = self.head_eff(lstm_output)
        # print('logits_eff', logits_eff.shape)
        # print('logits_eff', np.unique(labels_eff.detach().cpu().numpy()))

        loss = None
        loss1 = None
        loss2 = None
        if labels is not None and labels_eff is not None:
            # loss_fct = nn.CrossEntropyLoss()
            loss1 = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss2 = self.loss_fct2(logits_eff.view(-1, self.num_labels_eff), labels_eff.view(-1))

            loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2 

        if not return_dict:
            # output = (logits,) + outputs[1:]
            output = (logits,) + (logits_eff,) + outputs[1:]
  
            return ((loss,) + output) if loss is not None else output


import sys 
sys.path.append('/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/src/models')
from  debertabilstm import DebertaForTokenClassificationwithbiLSTM


class DebertaForTokenClassificationLSTM2_FP1(DebertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 15 #CFG.num_labels #config.num_labels # 15 
        self.num_labels_eff = 3 #CFG.num_labels_eff # 4 

        fp1_weights = '/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/checkpoints/debertalstm/deberta-large-f1/checkpoint-7000'
        self.mdl = DebertaForTokenClassificationwithbiLSTM.from_pretrained(fp1_weights)
        # self.deberta = mdl.deberta

        # self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, num_layers=2, dropout=config.hidden_dropout_prob, batch_first=True,
        #                       bidirectional=True)
        # self.bilstm = mdl.bilstm
        
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # self.classifier = mdl.classifier
        
        self.head_eff = nn.Linear(config.hidden_size, self.num_labels_eff)
        
        self.loss_fct = nn.CrossEntropyLoss()
        self.loss_fct2 = nn.CrossEntropyLoss()
        self.loss_weights = [1.0, 1.0]

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        labels_eff=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels  for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        Labels2 for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mdl.deberta(
        # outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # lstm_output, hc = self.bilstm(sequence_output)
        lstm_output, hc = self.mdl.bilstm(sequence_output)

        # logits = self.classifier(lstm_output)
        logits = self.mdl.classifier(lstm_output)

        # print('logits', logits.shape)
        # print('logits', np.unique(labels.detach().cpu().numpy()))

        logits_eff = self.head_eff(lstm_output)
        # print('logits_eff', logits_eff.shape)
        # print('logits_eff', np.unique(labels_eff.detach().cpu().numpy()))

        loss = None
        loss1 = None
        loss2 = None
        if labels is not None and labels_eff is not None:
            # loss_fct = nn.CrossEntropyLoss()
            loss1 = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss2 = self.loss_fct2(logits_eff.view(-1, self.num_labels_eff), labels_eff.view(-1))

            loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2 

        if not return_dict:
            # output = (logits,) + outputs[1:]
            output = (logits,) + (logits_eff,) + outputs[1:]
  
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput2(
            loss=loss, 
            logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
            logits_eff=logits_eff,
            loss1=loss1, 
            loss2=loss2, 
        )



###### NEW 

class DebertaForTokenClassificationV2_CD(DebertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 15 #CFG.num_labels #config.num_labels  
        self.num_labels_eff = 3 # CFG.num_labels_eff 

        fp1_weights = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/checkpoints/debertal_cd/fold0"
        self.mdl = AutoModelForTokenClassification.from_pretrained(fp1_weights)

        # self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        ### init new head (types) 
        self.mdl.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        self.head_eff = nn.Linear(config.hidden_size, self.num_labels_eff)
        
        self.loss_fct = nn.CrossEntropyLoss()
        self.loss_fct2 = nn.CrossEntropyLoss()
        self.loss_weights = [1.0, 1.0]

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        labels_eff=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels  for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        Labels2 for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mdl.deberta(
        # outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # logits = self.classifier(sequence_output)
        logits = self.mdl.classifier(sequence_output)

        logits_eff = self.head_eff(sequence_output)

        loss = None
        loss1 = None
        loss2 = None
        if labels is not None and labels_eff is not None:
            # loss_fct = nn.CrossEntropyLoss()
            loss1 = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss2 = self.loss_fct2(logits_eff.view(-1, self.num_labels_eff), labels_eff.view(-1))

            loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2 

        if not return_dict:
            # output = (logits,) + outputs[1:]
            output = (logits,) + (logits_eff,) + outputs[1:]
  
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput2(
            loss=loss, 
            logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
            logits_eff=logits_eff,
            loss1=loss1, 
            loss2=loss2, 
        )
    




