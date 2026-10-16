import torch
from torch import nn
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, LongformerTokenClassifierOutput, LongformerModel

from dataclasses import dataclass
from typing import Optional, Tuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from transformers.modeling_outputs import ModelOutput


@dataclass
class LongformerTokenClassifierOutput2(LongformerTokenClassifierOutput): #ModelOutput
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



class LongformerForTokenClassificationLSTMV2(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        
        self.num_labels = 15    #CFG.num_labels
        self.num_labels_eff = 3 #CFG.num_labels_eff

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, dropout=config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        
        self.head_eff = nn.Linear(config.hidden_size, self.num_labels_eff)
        
        self.loss_fct  = nn.CrossEntropyLoss()
        self.loss_fct2 = nn.CrossEntropyLoss()
        self.loss_weights = [1.0, 1.0]
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
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
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
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
        logits_eff = self.head_eff(lstm_output)

        loss = None
        loss1 = None
        loss2 = None
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            loss1 = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss2 = self.loss_fct2(logits_eff.view(-1, self.num_labels_eff), labels_eff.view(-1))

            loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2

        if not return_dict:
            # output = (logits,) + outputs[2:]
            output = (logits,) + (logits_eff,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerTokenClassifierOutput2(
            loss=loss, 
            logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
            logits_eff=logits_eff,
            loss1=loss1, 
            loss2=loss2, 
        )

    
##### NEW model #####

import sys 

from models.longformerbilstm import LongformerForTokenClassificationwithbiLSTM 



class LongformerForTokenClassificationFP102(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = CFG.num_labels #config.num_labels # 15 
        self.num_labels_eff = CFG.num_labels_eff # 5 

        ### Load FP1 2nd weights 
        fp1_weights = "/mnt/md1/Work/dev_IME/Git_hacks/Feedback3/Feedback/checkpoints/longformerbilstm/aug-longformer-large-4096-f0/checkpoint-5500" #CFG.pretrained_model_path
        self.mdl = LongformerForTokenClassificationwithbiLSTM.from_pretrained(fp1_weights)
        
        
        # self.backbone = LongformerModel(config, add_pooling_layer=False)
        # self.backbone = m.model
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.dropout  = nn.Dropout(0.1)
        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.2)
        # self.dropout3 = nn.Dropout(0.3)
        # self.dropout4 = nn.Dropout(0.4)
        # self.dropout5 = nn.Dropout(0.5)

        
        # self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, num_layers=2, dropout=config.hidden_dropout_prob, batch_first=True,
        #                       bidirectional=True)
        
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
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

        # outputs = self.backbone(
        outputs = self.mdl.longformer(
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
        # sequence_output, hc = self.bilstm(sequence_output)
        sequence_output, hc = self.mdl.bilstm(sequence_output)

        # logits = self.classifier(sequence_output)
        logits = self.mdl.classifier(sequence_output)

        # print('logits', logits.shape)
        # print('labels', np.unique(labels.detach().cpu().numpy()))

        logits_eff = self.head_eff(sequence_output)
        # print('logits_eff', logits_eff.shape)
        # print('labels_eff', np.unique(labels_eff.detach().cpu().numpy()))

        loss = None
        loss1 = None
        loss2 = None
        if labels is not None and labels_eff is not None:
            # loss_fct = nn.CrossEntropyLoss()
            # TODO add KL loss 
            loss1 = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss2 = self.loss_fct2(logits_eff.view(-1, self.num_labels_eff), labels_eff.view(-1))

            loss = self.loss_weights[0]*loss1 + self.loss_weights[1]*loss2 

        if not return_dict:
            # output = (logits,) + outputs[1:]
            output = (logits,) + (logits_eff,) + outputs[1:]
  
            return ((loss,) + output) if loss is not None else output

        return LongformerTokenClassifierOutput2(
            loss=loss, 
            logits=logits, 
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions,
            logits_eff=logits_eff,
            loss1=loss1, 
            loss2=loss2, 
        )


