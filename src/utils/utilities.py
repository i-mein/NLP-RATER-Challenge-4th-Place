from sklearn.metrics import mean_squared_error
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=None):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.window_size = window_size

    def reset(self):
        self.length = 0
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.window_size and (self.count >= self.window_size):
            self.reset()
        self.val = val 
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class WeightedLayerPooling(nn.Module):
    def __init__(self, layers = 12):
        super(WeightedLayerPooling, self).__init__()
        self.layers = layers
        self.layer_weights = nn.Parameter(
                torch.tensor([1] * layers, dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_hidden_states = torch.stack(all_hidden_states, dim=0)
        all_layer_embedding = all_hidden_states[-self.layers:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class WeightedLayerPoolingWithDrop(nn.Module):
    def __init__(self, layers = 12, drop_prob = .2):
        super(WeightedLayerPoolingWithDrop, self).__init__()
        self.layers = layers
        self.layer_weights = nn.Parameter(
                torch.tensor([1] * layers, dtype=torch.float)
            )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, all_hidden_states):
        all_hidden_states = torch.stack(all_hidden_states, dim=0)
        all_layer_embedding = all_hidden_states[-self.layers:, :, :, :]
        embeddings = []
        for i in range(all_layer_embedding.size(0)):
            embeddings.append(self.dropout(all_layer_embedding[0]))
        embeddings = torch.stack(embeddings, dim=0)
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(embeddings.size())
        weighted_average = (weight_factor*embeddings).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class Multisample_Dropout(nn.Module):
    def __init__(self):
        super(Multisample_Dropout, self).__init__()
        self.dropout = nn.Dropout(.1)
        self.dropouts = nn.ModuleList([nn.Dropout((i+1)*.1) for i in range(5)])
        
    def forward(self, x, module):
        x = self.dropout(x)
        return torch.mean(torch.stack([module(dropout(x)) for dropout in self.dropouts],dim=0),dim=0) 

class Bi_RNN(nn.Module):
    def __init__(self, size, hidden_size, layers):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(size, hidden_size, num_layers=layers, bidirectional=True, bias=False, batch_first=True)
                                
    def forward(self, x):
        self.rnn.flatten_parameters()
        x, hidden = self.rnn(x)
        return torch.cat((x[:,-1,:self.hidden_size], x[:,0,self.hidden_size:]), dim=-1) 

class Bi_RNN_FOUT(nn.Module):
    def __init__(self, size, hidden_size, layers):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(size, hidden_size, num_layers=layers, bidirectional=True, bias=False, batch_first=True)
                                
    def forward(self, x):
        self.rnn.flatten_parameters()
        x, hidden = self.rnn(x)
        return x#torch.cat((x[:,-1,:self.hidden_size], x[:,0,self.hidden_size:]), dim=-1) 

class Bi_RNN_FOUT_Res(nn.Module):
    def __init__(self, size, hidden_size, layers):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(size, hidden_size, num_layers=layers, bidirectional=True, bias=False, batch_first=True)
        self.linear = nn.Linear(hidden_size*2, size)
                                
    def forward(self, x):
        x_res = x
        self.rnn.flatten_parameters()
        x, hidden = self.rnn(x)
        x = self.linear(x)
        return x+x_res#torch.cat((x[:,-1,:self.hidden_size], x[:,0,self.hidden_size:]), dim=-1) 

def get_groups(param_groups, group_name):
        groups = sorted(set([param_g[group_name] for param_g in param_groups]))
        groups = ["{:2e}".format(group) for group in groups]
        return groups

def process_outputs(out, n_layers=6):
    out = torch.stack(out[-n_layers:], dim=0) 
    out_mean = torch.mean(out, dim=0)
    out_max, _ = torch.max(out, dim=0)
    out_std = torch.std(out, dim=0)
    last_hidden_states = torch.cat((out_mean, out_max, out_std), dim=-1)
    return last_hidden_states

class Attention(nn.Module):
    def __init__(self, in_dim, hidden_dim, pool_dim=1):
        super().__init__()
        self.pool_dim = pool_dim
        self.attention = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim, 1),
                                       nn.Softmax(dim=self.pool_dim))

    def forward(self, x):
        weights = self.attention(x)
        context = torch.sum((weights * x), dim=self.pool_dim)
        return context

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

class BasicMeanPooling(nn.Module):
    def __init__(self):
        super(BasicMeanPooling, self).__init__()
        
    def forward(self, last_hidden_state):
        return last_hidden_state.mean(1)

class ResidualLSTM(nn.Module):

    def __init__(self, d_model,rnn='LSTM'):
        super(ResidualLSTM, self).__init__()
        self.downsample=nn.Linear(d_model,d_model//2)
        if rnn=='GRU':
            self.rnn=nn.GRU(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        else:
            self.rnn=nn.LSTM(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        self.dropout1=nn.Dropout(0.2)
        self.norm1= nn.LayerNorm(d_model//2)
        self.linear1=nn.Linear(d_model//2, d_model*4)
        self.linear2=nn.Linear(d_model*4, d_model)
        self.dropout2=nn.Dropout(0.2)
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, x):
        self.rnn.flatten_parameters()
        res=x
        x=self.downsample(x)
        x, _ = self.rnn(x)
        x=self.dropout1(x)
        x=self.norm1(x)
        x=F.relu(self.linear1(x))
        x=self.linear2(x)
        x=self.dropout2(x)
        x=res+x
        return self.norm2(x)