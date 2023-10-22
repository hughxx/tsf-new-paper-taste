import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.rev = RevIN(configs.enc_in)
        self.dropout0 = nn.Dropout(configs.dropout)

        self.enc_in = configs.enc_in
        self.channel_embed = nn.Embedding(configs.enc_in, configs.d_model // 2)

        self.patch_len = 48
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            dropout=0.5,
            batch_first=True,
        )

        self.pos_embed = PositionalEmbedding(self.d_model // 2)
        self.channel_embed = nn.Embedding(configs.enc_in, configs.d_model // 2)

        self.dropout = nn.Dropout(0.5)
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x, x_mark, y, y_mark):
        #x = self.rev(x, 'norm') if self.rev else x
        #x = self.dropout0(x)

        B0, _, D0 = x.shape
        x = x.permute(0, 2, 1).reshape(B0 * D0, -1, 1).squeeze(-1)

        B, L = x.shape
        N = L // self.patch_len
        M = self.pred_len // self.patch_len
        D = self.d_model

        xw = x.unsqueeze(-1).reshape(B, N, -1)  # B, N, W
        xd = self.linear_patch(xw)  # B, N, D
        enc_in = self.relu(xd)  # B, N, D

        enc_out = self.gru(enc_in)[1].unsqueeze(2).repeat(1, 1, M, 1).reshape(-1, B * M, D, 1).squeeze(
            -1)  # Layer, B * M, D

        zeros = torch.zeros((B, M, D), device=x.device)
        pos_embed = self.pos_embed(zeros).repeat(B, 1, 1)  # B, M, D/2
        channel_embed = self.channel_embed(torch.arange(self.enc_in, device=x.device))
        channel_embed = channel_embed.unsqueeze(1).repeat(1, B0, 1).reshape(-1, D // 2, 1).squeeze(-1)
        channel_embed = channel_embed.unsqueeze(1).repeat(1, M, 1)  # B, M, D/2
        dec_in = torch.cat([pos_embed, channel_embed], dim=-1).reshape(B * M, 1, -1)  # B * M, 1, D

        dec_out = self.gru(dec_in, enc_out)[0]

        yw = self.dropout(dec_out)
        yd = self.linear_patch_re(yw)  # B * M, 1, D
        pred = yd.reshape(B, -1, 1).squeeze(-1) # B, H

        pred = pred.unsqueeze(1).reshape(B0, D0, -1).permute(0, 2, 1)

        #pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred