import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.enc_in = configs.enc_in
        self.channel_embed = nn.Embedding(configs.enc_in, configs.d_model // 2)

        self.patch_len = configs.patch_len
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.dropout = nn.Dropout(configs.dropout)
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x, x_mark, y, y_mark):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        B0, _, C = x.shape
        x = x.permute(0, 2, 1).reshape(B0 * C, -1, 1).squeeze(-1)

        B, L = x.shape
        N = L // self.patch_len
        M = self.pred_len // self.patch_len
        D = self.d_model

        xw = x.unsqueeze(-1).reshape(B, N, -1)  # B, N, W
        xd = self.linear_patch(xw)  # B, N, D
        enc_in = self.relu(xd)  # B, N, D

        enc_out = self.gru(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1,B,D -> 1,B,MD -> 1,BM,D

        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B0*C, 1, 1), # M,D//2 -> 1,M,D//2 -> B0*C,M,D//2
            self.channel_emb.unsqueeze(1).repeat(B0, M, 1) # C,D//2 -> C,1,D//2 -> B0*C,M,D//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B0*C,M,D -> B0*C*M,D -> B0*C*M, 1, D

        dec_out = self.gru(dec_in, enc_out)[0]  # B0*C*M, D

        yw = self.dropout(dec_out)
        yd = self.linear_patch_re(yw)  # B * M, 1, D
        pred = yd.reshape(B, -1, 1).squeeze(-1) # B, H

        pred = pred.unsqueeze(1).reshape(B0, C, -1).permute(0, 2, 1)

        pred = pred + seq_last

        return pred