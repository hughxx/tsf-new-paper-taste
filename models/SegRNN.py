import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # remove this, the performance will be bad
        self.lucky = nn.Embedding(configs.enc_in, configs.d_model // 2)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.patch_len = configs.patch_len
        self.d_model = configs.d_model


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

    def forward(self, x, x_mark, y_true, y_mark):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model

        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)  # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = self.relu(xd)

        enc_out = self.gru(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d

        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1), # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d

        dec_out = self.gru(dec_in, enc_out)[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1) # B, C, H

        y = y + seq_last

        return y