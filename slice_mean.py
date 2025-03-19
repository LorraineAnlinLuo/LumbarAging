import math
import torch
from torch import nn
import torchvision
from feature_visualization import draw_feature_map


def encoder_blk(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
        nn.InstanceNorm2d(out_channels),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU()
    )


class MeanPool(nn.Module):
    def forward(self, X):
        return X.mean(dim=1, keepdim=True), None


class MaxPool(nn.Module):
    def forward(self, X):
        return X.max(dim=1, keepdim=True)[0], None


class PooledAttention(nn.Module):
    def __init__(self, input_dim, dim_v, dim_k, num_heads, ln=False):
        super(PooledAttention, self).__init__()
        self.S = nn.Parameter(torch.zeros(1, dim_k))
        nn.init.xavier_uniform_(self.S)

        # transform to get key and value vector
        self.fc_k = nn.Linear(input_dim, dim_k)
        self.fc_v = nn.Linear(input_dim, dim_v)

        self.dim_v = dim_v
        self.dim_k = dim_k
        self.num_heads = num_heads

        if ln:
            self.ln0 = nn.LayerNorm(dim_v)

    def forward(self, X):
        B, C, H = X.shape  # torch.Size([16, 8, 128])

        Q = self.S.repeat(X.size(0), 1, 1)  # torch.Size([16, 1, 128])

        K = self.fc_k(X.reshape(-1, H)).reshape(B, C, self.dim_k)  # torch.Size([16, 8, 128])
        V = self.fc_v(X.reshape(-1, H)).reshape(B, C, self.dim_v)  # torch.Size([16, 8, 128])
        dim_split = self.dim_v // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)  # torch.Size([16, 1, 128])
        K_ = torch.cat(K.split(dim_split, 2), 0)  # torch.Size([16, 8, 128])
        V_ = torch.cat(V.split(dim_split, 2), 0)  # torch.Size([16, 8, 128])
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), 2)  # torch.Size([16, 1, 8])
        O = torch.cat(A.bmm(V_).split(B, 0), 2)  # torch.Size([16, 1, 128])
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)  # torch.Size([16, 1, 128])
        return O,A


class MRI_ATTN(nn.Module):
    def __init__(self, attn_num_heads, attn_dim, attn_drop=False):
        super(MRI_ATTN, self).__init__()

        self.num_heads = attn_num_heads
        self.attn_dim = attn_dim
        self.attn_drop = attn_drop

        # Build Encoder
        encoder_blocks = [
                encoder_blk(1, 32),
                encoder_blk(32, 64),
                encoder_blk(64, 128),
                encoder_blk(128, 256),
                encoder_blk(256, 256)
        ]

        self.encoder = nn.Sequential(*encoder_blocks)

        # Post processing
        self.post_proc = nn.Sequential(
            nn.Conv2d(256, 64, 1, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d([7, 7]),
            nn.Dropout(p=0.5) if self.attn_drop else nn.Identity(),
            nn.Conv2d(64, self.num_heads * self.attn_dim, 1)
        )

        self.pooled_attention = MeanPool()

        # Build regressor
        self.attn_post = nn.Linear(self.num_heads * self.attn_dim, 64)
        self.regressor = nn.Sequential(
                                       nn.ReLU(),
                                       nn.Linear(64, 1))
        self.init_weights()

    def init_weights(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and "regressor" in k:
                m.bias.data.fill_(62.68)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # 不画特征图
    def forward(self, x):
        B, C, H, W, D = x.size()  # torch.Size([16, 1, 224, 224, 8])
        new_input = torch.cat([x[:, :, :, :, i] for i in range(D)], dim=0)  # torch.Size([128, 1, 224, 224])
        encoding = self.encoder(new_input)  # torch.Size([128, 256, 7, 7])
        encoding = self.post_proc(encoding)  # torch.Size([128, 128, 1, 1])
        encoding = torch.cat([i.unsqueeze(4) for i in torch.split(encoding, B, dim=0)], dim=4)  # torch.Size([16, 128, 1, 1, 8])
        encoding = encoding.squeeze(3).squeeze(2)  # torch.Size([16, 128, 8])

        # swap dims for input to attention
        encoding = encoding.permute((0, 2, 1))  # torch.Size([16, 8, 128])
        encoding,attention = self.pooled_attention(encoding)  # torch.Size([16, 1, 128]), torch.Size([16, 1, 8])
        embedding= encoding.squeeze(1)  # torch.Size([16, 128])

        post = self.attn_post(embedding)
        y_pred = self.regressor(post)  # torch.Size([16, 1])

        return y_pred


def get_model(attn_num_heads, attn_dim, attn_drop):
    model = MRI_ATTN(attn_num_heads=attn_num_heads, attn_dim=attn_dim, attn_drop=attn_drop)
    return model
