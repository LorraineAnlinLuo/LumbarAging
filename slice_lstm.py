import torch
from torch import nn


def encoder_blk(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
        nn.InstanceNorm2d(out_channels),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU()
    )


class MRI_LSTM(nn.Module):
    def __init__(self, lstm_feat_dim, lstm_latent_dim):
        super(MRI_LSTM, self).__init__()

        self.feat_embed_dim = lstm_feat_dim
        self.latent_dim = lstm_latent_dim
        self.n_layers = 1

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
            nn.Dropout(p=0.5),
            nn.Conv2d(64, self.feat_embed_dim, 1)
        )

        self.lstm = nn.LSTM(self.feat_embed_dim, self.latent_dim, self.n_layers, batch_first=True)

        # Build regressor
        self.lstm_post = nn.Linear(self.latent_dim, 64)
        self.regressor = nn.Sequential(nn.ReLU(), nn.Linear(64, 1))

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

    def init_hidden(self, x):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.latent_dim, device=x.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.latent_dim, device=x.device)
        h_0.requires_grad = True
        c_0.requires_grad = True
        return h_0, c_0

    def forward(self, x):
        h_0, c_0 = self.init_hidden(x)
        B, C, H, W, D = x.size()

        new_input = torch.cat([x[:, :, :, :, i] for i in range(D)], dim=0)
        encoding = self.encoder(new_input)
        encoding = self.post_proc(encoding)
        encoding = torch.cat([i.unsqueeze(4) for i in torch.split(encoding, B, dim=0)], dim=4)
        encoding = encoding.squeeze(3).squeeze(2)

        # lstm take  batch x seq_len x dim
        encoding = encoding.permute(0, 2, 1)
        _, (encoding, _) = self.lstm(encoding)
        # output is 1 X batch x hidden
        embedding = encoding.squeeze(0)
        post = self.lstm_post(embedding)
        y_pred = self.regressor(post)
        return y_pred


def get_model(lstm_feat_dim, lstm_latent_dim):
    model = MRI_LSTM(lstm_feat_dim=lstm_feat_dim, lstm_latent_dim=lstm_latent_dim)
    return model