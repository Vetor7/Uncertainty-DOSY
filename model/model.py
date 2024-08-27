import copy

import torch
import torch.nn as nn
import pytorch_lightning as pl
import config
from model.Simplying_encoder import Simply_encode, MLP
import torch.nn.functional as F

# Set the device based on the specified GPU in the configuration.
DEVICE = torch.device(f"cuda:{config.gpu}")

class LayerNorm(nn.Module):
    """Layer normalization module."""

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # Scale parameter
        self.b_2 = nn.Parameter(torch.zeros(features))  # Bias parameter
        self.eps = eps  # Epsilon for numerical stability

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder(nn.Module):
    """Custom encoder with multiple Simply_encode layers."""

    def __init__(self, label_size, n_heads, dropout, N, mult=20):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([Simply_encode(label_size, n_heads, dropout, mult, layer_idx=i) for i in range(N)])
        self.norm = LayerNorm(label_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class CustomCNN(nn.Module):
    """Custom CNN for signal processing."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(1, 7), stride=(1, 2), padding=(0, 2)),
            LayerNorm(14), nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            LayerNorm(14), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(20, 140, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            LayerNorm(7), nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1)),
        )

    def forward(self, x):
        return self.layers(x)

def beta_nll_loss(mean, variance, target, beta=1):
    """Beta Negative Log-Likelihood Loss."""
    loss = 0.5 * ((target - mean) ** 2 / variance + torch.log(variance))
    return loss.mean() if beta == 0 else (loss * (variance.detach() ** beta)).mean()

def Faithful_Heteroscedastic(output, label):
    mean, var = output
    loss = F.mse_loss(mean, label) / 2 + F.gaussian_nll_loss(mean.detach(), label, var)
    return loss


class ModifiedAttention(nn.Module):
    def __init__(self, d_k, layer_idx):
        super(ModifiedAttention, self).__init__()
        self.value = nn.Linear(d_k, d_k)
        self.query = nn.Linear(d_k, d_k)
        self.key = nn.Linear(d_k, d_k)
        self.mv = nn.Linear(d_k, d_k)
        self.layer_idx = layer_idx

    def forward(self, src, mean):
        value = self.value(src)
        if self.layer_idx:
            attn = F.softmax(self.query(src), 1)
            var = F.softmax(attn, -1) * value
        else:
            var = (F.softmax(self.query(mean), -1)) * value
        var = self.mv(var)
        return var


class ModifiedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, idx, dropout=0.1):
        super(ModifiedTransformerEncoderLayer, self).__init__()
        self.modified_attention = ModifiedAttention(d_model, idx)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.mlp = MLP(dim_feedforward, d_model, dropout)

    def forward(self, src, mean):
        src2 = self.norm1(src)
        src2 = self.modified_attention(src2, mean)
        src2 = src + self.dropout1(src2)
        src3 = self.norm2(src2)
        src3 = self.mlp(src3)
        src = src2 + self.dropout2(src3)

        return src


class ModifiedTransformerEncoder(nn.Module):
    def __init__(self, num_layers, dim=140):
        super(ModifiedTransformerEncoder, self).__init__()
        self.fc = nn.Linear(30, dim)
        self.layers = nn.ModuleList(
            [copy.deepcopy(ModifiedTransformerEncoderLayer(dim, 4 * dim, i)) for i in
             range(num_layers)])
        self.norm = nn.LayerNorm(dim, 1e-6)

    def forward(self, src, mean):
        src = self.fc(src)
        for layer in self.layers:
            src = layer(src, mean)
        src = self.norm(src)
        return src


class DReaM_net(pl.LightningModule):

    def __init__(self, lr=config.lr, n_layers=config.n_layers, n_heads=config.n_heads, label_size=config.label_size,
                 signal_dim=config.signal_dim, dropout=config.dropout, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.CNN = CustomCNN().to(DEVICE)

        self.encoder_var = ModifiedTransformerEncoder(self.hparams.n_layers, self.hparams.label_size).to(DEVICE)

        self.encoder_mean = Encoder(self.hparams.label_size, self.hparams.n_heads, self.hparams.dropout,
                                    self.hparams.n_layers).to(DEVICE)
        self.use_alternate_var = True

    def var_ones(self):
        self.use_alternate_var = False

    def forward(self, src):
        src = src / src[:, :, 0].unsqueeze(2)
        mean = src.unsqueeze(1)
        mean = self.CNN(mean).transpose(1, 2).reshape(mean.shape[0], mean.shape[2], -1)
        mean = self.encoder_mean(mean)
        if self.use_alternate_var:
            var = self.encoder_var(src, mean.detach())
            var = torch.exp(var)

        else:
            var = torch.ones_like(mean)
        output = torch.stack([mean, var], 0)
        return output

    def training_step(self, batch, batch_idx):
        train_input, train_label = batch
        out = self(train_input)
        loss = Faithful_Heteroscedastic(out, train_label)
        # loss = F.mse_loss(out, train_label)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        train_input, train_label = batch
        mean, var = self(train_input)
        loss = beta_nll_loss(mean, var, train_label, beta=0)
        mse_loss = F.mse_loss(mean, train_label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True)

    def prepare_for_testing(self):
        """Prepare the model for testing."""
        self.cpu().eval().freeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = self.choose_scheduler(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def choose_scheduler(self, optimizer):
        if config.lr_scheduler == 'ReduceLR':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3,
                                                              verbose=True)
        elif config.lr_scheduler == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.cos_T0, T_mult=2,
                                                                        eta_min=1e-6, verbose=True)


def make_model():
    return DReaM_net().to(DEVICE)
