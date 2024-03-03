import torch
import torch.nn as nn
import pytorch_lightning as pl
import config
from Simplying_encoder import Simply_encode
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


class Transformer(pl.LightningModule):
    """Transformer model combining CustomCNN and Encoder for signal processing."""

    def __init__(self, lr=config.lr, n_layers=config.n_layers, n_heads=config.n_heads, label_size=config.label_size,
                 signal_dim=config.signal_dim, dropout=config.dropout, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.CNN = CustomCNN().to(DEVICE)
        self.fc = nn.Linear(self.hparams.signal_dim, self.hparams.label_size).to(DEVICE)
        self.encoder_mean = Encoder(self.hparams.label_size, self.hparams.n_heads, self.hparams.dropout,
                                    self.hparams.n_layers).to(DEVICE)
        self.encoder_var = Encoder(self.hparams.label_size, self.hparams.n_heads, self.hparams.dropout,
                                   self.hparams.n_layers, 4).to(DEVICE)
        self.use_alternate_var = True

    def var_ones(self):
        self.use_alternate_var = False

    def forward(self, src):
        src = src / src[:, :, 0].unsqueeze(2)
        mean = src.unsqueeze(1)
        mean = self.CNN(mean).transpose(1, 2).reshape(mean.shape[0], mean.shape[2], -1)
        mean = self.encoder_mean(mean)
        if self.use_alternate_var:
            var = self.fc(src)
            var = self.encoder_var(var)
            var = F.softplus(var) + 1e-6
        else:
            var = torch.ones_like(mean)

        return mean, var

    def training_step(self, batch, batch_idx):
        train_input, train_label = batch
        out, var = self(train_input)
        loss = beta_nll_loss(out, var, train_label)
        # loss = F.mse_loss(out, train_label)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        train_input, train_label = batch
        out, var = self(train_input)
        loss = beta_nll_loss(out, var, train_label, beta=0)
        loss_mse = F.mse_loss(out, train_label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("mse_loss", loss_mse, on_step=False, on_epoch=True, prog_bar=True)

    def prepare_for_testing(self):
        """Prepare the model for testing."""
        self.cpu().eval().freeze()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = self.choose_scheduler(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def choose_scheduler(self, optimizer):
        if config.lr_scheduler == 'ReduceLR':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                                              verbose=True)
        elif config.lr_scheduler == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.cos_T0, T_mult=2,
                                                                        eta_min=1e-6, verbose=True)


def make_model():
    # model = Transformer().load_from_checkpoint('Result/DOSY/epoch=39-step=22520.ckpt').to(DEVICE)
    return Transformer().to(DEVICE)
