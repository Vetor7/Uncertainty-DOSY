import torch
import torch.nn as nn
import pytorch_lightning as pl
import config
from Simplying_encoder import Simply_encode
import torch.nn.functional as F

DEVICE = torch.device(f"cuda:{config.gpu}")
torch.manual_seed(42)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # Initialize α to 1, and β to 0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # Smoothing items
        self.eps = eps

    def forward(self, x):
        # Calculate the mean and variance by the last dimension
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # Return the result of Layer Norm
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class Encoder(nn.Module):
    def __init__(self, label_size, n_heads, dropout, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([Simply_encode(label_size, n_heads, dropout, layer_idx=i) for i in range(N)])
        self.norm = LayerNorm(label_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, 7), stride=(1, 2), padding=(0, 2)),
            LayerNorm(14),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            LayerNorm(14),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=20, out_channels=140, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            LayerNorm(7),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1)),
        )

    def forward(self, x):
        return self.layers(x)


class Transformer(pl.LightningModule):

    def __init__(self, lr=config.lr, n_layers=config.n_layers, n_heads=config.n_heads, label_size=config.label_size,
                 dropout=config.dropout, signal_dim=config.signal_dim,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(self.hparams.label_size, self.hparams.n_heads, self.hparams.dropout,
                               self.hparams.n_layers).to(DEVICE)
        self.encoder2 = Encoder(self.hparams.label_size, self.hparams.n_heads, self.hparams.dropout,
                                self.hparams.n_layers//2).to(DEVICE)
        self.linear_en = nn.Linear(self.hparams.signal_dim, self.hparams.label_size).to(DEVICE)
        self.fc = nn.Linear(self.hparams.label_size + self.hparams.signal_dim, self.hparams.label_size).to(DEVICE)
        self.CNN = CustomCNN().to(DEVICE)
        self.CNN2 = CustomCNN().to(DEVICE)
        self.freeze_encoder2 = True

        for layer in [self.encoder2, self.linear_en]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, src):

        src = src / (src[:, :, 0].unsqueeze(2))
        batch, channel, dim = src.size()
        src2 = src
        src = src.unsqueeze(1)
        src = self.CNN(src).transpose(1, 2).reshape(batch, channel, -1)
        out = self.encoder(src)
        if self.freeze_encoder2:
            var = torch.tensor(1)
        else:
            # src2 = self.CNN2(src2).transpose(1, 2).reshape(batch, channel, -1)
            src2 = torch.concat([src2, src], -1)
            var = self.encoder2(self.fc(src2))
            var = F.softplus(var) + 1e-6
        return out, var

    def heteroscedastic_loss(self, target, mean, variance):

        loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())
        return loss.mean()

    def beta_nll_loss(self, mean, variance, target, beta=0.5):
        loss = 0.5 * ((target - mean) ** 2 / variance + variance.log())

        if beta > 0:
            loss = loss * (variance.detach() ** beta)

        return loss.mean()

    def on_validation_epoch_end(self):
        if self.current_epoch == 39:
            self.freeze_encoder2 = False
            self._unfreeze_encoder2()
            self.reset_optimizer()
            for param in self.CNN.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        train_input, train_label = batch
        out, var = self(train_input)
        if not self.freeze_encoder2:
            loss = self.heteroscedastic_loss(out, train_label, var)
        else:
            loss = F.mse_loss(out, train_label)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):

        train_input, train_label = batch
        out, var = self(train_input)
        loss = F.mse_loss(out, train_label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def prepare_for_testing(self):
        """
        Prepare the model for testing by unfreezing encoder2.
        """
        self.freeze_encoder2 = False
        self.cpu()
        self.eval()
        self.freeze()

    def _unfreeze_encoder2(self):
        for layer in [self.encoder2, self.linear_en]:
            for param in layer.parameters():
                param.requires_grad = True

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)
        if config.lr_scheduler == 'ReduceLR':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5,
                                                                   verbose=True)  # TODO
        elif config.lr_scheduler == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.cos_T0, verbose=True,
                                                                             T_mult=2, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def reset_optimizer(self):

        new_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.trainer.optimizers = [new_optimizer]


def make_model():
    model = Transformer()
    return model.to(DEVICE)
