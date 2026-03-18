import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torchmetrics import R2Score
from joblib import load

# =====================
# Gamma Loss (for positive targets)
# =====================

class GammaNLLLoss(nn.Module):
    """
    Assumes prediction outputs mean (mu > 0)
    Learns log-variance implicitly via shape parameter k
    Simplified Gamma NLL
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=self.eps)

        # Gamma NLL (simplified)
        loss = target / pred + torch.log(pred)
        return loss.mean()


# =====================
# Blocks
# =====================

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ContractionBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        conv_out = self.conv(x)
        return conv_out, self.pool(conv_out)


class ExpansionBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, skip, x):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# =====================
# Model (R2-optimized)
# =====================

class LitURNet3D(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-5, batch_size=5):
        super().__init__()
        self.save_hyperparameters()

        # Data
        self.df = pd.read_csv('cleaned_df_5_31.csv')
        self.hgb_pipe = load('/home/madar/unet2021/hgb_model.joblib')

        # Encoder
        self.cb1 = ContractionBlock3D(1, 16)
        self.cb2 = ContractionBlock3D(16, 32)
        self.cb3 = ContractionBlock3D(32, 64)
        self.cb4 = ContractionBlock3D(64, 128)

        # Bottleneck
        self.bottleneck = ConvBlock3D(128, 256)

        # Regression head (positive outputs enforced)
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Softplus()  # ensures positivity for Gamma
        )

        # Learnable fusion instead of static GBM weighting
        self.fusion_head = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Decoder
        self.up1 = ExpansionBlock3D(256, 128)
        self.up2 = ExpansionBlock3D(128, 64)
        self.up3 = ExpansionBlock3D(64, 32)
        self.up4 = ExpansionBlock3D(32, 16)

        self.seg_head = nn.Conv3d(16, 4, kernel_size=1)

        # Loss
        self.gamma_loss = GammaNLLLoss()

        # Metrics
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()

    # =====================
    # Forward
    # =====================

    def forward(self, x):
        c1, x = self.cb1(x)
        c2, x = self.cb2(x)
        c3, x = self.cb3(x)
        c4, x = self.cb4(x)

        x = self.bottleneck(x)

        reg_out = self.reg_head(x)

        x = self.up1(c4, x)
        x = self.up2(c3, x)
        x = self.up3(c2, x)
        x = self.up4(c1, x)

        seg_out = self.seg_head(x)

        return reg_out, seg_out

    # =====================
    # Training
    # =====================

    def training_step(self, batch, batch_idx):
        X, y_img, y_adas, filenames = batch

        X = X.unsqueeze(1)
        y_img = y_img.squeeze(1).long()
        y_adas = y_adas.float().view(-1, 1)

        reg_out, seg_out = self(X)

        fused_pred = self.fuse_with_tabular(reg_out, filenames)

        seg_loss = F.cross_entropy(seg_out, y_img)
        reg_loss = self.gamma_loss(fused_pred, y_adas)

        loss = seg_loss + reg_loss

        r2 = self.train_r2(fused_pred.view(-1), y_adas.view(-1))

        self.log_dict({
            'train_loss': loss,
            'train_reg_loss': reg_loss,
            'train_r2': r2
        }, prog_bar=True)

        return loss

    # =====================
    # Validation and Testing
    # =====================

    def validation_step(self, batch, batch_idx):
        X, y_img, y_adas, filenames = batch

        X = X.unsqueeze(1)
        y_img = y_img.squeeze(1).long()
        y_adas = y_adas.float().view(-1, 1)

        reg_out, seg_out = self(X)

        fused_pred = self.fuse_with_tabular(reg_out, filenames)

        seg_loss = F.cross_entropy(seg_out, y_img)
        reg_loss = self.gamma_loss(fused_pred, y_adas)

        loss = seg_loss + reg_loss

        r2 = self.val_r2(fused_pred.view(-1), y_adas.view(-1))

        self.log_dict({
            'val_loss': loss,
            'val_reg_loss': reg_loss,
            'val_r2': r2
        }, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y_img, y_adas, filenames = batch

        X = X.unsqueeze(1)
        y_adas = y_adas.view(-1, 1)

        reg_out, _ = self(X)
        pred = self.fuse_with_tabular(reg_out, filenames)

        r2 = self.val_r2(pred.view(-1), y_adas.view(-1))
        self.log("test_r2", r2, prog_bar=True)

    # =====================
    # Fusion (learned)
    # =====================

    def fuse_with_tabular(self, reg_out, filenames):
        unet_preds = reg_out.detach().cpu().numpy().flatten()

        tab_preds = []
        for i, f in enumerate(filenames):
            row = self.df[self.df['filenames'] == f]
            if len(row) > 0:
                x = row.drop(columns=['filenames', 'ADAS11', 'MMSE'])
                tab_preds.append(self.hgb_pipe.predict(x)[0])
            else:
                tab_preds.append(unet_preds[i])

        tab_preds = torch.tensor(tab_preds, dtype=torch.float32, device=self.device).unsqueeze(1)
        unet_preds = reg_out

        fused = torch.cat([unet_preds, tab_preds], dim=1)
        return self.fusion_head(fused)

    # =====================
    # Optimizer
    # =====================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)
        return [optimizer], [scheduler]

    # =====================
    # Data
    # =====================

    def setup(self, stage=None):
        from CogDataset3d import get_ds_dl
        self.ds_train, self.ds_val, self.ds_test, \
        self.dl_train, self.dl_val, self.dl_test = get_ds_dl(
            batch_size=self.hparams.batch_size
        )

    def train_dataloader(self):
        return self.dl_train

    def val_dataloader(self):
        return self.dl_val

    def test_dataloader(self):
        return self.dl_test


# =====================
# Trainer
# =====================

if __name__ == '__main__':
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint(
        monitor='val_r2',
        mode='max',
        save_top_k=3
    )

    model = LitURNet3D()

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=45,
        precision=16,
        gradient_clip_val=1.0,
        callbacks=[checkpoint]
    )

    trainer.fit(model)
