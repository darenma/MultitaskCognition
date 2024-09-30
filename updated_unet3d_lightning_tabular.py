
from imports import *
from CogDataset3d import *
import numpy as np
import h5py
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from torchmetrics import R2Score as _r2score

# Custom gamma loss function (based on squared log error)
def gamma_loss(predt, target):
    predt = torch.clamp(predt, min=1e-6)  # Avoid log of zero
    target = torch.clamp(target, min=1e-6)
    return torch.mean((torch.log1p(predt) - torch.log1p(target)) ** 2)

class LitURNet3d(pl.LightningModule):
    def __init__(self,
                 in_channels=1,
                 num_classes=4, 
                 batch_size=PARAMS['batch_size'],
                 lr=PARAMS['learning_rate'],
                 weight_decay=PARAMS['weight_decay']):
        super(LitURNet3d, self).__init__()
        
        self.automatic_optimization = False
        self.df = pd.read_csv('cleaned_df_5_31.csv')
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.final_w1_epoch = 0.5
        self.final_w2_epoch = 0.5

        # UNet Architecture
        self.cb1 = contraction_block3d(in_ch=1, out_ch=16)
        self.cb2 = contraction_block3d(in_ch=16, out_ch=32)
        self.cb3 = contraction_block3d(in_ch=32, out_ch=64)
        self.cb4 = contraction_block3d(in_ch=64, out_ch=128)
        self.convb = conv_block3d(in_ch=128, out_ch=256)
        self.regb = regression_block3d(in_ch=256, out_ch=128)  
        self.dropout = nn.Dropout(0.15)

        # Cognition, alpha, and diagnosis predictions
        self.cognition_head = nn.Linear(128*8*8, 1)  # Output for cognition prediction
        self.alpha_head = nn.Linear(128*8*8, 1)      # Output for alpha prediction
        self.diagnosis_head = nn.Linear(128*8*8, 2)  # Output for diagnosis (2 classes for binary classification)
        
        self.exb1 = expansion_block3d(in_ch=256, out_ch=128)
        self.exb2 = expansion_block3d(in_ch=128, out_ch=64)
        self.exb3 = expansion_block3d(in_ch=64, out_ch=32)
        self.exb4 = expansion_block3d(in_ch=32, out_ch=16)

        self.final_conv = nn.Conv3d(in_channels=16, out_channels=num_classes, kernel_size=1)
        
    def forward(self, x):
        # Contraction path
        c1 = self.cb1(x)
        c2 = self.cb2(c1)
        c3 = self.cb3(c2)
        c4 = self.cb4(c3)
        
        # Bottom block
        bottleneck = self.convb(c4)
        bottleneck_flat = torch.flatten(bottleneck, start_dim=1)
        
        # Multi-task heads: cognition, alpha, diagnosis
        cognition_output = self.cognition_head(bottleneck_flat)
        alpha_output = self.alpha_head(bottleneck_flat)
        diagnosis_output = self.diagnosis_head(bottleneck_flat)
        
        # Expansion path (can be used if segmentation tasks are still relevant)
        e1 = self.exb1(bottleneck)
        e2 = self.exb2(e1)
        e3 = self.exb3(e2)
        e4 = self.exb4(e3)
        final_output = self.final_conv(e4)

        return cognition_output, alpha_output, diagnosis_output
    
    def training_step(self, batch, batch_idx):
        x, y_cognition, y_alpha, y_diagnosis = batch
        
        # Forward pass
        cognition_pred, alpha_pred, diagnosis_pred = self(x)
        
        # Losses for each task
        cognition_loss = gamma_loss(cognition_pred, y_cognition)
        alpha_loss = gamma_loss(alpha_pred, y_alpha)
        diagnosis_loss = F.cross_entropy(diagnosis_pred, y_diagnosis)
        
        # Total loss (can use a weighted combination)
        total_loss = cognition_loss + alpha_loss + diagnosis_loss

        self.log('train_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
