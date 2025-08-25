import random 
import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random

import torchvision
import math
import os 
import tqdm
from Modules.models.dn3_ext import BENDR, ConvEncoderBENDR
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

from utils import *
from sklearn import metrics
from utils_eval import get_metrics


class LitEEGPTCausal(pl.LightningModule):

    def __init__(self):
        super().__init__()    
        encoder = ConvEncoderBENDR(20, encoder_h=512, dropout=0., projection_head=False)
        encoder.load("Modules/models/encoder.pt")
        
        self.model = encoder
        self.scale_param    = torch.nn.Parameter(torch.tensor(1.))
        
        self.downsample     = torch.nn.Conv1d(64, 19, 3, stride=1)
        
        self.linear_probe   = torch.nn.Linear(3072, 4)
        
        self.drop           = torch.nn.Dropout(p=0.10)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True
        
    
    def forward(self, x):
        B, C, T = x.shape
        x = self.downsample(x)
        x = torch.cat([x, self.scale_param.repeat((x.shape[0], 1, x.shape[-1]))], dim=-2)
        
        h = self.model(x)
        
        h = h.flatten(1)
        h = self.drop(h)
        pred = self.linear_probe(h)
        return x, pred
    def on_train_epoch_start(self) -> None:
        self.running_scores["train"]=[]
        return super().on_train_epoch_start()
    def on_train_epoch_end(self) -> None:
        label, y_score = [], []
        for x,y in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('train_rocauc', rocauc, on_epoch=True, on_step=False)
        return super().on_train_epoch_end()
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        
        loss = self.loss_fn(logit, label)
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            list([self.scale_param])+
            list(self.downsample.parameters())+
            list(self.model.parameters())+
            list(self.linear_probe.parameters()),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        
# load configs
# -- LOSO 

global max_epochs
global steps_per_epoch
global max_lr

class P300SubjectDataset(Dataset):
    def __init__(self, root, subjects):
        """
        root: path to PhysioNetP300 folder
        subjects: list of subject numbers to include, e.g. [1,2,3]
        """
        self.files = []
        self.labels = []

        for label in ['0','1']:
            folder = os.path.join(root, label)
            for f in os.listdir(folder):
                if f.endswith('.pt') and any(f".sub{s}" in f for s in subjects):
                    self.files.append(os.path.join(folder, f))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.files[idx])
        y = self.labels[idx]
        x = x.float()  # Convert to float32
        return x, y

batch_size=64
max_epochs = 100


all_subjects = [1,2,3,4,5,6,7,9,11]
for val_sub in (all_subjects):

    train_subs = [s for s in all_subjects if s != val_sub] # all except the current
    train_dataset = P300SubjectDataset(root="../datasets/downstream/PhysioNetP300", subjects=train_subs)
    val_dataset = P300SubjectDataset(root="../datasets/downstream/PhysioNetP300", subjects=[val_sub])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    steps_per_epoch = math.ceil(len(train_loader))

    # init model
    model = LitEEGPTCausal()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    max_lr = 1e-5
    trainer = pl.Trainer(accelerator='cuda',
                        max_epochs=max_epochs, 
                        callbacks=callbacks,
                        logger=[pl_loggers.TensorBoardLogger('./logs/', name="BENDR_PhysioP300_tb", version=f"subject{val_sub}"), 
                                pl_loggers.CSVLogger('./logs/', name="BENDR_PhysioP300_csv")])

    trainer.fit(model, train_loader, valid_loader, ckpt_path='last')
