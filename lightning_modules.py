import lightning.pytorch as pl
import torch
from torch import nn
from utils.accuracy_metrics import *


class LitModelWrapper(pl.LightningModule):
    def __init__(self, model, loss_config, optim_config):
        super().__init__()
        # wrap model passed in
        self.model = model
        # instance loss 
        self.loss_fn = eval(loss_config['loss_fn'])()
        # will instance optimizer in configure_optimizers method
        self.optim_config=optim_config
        self.acc = multilabel_accuracy
    

    def forward(self, batch):
        x, y = batch
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size=len(y)
        logits = self.model(x) #in logits form
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        accuracy = self.acc(logits, y)
        self.log ('train_accuracy', accuracy, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size=len(y)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss, sync_dist = True)
        accuracy = self.acc(logits, y)
        self.log ('validation_accuracy', accuracy, batch_size=batch_size)
      
    def test_step(self, batch, batch_idx):
        x, y = batch
        batch_size=len(y)
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log('test_loss', loss, sync_dist = True)
        accuracy = self.acc(logits, y)
        self.log ('test_accuracy', accuracy, batch_size=batch_size)    

        
    def configure_optimizers(self):
        optim_fn = eval(self.optim_config['optim_fn'])
        optimizer = optim_fn(self.parameters(), **self.optim_config['optim_kwargs'])

        """
        # learning rate scheduler
        sched_fn = eval(self.optim_config["scheduler"]) 
        scheduler = sched_fn(optimizer,  **self.optim_config['scheduler_kwargs'])
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "name":"learning_rate"
        }
        optimizer_dict = {"optimizer" : optimizer,
                          "lr_scheduler" : scheduler_config
         }
         """
        return optimizer