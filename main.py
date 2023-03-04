import torch
import pytorch_lightning as pl
# from pytorch_lightning import Trainer
from utils.config import config
from model.trainer import Train_GraphConversation

if __name__ == "__main__":
    model = Train_GraphConversation(config)
    
    trainer = pl.Trainer(fast_dev_run=True, accelerator='cpu')
    trainer.fit(model)