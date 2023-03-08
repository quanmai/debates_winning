import torch
import pytorch_lightning as pl
# from pytorch_lightning import Trainer
from utils.config import config
from model.trainer import Train_GraphConversation

if __name__ == "__main__":
    model = Train_GraphConversation(config)
    trainer_config = dict(
        fast_dev_run=False,
        max_epochs=2,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        accelerator='cpu',
        enable_progress_bar = False,
    )

    trainer = pl.Trainer(**trainer_config)
    trainer.fit(model)
    trainer.validate(model)
    # trainer.validate(model=model, dataloaders=val_dataloaders)