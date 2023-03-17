import os
import torch
import pytorch_lightning as pl
# from pytorch_lightning import Trainer
from utils.config import config
from model.trainer import Train_GraphConversation

if __name__ == "__main__":
    model = Train_GraphConversation(config)
    logger = pl.loggers.TensorBoardLogger(
        save_dir='.'
    )

    ckpt_args = dict(
        monitor='val_loss',
        mode='min',
    )

    early_stopping = pl.callbacks.EarlyStopping(
        patience=config.patience,
        strict=True,
        verbose=True,
        **ckpt_args
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        # filepath=os.path.join(logger.log_dir, '{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}'),
        dirpath=logger.log_dir,
        filename='{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k=1,
        verbose=True,
        **ckpt_args
    )


    trainer_config = dict(
        num_sanity_val_steps=0,
        fast_dev_run=False,
        max_epochs=config.epoch,
        callbacks=[early_stopping,ckpt_callback],
        check_val_every_n_epoch=1,
        logger=logger,
        log_every_n_steps=1,
        deterministic=False,
        accelerator=config.device,
        devices=1,
        precision=config.precision,
        # enable_progress_bar = False,
    )

    trainer = pl.Trainer(**trainer_config)
    # if config.device=='gpu':
    #     torch.set_float32_matmul_precision('medium')

    print('Hey')
    trainer.fit(model)
    print('HeyHey')
    trainer.validate(model)
    print('HeyHeyHey')
    # trainer.test(model, ckpt_path=logger.log_dir) #ogger.log_dir
    ckpt_callback.best_model_path
    trainer.test(model,ckpt_path='best')