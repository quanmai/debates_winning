import os
import torch
import pytorch_lightning as pl
# from pytorch_lightning import Trainer
from utils.config import config
from model.trainer import Train_GraphConversation
#from model.callbacks import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
import glob
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    model = Train_GraphConversation(config)
    logger = pl.loggers.TensorBoardLogger(
        # save_dir='./run100' if config.test_ver else '.'
        save_dir='./run100' if config.run100 or config.test_ver else '.'
    )

    ckpt_args = dict(
        monitor='val_loss',
        mode='min',
    )

    early_stopping = pl.callbacks.EarlyStopping(
        patience=config.patience,
        strict=True,
        verbose=True,
        # warm_up = 5,
        **ckpt_args
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        # filepath=os.path.join(logger.log_dir, '{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}'),
        dirpath=logger.log_dir,
        filename='{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-{val_f1:.3f}',
        save_top_k=1,
        verbose=True,
        **ckpt_args
    )


    trainer_config = dict(
        num_sanity_val_steps=0,
        fast_dev_run=False,
        max_epochs=config.epoch,
        callbacks=[early_stopping,ckpt_callback],
        check_val_every_n_epoch=config.check_val_freq,
        logger=logger,
        log_every_n_steps=1,
        deterministic=False,
        accelerator=config.accelerator,
        devices=1,
        precision=config.precision,
        # enable_progress_bar = False,
    )
    if isinstance(trainer_config['devices'], list) and len(trainer_config['devices']) > 1:
        trainer_config['strategy'] = "ddp_find_unused_parameters_false"
        print(trainer_config['strategy'])
    print(f'lr = {config.lr}')
    if config.accelerator=='gpu':
        torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(**trainer_config)

    if config.test_ver == 0:
        trainer.fit(model)
        trainer.validate(model)
        ckpt_callback.best_model_path
        trainer.test(model,ckpt_path='best') #last
    else: # test mode
        ver = 'version_' + str(config.test_ver)
        path = 'lightning_logs/' + ver
        dir = os.path.join(path, '*ckpt')
        test_model = glob.glob(dir)[0]
        model = model.load_from_checkpoint(test_model)
        trainer.test(model)
    print(logger.log_dir)