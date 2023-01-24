import torch
from .utils.data_reader import load_dataset
from .utils.data_loader import Dataset
from .model.model import GraphArguments

from pytorch_lightning.core.lightning import LightningModule

class Train_GraphConversation(LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        
        #TODO: implement utils.load_dataset()
        train_data, val_data, test_data = load_dataset()
        self.train, self.val, self.test = Dataset(train_data), Dataset(val_data), Dataset(test_data)
        self.model = GraphArguments(config)

    def configure_optimizers(self):
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr)
        if len(self.config.scheduler) == 0:
            return optimizer
        elif self.config.scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.config.lr_decay)
            return optimizer, scheduler
        else: #cyclic
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, self.config.base_lr, self.config.max_lr, cycle_momentum=False)
            return optimizer, scheduler

    def forward(self, batch):
        self.model(batch)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss