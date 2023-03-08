import torch
from utils.data_reader import load_dataset
from utils.data_loader import ArgDataset, collate_fn
from utils.loss import PairBCELoss, PairHingeLoss
from model.model import GraphArguments
from utils.helpers import acc_score
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule

class Train_GraphConversation(LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        train_data, val_data, test_data = load_dataset()
        self.train_data, self.val_data, self.test_data = ArgDataset(train_data), ArgDataset(val_data), ArgDataset(test_data)
        print('HeyHeyHey')
        self.model = GraphArguments(config)
        self.loss = PairHingeLoss()
        self.acc_metric = acc_score

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

    def forward(self, g):
        return self.model(g)

    def training_step(self, batch, batch_idx):
        g, y = batch
        s1, s2 = self(g) # = self.forward(g)
        # element-wise product: y*s, 
        # y=1 (winner) -> s, y=-1 (loser) -> -s
        s1 = s1 * y
        s1 = s2 * y
        loss = self.loss(s1, s2)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        g, y = batch
        s1, s2 = self(g)
        print(f's1, s2: {s1,s2}')
        s1 = s1 * y
        s1 = s2 * y
        loss = self.loss(s1, s2)
        pred = (s1>s2).int() #.squueze()
        acc = self.acc_metric(y, pred)

        return {'val_loss': loss,
                'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        print('val_loss: {:.4f}, val_acc: {:.4f}'.format(loss.item(), acc))
        return {'val_loss': loss.item(), 'val_acc': acc}

    def epoch_end(self, epoch, result):
        print('Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}'.format( \
            epoch, result['val_loss,'], result['val_acc']))

    def train_dataloader(self):
        print('Train dataloader')
        train_loader = DataLoader(self.train_data, shuffle=self.config.shuffle, num_workers=self.config.num_workers, \
                        batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=False)
        return train_loader

    def val_dataloader(self):
        print('Val dataloader')
        val_loader = DataLoader(self.val_data, shuffle=False, num_workers=self.config.num_workers, \
                        batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=False)
        return val_loader    