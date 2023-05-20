import torch
from utils.data_reader import load_dataset
from utils.data_loader import ArgDataset, collate_fn
from utils.loss import PairBCELoss, PairHingeLoss
from model.model import GraphGRUArguments
from utils.helpers import acc_score, f1_score
from torch.utils.data import DataLoader
from utils.config import config

import dgl

from pytorch_lightning import LightningModule

class Train_GraphConversation(LightningModule):
    def __init__(self, config=config):
        super().__init__()
        self.config = config
        train_data, val_data, test_data = load_dataset()
        self.train_data, self.val_data, self.test_data = ArgDataset(train_data), ArgDataset(val_data), ArgDataset(test_data)
        self.model = GraphGRUArguments(config)
        if config.loss == 'pair':
            self.loss = PairBCELoss()
            # self.loss = PairHingeLoss() # this always cause negative s1
        elif config.loss == 'ranking':
            self.loss = torch.nn.MarginRankingLoss()
        else: #binary
            self.loss = torch.nn.BCEWithLogitsLoss()
        self.acc_metric = acc_score
        self.f1_metric = f1_score

    def configure_optimizers(self):
        if self.config.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr, momentum=0.9)
        if len(self.config.scheduler) == 0:
            return optimizer
        elif self.config.scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.config.lr_decay)
            return [optimizer], [scheduler]
        else: #cyclic
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, self.config.base_lr, self.config.max_lr, cycle_momentum=False)
            return [optimizer], [scheduler]

    def forward(self, g):
        return self.model(g)

    def training_step(self, batch, batch_idx):
        # print(f'{batch_idx=}')
        g, y = batch
        s1, s2 = self(g) # = self.forward(g)
        # element-wise product: y*s, 
        # y=1 (winner) -> s, y=-1 (loser) -> -s
        if self.config.loss == 'pair':
            loss = self.loss(s1*y, s2*y)
            # print(f's1: {s1}')
            # print(f's2: {s2}')
        elif self.config.loss == 'ranking':
            loss = self.loss(s1, s2, y)
        else: #binary
            s2 = torch.reshape(s2, (y.shape))
            # y_float = y.float()
            # print(f'{s2.item()=}, {y_float.item()=}')
            loss = self.loss(s2,y.float())
            # print(f'{loss.item()=}')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        g, y = batch
        s1, s2 = self(g)
        if self.config.loss == 'pair':
            loss = self.loss(s1*y, s2*y)
            pred = torch.where(s1>s2, 1, -1)
        elif self.config.loss == 'ranking':
            loss = self.loss(s1, s2, y)
            pred = torch.where(s1>s2, 1, -1)
        else:
            s2 = torch.reshape(s2, (y.shape))
            # y_float = y.float()
            # print(f'{s2.item()=}, {y_float.item()=}')
            loss = self.loss(s2,y.float())
            # print(f'{loss.item()=}')
            # pred = torch.round(s2)
            pred = torch.sigmoid(s2).round()

        acc = self.acc_metric(y, pred)
        f1 = self.f1_metric(y, pred)
        log = {'val_loss': loss, 'val_acc': acc, 'val_f1': f1}
        self.log_dict(log)

        return {
                'val_loss': loss,
                'val_acc': acc,
                'val_f1': f1,
                }

    def test_step(self, batch, batch_idx):
        g, y = batch
        s1, s2 = self(g)
        if self.config.loss == 'pair':
            loss = self.loss(s1*y, s2*y)
            pred = torch.where(s1>s2, 1, -1)
            print(f's1: {s1}')
        elif self.config.loss == 'ranking':
            loss = self.loss(s1, s2, y)
            pred = torch.where(s1>s2, 1, -1)
            print(f's1: {s1}')
        else:
            loss = self.loss(s2,y.float())
            # pred = torch.round(s2)
            pred = torch.sigmoid(s2).round()
        print(f's2: {s2}')
        print(f'y: {y}')
        print(f'p: {pred}')
        acc = self.acc_metric(y, pred)
        f1 = self.f1_metric(y, pred)
        return {'test_loss': loss,
                'test_acc': acc,
                'test_f1': f1, 
                }

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        print('val_loss: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f}'.format(loss.item(), acc, f1))
        tensorboard_logs = {
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1,
        }
        return {
            'progress_bar': tensorboard_logs,
            'val_loss': loss.item(), 
            'val_acc': acc,
            'val_f1': f1,
            }

    def test_epoch_end(self, outputs):
        loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        print('test_loss: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), acc, f1))
        return {'test_loss': loss.item(), 
                'test_acc': acc,
                'test_f1':f1,
                }

    def epoch_end(self, epoch, result):
        print('Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}, val_f1: {:.4f}'.format( \
            epoch, result['val_loss,'], result['val_acc'], result['val_f1']))

    def train_dataloader(self):
        train_loader = DataLoader(self.train_data, shuffle=True, num_workers=self.config.num_workers, \
                        batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data, shuffle=False, num_workers=self.config.num_workers, \
                        batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=True)
        return val_loader  

    def test_dataloader(self):
        test_loader = DataLoader(self.test_data, shuffle=False, num_workers=self.config.num_workers, \
                        batch_size=self.config.batch_size, collate_fn=collate_fn, pin_memory=True)
        return test_loader