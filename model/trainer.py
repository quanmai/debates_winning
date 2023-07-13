import torch
from utils.reader import load_dataset
from utils.loader import Dataset, collate_fn
from utils.loss import PairBCELoss
from model.model import GraphGRUArgument
from utils.helpers import acc_score, f1_score
from torch.utils.data import DataLoader
from utils.config import config
from pytorch_lightning import LightningModule


class Train_GraphConversation(LightningModule):
    def __init__(self, config=config):
        super().__init__()
        self.config = config
        # train_data, val_data, test_data, vocab = load_dataset()
        train_data, val_data, test_data, vocab_train, vocab_dev, vocab_test= load_dataset()
        self.train_data = Dataset(train_data, vocab_train, config.embed_f_train)
        self.val_data = Dataset(val_data, vocab_dev, config.embed_f_dev)
        self.test_data = Dataset(test_data, vocab_test, config.embed_f_test)
        self.model = GraphGRUArgument(config)
        self.loss = (PairBCELoss() 
                     if config.loss == 'pair' 
                     else torch.nn.BCEWithLogitsLoss())
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

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        y = batch['label']
        s1, s2 = self(batch)
        if self.config.loss == 'pair':
            if s2.shape != y.shape:
                s1 = torch.reshape(s1, (y.shape))
                s2 = torch.reshape(s2, (y.shape))
            loss = self.loss(s1, s2, y)
        else: #binary
            s2 = torch.reshape(s2, (y.shape))
            loss = self.loss(s2,y.float())
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        y = batch['label']
        s1, s2 = self(batch)
        if self.config.loss == 'pair':
            if s2.shape != y.shape:
                s1 = torch.reshape(s1, (y.shape))
                s2 = torch.reshape(s2, (y.shape))
            loss = self.loss(s1, s2, y)
            pred = torch.where(s1>=s2, 1., -1.)
        else:
            s2 = torch.reshape(s2, (y.shape))
            loss = self.loss(s2,y.float())
            pred = torch.sigmoid(s2).round()

        acc = self.acc_metric(y, pred)
        f1 = self.f1_metric(y, pred)
        log = {'val_loss': loss, 'val_acc': acc, 'val_f1': f1}
        self.log_dict(log,
                      batch_size=y.shape[0],
                      sync_dist=True if len(self.config.device) > 1 else False)
        return log

    def test_step(self, batch, batch_idx):
        y = batch['label'] 
        s1, s2 = self(batch)

        if self.config.loss == 'pair':
            if s2.shape != y.shape:
                s1 = torch.reshape(s1, (y.shape))
                s2 = torch.reshape(s2, (y.shape))
            loss = self.loss(s1, s2, y)
            pred = torch.where(s1>=s2, 1., -1.)
            print(f's1: {s1}')
        else:
            s2 = torch.reshape(s2, (y.shape))
            loss = self.loss(s2,y.float())
            # pred = torch.round(s2)
            pred = torch.sigmoid(s2).round()
        print(f's2: {s2}')
        print(f'y: {y}')
        print(f'p: {pred}')
        acc = self.acc_metric(y, pred)
        f1 = self.f1_metric(y, pred)
        log = {'test_loss': loss, 
               'test_acc': acc, 
               'test_f1': f1}

        self.log_dict(log,
                      batch_size=y.shape[0],
                      sync_dist=True if len(self.config.device) > 1 else False)
        return log

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