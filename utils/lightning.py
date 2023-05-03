import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.metrics import precision_recall_curve, average_precision_score

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision
import torchmetrics

from utils.dataset import SoilErosionDataset
from utils import get_transforms
from settings.config import Config


class SoilErosionDataModule(pl.LightningDataModule):
    def __init__(self, data_path=Config.data_path, transforms=get_transforms, batch_size=Config.batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.transforms = transforms
        
    def setup(self, stage):
        self.datasets = {name: SoilErosionDataset(name, self.transforms(name)) for name in ['train', 'valid', 'test']}
        
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=Config.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.datasets['valid'], batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=Config.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=Config.num_workers)
    
    
class SoilErosionSegmentation(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, lr_scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.learning_rate = Config.learning_rate
        self.batch_size = Config.batch_size
        self.IoU = torchmetrics.classification.BinaryJaccardIndex()
        self.CF = torchmetrics.ConfusionMatrix(task='binary', num_classes=Config.num_classes)
        self.PRC = torchmetrics.classification.BinaryPrecisionRecallCurve()
        self.AP = torchmetrics.classification.BinaryAveragePrecision()
        
        self.batch_freq = {'Train': 30, 'Validation': 5}
        self.plot_step = {'Train': 0, 'Validation': 0, 'Test': 0}
        self.example_input_array = torch.zeros(Config.batch_size, 3, Config.resize_to, Config.resize_to)
        #self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        loss, outputs, masks = self._common_step(batch, batch_idx)
        predictions = torch.round(torch.sigmoid(outputs))
        
        if batch_idx % self.batch_freq['Train'] == 0:
            self.plot2tb(predictions, outputs, batch, stage='Train')
            self.plot_step['Train'] += 1
        
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, logger=True)
        return {'loss': loss, 'outputs': outputs, 'masks': masks}
    
    def training_epoch_end(self, outputs):
        outs = torch.cat([el['outputs'] for el in outputs])
        masks = torch.cat([el['masks'] for el in outputs])
        predictions = torch.round(torch.sigmoid(outs))
        
        if self.current_epoch == 0:    
            self.logger.experiment.add_graph(self.model, self.example_input_array.to(self.device))
            
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log_dict({'train_IoU': self.IoU(predictions, masks), 'LR': current_lr}, on_epoch=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        loss, outputs, masks = self._common_step(batch, batch_idx)
        predictions = torch.round(torch.sigmoid(outputs))
        IoU = self.IoU(predictions, masks)
        
        if batch_idx % self.batch_freq['Validation'] == 0:
            self.plot2tb(predictions, outputs, batch, stage='Validation')
            self.plot_step['Validation'] += 1
            
        #self.logger.experiment.add_hparams({
        #        'lr': Config.learning_rate,
        #        'batch_size': Config.batch_size,
        #        'optimizer': type(self.optimizer).__name__,
        #        'criterion': type(self.criterion).__name__,
        #        'lr_scheduler': type(self.lr_scheduler).__name__,
        #    },
        #    {
        #        'val_loss': loss,
        #        'val_IoU': IoU
        #    })
        self.log_dict({'val_loss': loss, 'val_IoU': IoU}, on_step=True, on_epoch=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        loss, outputs, masks = self._common_step(batch, batch_idx)
        predictions = torch.round(torch.sigmoid(outputs))
        IoU = self.IoU(predictions, masks)
        
        if batch_idx % self.batch_freq['Validation'] == 0:
            self.plot2tb(predictions, outputs, batch, stage='Test')
            self.plot_step['Test'] += 1
        
        self.log_dict({'test_loss': loss, 'test_IoU': IoU}, on_step=True, on_epoch=True, logger=True)
        
    def predict_step(self, batch, batch_idx):
        imgs, masks = batch
        outputs = self.forward(imgs)
        predictions = torch.round(torch.sigmoid(outputs))
        return predictions
    
    def _common_step(self, batch, batch_idx):
        imgs, masks = batch
        outputs = self.forward(imgs)
        loss = self.criterion(outputs, masks)
        return loss, outputs, masks
    
    def configure_optimizers(self):
        optimizer = self.optimizer
        if self.lr_scheduler:
            scheduler = self.lr_scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
            else:
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}
    
    def plot2tb(self, predictions, outputs, batch, stage):
        # Confusion Matrix & Precision-Recall curve & Image grid
        imgs, masks = batch
        predictions_grid = torchvision.utils.make_grid(predictions)
        predictions_grid = predictions_grid.cpu().permute(1, 2, 0).numpy()
        masks_grid = torchvision.utils.make_grid(masks)
        masks_grid = masks_grid.cpu().permute(1, 2, 0).numpy()
        imgs_grid = torchvision.utils.make_grid(imgs)
        imgs_grid = imgs_grid.cpu().permute(1, 2, 0).numpy() * Config.std + Config.mean
        fig, ax = plt.subplots(2, 1, figsize=(10,7))
        ax[0].imshow(imgs_grid)
        ax[0].imshow(masks_grid, alpha=0.2)
        ax[0].set_xlabel('Target masks')
        ax[0].grid(False)
        ax[1].imshow(imgs_grid)
        ax[1].imshow(predictions_grid, alpha=0.2)
        ax[1].set_xlabel('Predicted masks')
        ax[1].grid(False)
        self.logger.experiment.add_figure(f'{stage} outputs', fig, self.plot_step[stage])
        plt.close()
        
        if stage != 'Train':
            confusion_matrix = self.CF(predictions, masks)
            df_cm = pd.DataFrame(confusion_matrix.cpu().numpy(), index = range(2), columns=range(2))
            plt.figure(figsize = (10,7))
            fig = sns.heatmap(df_cm, annot=True, cmap='binary').get_figure()
            plt.xlabel('Predicted')
            plt.ylabel('True')
            self.logger.experiment.add_figure(f"{stage} Confusion matrix", fig, self.plot_step[stage])
            plt.close()
            
            masks = masks.to(torch.int64)
            precision, recall, _ = self.PRC(outputs, masks)
            ap = self.AP(outputs, masks)
            plt.figure(figsize = (10,7))
            plt.plot(recall.cpu().numpy(), precision.cpu().numpy(), label=f'AP: {ap.cpu().numpy():.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            self.logger.experiment.add_figure(f"{stage} Precision-Recall Curve", plt.gcf(), self.plot_step[stage])
            plt.close()