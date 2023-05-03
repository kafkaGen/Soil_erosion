from .dataset import SoilErosionDataset
from .train import train
 
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from settings.config import Config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
def get_transforms(subset):
    if subset == 'train':
        transforms = A.Compose([
            A.Resize(height=Config.resize_to, width=Config.resize_to),
            A.OneOf([
               A.ColorJitter(brightness=[0.8, 1.2], hue=0.05),
               A.RandomGamma(gamma_limit=[80, 120], eps=1e-4)
            ], p=0.7),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(shift_limit=[0.1, 0.15], scale_limit=0, rotate_limit=0, p=0.7),
            A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=20),
            A.ElasticTransform(alpha=30, sigma=3, alpha_affine=5, p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=Config.mean, std=Config.std),
            ToTensorV2(transpose_mask=True)
        ])
    else:
        transforms = A.Compose([
            A.Resize(height=Config.resize_to, width=Config.resize_to),
            A.Normalize(mean=Config.mean, std=Config.std),
            ToTensorV2(transpose_mask=True)
        ])
    
    return transforms


def get_shuffle(subset):
    if subset == 'train':
        return True
    else:
        return False
    
      
def visualize_callbackers(callbackers):
      fig, ax =  plt.subplots(3, 2, figsize=(20, 12))
      ax[0][0].plot(np.round(callbackers['train_loss'], 5), color='r', linestyle='dashdot', label='Train Loss')
      ax[0][0].plot(np.round(callbackers['valid_loss'], 5), color='b', linestyle=(5, (10,3)), label='Valid Loss')
      ax[0][0].set_title('Epochs Losses')
      ax[0][0].legend()
      ax[0][1].plot(np.round(callbackers['train_batch_loss'], 5), color='r', label='Train Loss', alpha=0.7)
      ax[0][1].plot(np.round(callbackers['valid_batch_loss'], 5), color='b', label='Valid Loss', alpha=0.7)
      ax[0][1].set_title('Batches Losses')
      ax[0][1].legend()
      ax[1][0].plot(np.round(callbackers['valid_iou'], 5), color='b', linestyle=(5, (10,3)), label='Valid IoU')
      ax[1][0].set_title('Epochs IoU')
      ax[1][0].legend()
      ax[1][1].plot(np.round(callbackers['valid_batch_iou'], 5), color='b', label='Valid IoU', alpha=0.7)
      ax[1][1].set_title('Batches IoU')
      ax[1][1].legend()
      
      targets = np.array(callbackers['targets']).flatten()
      predictions_proba = np.array(callbackers['predictions_proba']).flatten()
      sns.heatmap(confusion_matrix(targets, np.round(predictions_proba)), annot=True, ax=ax[2][0])
      ax[2][0].set_title('Confusion Matrix')
      precision, recall, tresh = precision_recall_curve(targets, predictions_proba)
      ap = average_precision_score(targets, predictions_proba)
      ax[2][1].plot(recall, precision, label=f'AP: {ap:.3f}')
      ax[2][1].set_xlabel('Recall')
      ax[2][1].set_ylabel('Precision')
      ax[2][1].set_title('Precision-Recall Curve')
      ax[2][1].legend()
      
      plt.tight_layout()
      plt.show()
      
      
def plot_results(model, dataset, n_rows=8, figsize=(11,25), device='cpu'):
      fig, ax = plt.subplots(n_rows, 3, figsize=figsize)
      imgs, masks = [], []
      for img, mask in dataset:
            imgs.append(img)
            masks.append(mask)
      
      seed = np.random.randint(10_000) 
      rng = np.random.default_rng(seed=seed)
      rng.shuffle(imgs)
      rng = np.random.default_rng(seed=seed)
      rng.shuffle(masks)
      for img, mask in zip(imgs, masks):
            if not n_rows:
                  break
            img = img.to(device)
            mask = mask.to(device)
            model.to(device)
            
            model.eval()
            out = model(torch.unsqueeze(img, 0))
            prediction = torch.squeeze(torch.round(torch.sigmoid(out))).detach().numpy().astype(np.int64)
            n_rows -= 1
            img = torch.squeeze(img).permute(1,2,0).numpy()
            img = img * Config.std + Config.mean
            mask = torch.squeeze(mask).numpy().astype(np.int64)
            ax[n_rows][0].imshow(img)
            ax[n_rows][0].grid(False)
            ax[n_rows][0].axis('off')
            ax[n_rows][0].set_title('Image')
            ax[n_rows][1].imshow(img)
            ax[n_rows][1].imshow(prediction, alpha=0.3)
            ax[n_rows][1].grid(False)
            ax[n_rows][1].axis('off')
            ax[n_rows][1].set_title('Predicted Mask')
            ax[n_rows][2].imshow(img)
            ax[n_rows][2].imshow(mask, alpha=0.3)
            ax[n_rows][2].grid(False)
            ax[n_rows][2].axis('off')
            ax[n_rows][2].set_title('True Mask')
             

def SoftDiceLoss(logits, targets):
    smooth = 1
    num = targets.size(0)
    probs = torch.sigmoid(logits)
    m1 = probs.view(num, -1)
    m2 = targets.view(num, -1)
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / num
    return score


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr