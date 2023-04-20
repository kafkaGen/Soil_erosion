from .dataset import SoilErosionDataset
from .train import train

import torch
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
            #A.OneOf([
            #   A.ColorJitter(brightness=[0.8, 1.2], hue=0.05),
            #   A.RandomBrightnessContrast(brightness_limit=[0.05, 0.1], contrast_limit=[0.05, 0.2]),
            #   A.RandomGamma(gamma_limit=[80, 120], eps=1e-4)
            #], p=0.7),
            #A.HorizontalFlip(),
            #A.OneOf([
            #   A.ShiftScaleRotate(),
            #   A.ElasticTransform(alpha=30, sigma=3, alpha_affine=5)
            #], p=1.0),
            #A.Blur(blur_limit=3, p=0.2),
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