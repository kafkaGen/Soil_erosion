import numpy as np
import torch
import torchmetrics

from settings.config import Config


def train(model, optimizer, criterion, train_loader, valid_loader, lr_scheduler=None, model_name='model',
          num_epochs=Config.epochs, valid_iou_max=0, device=Config.device):
      callbackers = {}
      callbackers['train_loss'] = []
      callbackers['train_batch_loss'] = []
      callbackers['valid_loss'] = []
      callbackers['valid_iou'] = []
      callbackers['valid_batch_loss'] = []
      callbackers['valid_batch_iou'] = []
      callbackers['predictions_proba'] = []
      callbackers['targets'] = []
      IoU = torchmetrics.JaccardIndex(task='binary', num_classes=Config.num_classes).to(device)
      model.to(device)
      
      for e in np.arange(num_epochs):
            train_loss = 0.0
            valid_loss = 0.0
            valid_iou = 0.0
            model.train()
            
            for imgs, masks in train_loader:
                  imgs = imgs.to(device)
                  masks = masks.to(device)
                  
                  out = model(imgs)
                  prediction = torch.round(torch.sigmoid(out))
                  loss = criterion(out, masks)
                  
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  
                  callbackers['train_batch_loss'].append(loss.item()*imgs.shape[0])
                  train_loss += loss.item()*imgs.shape[0]
                  
            model.eval()
            with torch.no_grad():
                  for imgs, masks in valid_loader:
                        imgs = imgs.to(device)
                        masks = masks.to(device)
                        
                        out = model(imgs)
                        prediction = torch.round(torch.sigmoid(out))
                        loss = criterion(out, masks)
                        
                        iou = IoU(prediction, masks).item()
                        callbackers['predictions_proba'].extend(torch.sigmoid(out).to('cpu').numpy())
                        callbackers['targets'].extend(masks.clone().to('cpu').numpy())
                        callbackers['valid_batch_loss'].append(loss.item()*imgs.shape[0])
                        callbackers['valid_batch_iou'].append(iou)
                        valid_loss += loss.item()*imgs.shape[0]
                        valid_iou += iou
            
            if lr_scheduler:
                  lr_scheduler.step(valid_loss)
                        
            train_loss /= len(train_loader.sampler)
            valid_loss /= len(valid_loader.sampler)
            valid_iou /= len(valid_loader.sampler)
            
            print(f'Epoch {e+1}/{num_epochs}: TrainLoss {train_loss:.3f} ValidLoss: {valid_loss:.3f} ValidIoU: {valid_iou:.3f}')
            
            callbackers['train_loss'].append(train_loss)
            callbackers['valid_loss'].append(valid_loss)
            callbackers['valid_iou'].append(valid_iou)
            
            if valid_iou > valid_iou_max:
                  #script = model.to_torchscript()
                  #torch.jit.save(script, f"{Config.model_path}/{model_name}.pt")
                  torch.save(model.state_dict(), f"{Config.model_path}/{model_name}.pt")
                  valid_iou_max = valid_iou
                  
      return callbackers
  