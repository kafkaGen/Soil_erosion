import torch
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler

from settings.config import Config
from utils.lightning import SoilErosionDataModule, SoilErosionSegmentation
from utils import CosineAnnealingWarmupRestarts

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--max_lr', default=Config.learning_rate, type=float)
    parser.add_argument('--min_lr', default=3e-4, type=float)
    parser.add_argument('--first_cycle_steps', default=25, type=int)
    parser.add_argument('--cycle_mult', default=0.85, type=float)
    parser.add_argument('--warmup_steps', default=8, type=int)
    parser.add_argument('--gamma', default=0.75, type=float)
    parser.add_argument('--pos_weight', default=7, type=int)
    parser.add_argument('--max_epochs', default=Config.epochs, type=int)
    parser.add_argument('--model_name', default='model', type=str)
    args = parser.parse_args()

    datamodule = SoilErosionDataModule()
    unet = smp.Unet(encoder_name='efficientnet-b5', encoder_depth=3, encoder_weights='imagenet',
                            decoder_channels=[512, 256, 128], classes=Config.num_classes,
                            decoder_attention_type='scse', decoder_use_batchnorm=True)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=Config.learning_rate)
    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.first_cycle_steps, cycle_mult=args.cycle_mult, 
                                                 max_lr=args.max_lr, min_lr=args.min_lr, warmup_steps=args.warmup_steps, gamma=args.gamma)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight))
    model = SoilErosionSegmentation(unet, optimizer, criterion, lr_scheduler=lr_scheduler)

    logger = TensorBoardLogger(Config.logs_path, name=args.model_name)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    checkpoint = ModelCheckpoint(dirpath=Config.model_path, filename=args.model_name+'_{val_IoU:.2f}', 
                                monitor='val_IoU', save_top_k=1, mode='max')
    profiler = PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profilers"),
            schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
        )

    trainer = pl.Trainer(accelerator=Config.accelerator, logger=logger, max_epochs=args.max_epochs,
                        log_every_n_steps=1, track_grad_norm=2, profiler=profiler,
                        callbacks=[early_stopping, checkpoint])
    trainer.fit(model=model, datamodule=datamodule)