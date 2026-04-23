import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar

def get_model_checkpoint_callback(cfg):
    return ModelCheckpoint(
        monitor=cfg['training']['monitor'],
        mode=cfg['training']['monitor_mode'],
        save_top_k=1,
        save_last=True,
        filename='best_model-{epoch:02d}'
    )

def get_learning_rate_monitor_callback(cfg):
    return LearningRateMonitor(logging_interval='epoch')

def get_early_stopping_callback(cfg):
    return EarlyStopping(
        monitor=cfg['training']['monitor'],
        patience=cfg['training']['early_stop_patience'],
        mode=cfg['training']['monitor_mode'],
    )

def get_rich_progress_bar_callback():
    return RichProgressBar()