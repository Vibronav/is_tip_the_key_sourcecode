import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from .utils.metrics import get_classification_metrics
from .utils.losses import get_loss
from .utils.optimizers import get_optimizer, configure_scheduler
from torchmetrics.classification import MulticlassConfusionMatrix

class ModelWrapper(pl.LightningModule):

    def __init__(self, model: nn.Module, cfg, class_weights, label_names, idx_to_label, train_dataset_name, valid_dataset_name=None):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.label_names = label_names
        self.idx_to_label = idx_to_label
        self.register_buffer('class_weights', class_weights)
        self.num_classes = len(label_names)
        self.criterion = get_loss(cfg, class_weights=class_weights)

        self.train_metrics = get_classification_metrics(cfg, self.num_classes, prefix="train/")
        self.val_metrics = get_classification_metrics(cfg, self.num_classes, prefix="val/")
        self.test_metrics = get_classification_metrics(cfg, self.num_classes, prefix="test/")

        self.val_cm = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize='true')
        self.test_cm = MulticlassConfusionMatrix(num_classes=self.num_classes, normalize='true')

        self.save_hyperparameters({
            "cfg": cfg,
            "label_names": label_names,
            "idx_to_label": idx_to_label,
            "train_dataset_name": train_dataset_name,
            "valid_dataset_name": valid_dataset_name if valid_dataset_name else train_dataset_name
        })

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        _, _, x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.train_metrics(preds=preds, target=y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0), sync_dist=True)

        return loss
    
    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self._log_metrics(metrics)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        _, _, x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.val_metrics(preds=preds, target=y)
        self.val_cm.update(preds, y)
        
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0), sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        self._log_metrics(metrics)
        self.val_metrics.reset()
        try:
            cm = self.val_cm.compute().detach().cpu()
            self._log_confusion_matrix(cm, tag_prefix="val")
        finally:
            self.val_cm.reset()

    def test_step(self, batch, batch_idx):
        _, _, x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.test_metrics(preds=preds, target=y)
        self.test_cm.update(preds, y)

        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0), sync_dist=True)

        return loss
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self._log_metrics(metrics)
        self.test_metrics.reset()
        try:
            cm = self.test_cm.compute().detach().cpu()
            self._log_confusion_matrix(cm, tag_prefix="test")
        finally:
            self.test_cm.reset()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.cfg, self.parameters())
        scheduler = configure_scheduler(self.cfg, optimizer)
        if scheduler is None:
            return optimizer
        else:
            return scheduler
    
    def _log_metrics(self, metrics):
        to_log = {}
        for k, v in metrics.items():

            if isinstance(v, torch.Tensor) and v.ndim > 0:
                for i, val in enumerate(v):
                    class_name = self.label_names[i] if i < len(self.label_names) else str(i)
                    to_log[f'{k}_{class_name}'] = val
            else:
                to_log[k] = v

        to_log['step'] = self.current_epoch

        self.log_dict(to_log, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    
    def _log_fig(self, fig, tag):
        try:
            buf = io.BytesIO()
            fig.tight_layout()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            img_tensor = TF.to_tensor(img)

            logger = getattr(self, "logger", None)
            if logger is None or not hasattr(logger, "experiment"):
                return
            if getattr(self, "global_rank", 0) != 0:
                return

            xp = logger.experiment
            step = self.current_epoch
            if hasattr(xp, "add_image"):
                xp.add_image(tag, img_tensor, global_step=step)
        except Exception as e:
            self.print("Warn: logging plot failed", e)
    
    def _log_confusion_matrix(self, cm, tag_prefix='val'):
        fig = plt.figure(figsize=(4.5, 4))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest')
        ax.set_title('Normalized Confusion Matrix')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_xticks(range(len(self.label_names)))
        ax.set_yticks(range(len(self.label_names)))
        ax.set_xticklabels(self.label_names, rotation=45, ha='right')
        ax.set_yticklabels(self.label_names)
        for (i, j), v in np.ndenumerate(np.asarray(cm)):
            ax.text(j, i, f'{v:.2f}' if cm.dtype != int else f'{int(v)}',
                    ha='center', va='center', fontsize=8)
        
        self._log_fig(fig, f'{tag_prefix}/confusion_matrix_normalized')

    def _plot_rmap(self, r_map, xticks_frames, xticklabels_sec, yticks_pixels, ytickslabels_hz, tag, title):
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)
        im = ax.imshow(r_map, origin='lower', aspect='auto', cmap='magma')
        ax.set_xticks(xticks_frames)
        ax.set_xticklabels([f"{t:.2f}" for t in xticklabels_sec])
        ax.set_yticks(yticks_pixels)
        ax.set_yticklabels([f"{hz:.0f}" for hz in ytickslabels_hz])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        self._log_fig(fig, tag)

    def _plot_r_wrt_freq(self, r_wrt_freq, yticks_pixels, ytickslabels_hz, tag, title):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(r_wrt_freq)
        ax.set_xticks(yticks_pixels)
        ax.set_xticklabels([f"{hz:.0f}" for hz in ytickslabels_hz])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Relevance score")
        ax.set_title(title)

        self._log_fig(fig, tag)

    def _plot_hist(self, max_freqs, yticks_pixels, ytickslabels_hz, tag, title):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(max_freqs)
        ax.set_xticks(yticks_pixels)
        ax.set_xticklabels([f"{hz:.0f}" for hz in ytickslabels_hz])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(title)

        self._log_fig(fig, tag)

    def _plot_violin(self, max_freqs, yticks_pixels, ytickslabels_hz, tag, title):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.violinplot(max_freqs, showmeans=True)
        ax.set_yticks(yticks_pixels)
        ax.set_yticklabels([f"{hz:.0f}" for hz in ytickslabels_hz])
        ax.set_title(title)

        self._log_fig(fig, tag)
