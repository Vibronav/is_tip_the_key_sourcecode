import argparse
import yaml
import torch
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from .utils.import_task import load_model_builder, load_dataset_builder

from .data_module import AudioDataModule
from .model_wrapper import ModelWrapper
from .utils.callbacks import (
    get_model_checkpoint_callback, 
    get_learning_rate_monitor_callback, 
    get_early_stopping_callback,
    get_rich_progress_bar_callback
)
from .utils.merge_datasets import merge_datasets
from .explainability_module import ExplainabilityModule
from .utils.random_context import random_state_context_torch
from .utils.seeds import MODEL_SEED, EXPLANATION_SEED


def main(config_file_path, train_dataset_name, train_dataset_name2, shaft_dataset_name, shaft_dataset_name2, valid_dataset_name=None):

    with open(config_file_path, 'r') as f:
        cfg = yaml.safe_load(f)

    seed = cfg['seed']
    pl.seed_everything(seed, workers=True)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    merged_train_dataset_name = merge_datasets(cfg, train_dataset_name, train_dataset_name2)
    merged_train_shaft_dataset_name = merge_datasets(cfg, shaft_dataset_name, shaft_dataset_name2)

    dataset_builder = load_dataset_builder(cfg['data']['name'])
    model_builder = load_model_builder(cfg['model']['name'])

    dm = AudioDataModule(
        cfg, 
        dataset_builder=dataset_builder, 
        train_dataset_name=merged_train_dataset_name,
        train_shaft_dataset_name=merged_train_shaft_dataset_name,
        valid_dataset_name=valid_dataset_name
    )
    dm.setup()
    unique_label_names = dm.train_dataset.unique_label_names
    idx_to_label = dm.train_dataset.idx_to_label

    # Model
    with random_state_context_torch(seed + MODEL_SEED):
        backbone = model_builder.build_model(cfg=cfg, num_classes=len(unique_label_names))

    model_wrapper = ModelWrapper(
        model=backbone, 
        cfg=cfg, 
        class_weights=dm.class_weights,
        label_names=unique_label_names,
        idx_to_label=idx_to_label,
        train_dataset_name=merged_train_dataset_name,
        valid_dataset_name=valid_dataset_name
    )

    logger = TensorBoardLogger(
        save_dir=cfg.get('logs_dir', 'runs'),
        name=f'{cfg["data"]["name"]}/{cfg.get("run_suffix", None)}'
    )

    callbacks = [
        get_model_checkpoint_callback(cfg),
        get_early_stopping_callback(cfg),
        get_learning_rate_monitor_callback(cfg),
        get_rich_progress_bar_callback()
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        deterministic=True
    )

    trainer.fit(model_wrapper, dm)

    best_model = ModelWrapper.load_from_checkpoint(
        checkpoint_path=trainer.checkpoint_callback.best_model_path,
        model=backbone,
        class_weights=dm.class_weights
    )

    # Always validate checkpoint after training to see how best model perform on validation set
    print("Validating on the best checkpoint")
    trainer.validate(best_model, dataloaders=dm.val_dataloader())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    
    with random_state_context_torch(seed + EXPLANATION_SEED):
        explainability_module = ExplainabilityModule(best_model.model, dm.train_dataset, idx_to_label, cfg)
        xticks_frames, xticklabels_sec, yticks_pixels, ytickslabels_hz = explainability_module.generate_plotting_axes()
        r_by_class, r_max_freqs_by_class = explainability_module.generate_relevance_maps()
        sal_by_class, sal_max_freqs_by_class = explainability_module.generate_saliency_maps()

    for class_idx, r_maps in r_by_class.items():
        r_map_class = np.mean(r_maps, axis=0)
        class_name = idx_to_label[class_idx]
        best_model._plot_rmap(
            r_map_class,
            xticks_frames,
            xticklabels_sec,
            yticks_pixels,
            ytickslabels_hz,
            tag=f"relevance_maps/{class_name}/r_map",
            title=f"Relevance Map - {class_name}"
        )
        r_wrt_freq = np.mean(r_map_class, axis=1)
        best_model._plot_r_wrt_freq(
            r_wrt_freq,
            yticks_pixels,
            ytickslabels_hz,
            tag=f"relevance_maps/{class_name}/r_wrt_freq",
            title=f"Relevance vs Frequency - {class_name}"
        )
        max_freqs_class = r_max_freqs_by_class[class_idx]
        best_model._plot_hist(
            max_freqs_class,
            yticks_pixels,
            ytickslabels_hz,
            tag=f"relevance_maps/{class_name}/hist",
            title=f"Histogram: most relevant frequency - {class_name}"
        )
        best_model._plot_violin(
            max_freqs_class,
            yticks_pixels,
            ytickslabels_hz,
            tag=f"relevance_maps/{class_name}/violin",
            title=f"Violin plot: most relevant frequency - {class_name}"
        )

    for class_idx, sal_maps in sal_by_class.items():
        sal_map_class = np.mean(sal_maps, axis=0)
        class_name = idx_to_label[class_idx]
        best_model._plot_rmap(
            sal_map_class,
            xticks_frames,
            xticklabels_sec,
            yticks_pixels,
            ytickslabels_hz,
            tag=f"saliency_maps/{class_name}/sal_map",
            title=f"Saliency Map - {class_name}"
        )
        r_wrt_freq = np.mean(sal_map_class, axis=1)
        best_model._plot_r_wrt_freq(
            r_wrt_freq,
            yticks_pixels,
            ytickslabels_hz,
            tag=f"saliency_maps/{class_name}/sal_wrt_freq",
            title=f"Saliency vs Frequency - {class_name}"
        )
        max_freqs_class = sal_max_freqs_by_class[class_idx]
        best_model._plot_hist(
            max_freqs_class,
            yticks_pixels,
            ytickslabels_hz,
            tag=f"saliency_maps/{class_name}/hist",
            title=f"Histogram: most relevant frequency - {class_name}"
        )
        best_model._plot_violin(
            max_freqs_class,
            yticks_pixels,
            ytickslabels_hz,
            tag=f"saliency_maps/{class_name}/violin",
            title=f"Violin plot: most relevant frequency - {class_name}"
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for audio classification using PyTorch Lightning.")
    parser.add_argument('--config', required=True, type=str,
                        help='Path to the configuration file')
    parser.add_argument('--train-dataset-name', required=True, type=str,
                        help='Name of the training dataset')
    parser.add_argument('--train-dataset-name2', required=True, type=str,
                        help='Name of the second training dataset. Train datasets will be merged')
    parser.add_argument('--shaft-dataset-name', required=True, type=str,
                        help='Name of the shaft training dataset')
    parser.add_argument('--shaft-dataset-name2', required=True, type=str,
                        help='Name of the second shaft training dataset')
    parser.add_argument('--valid-dataset-name', required=False, type=str,
                        help='Name of the validation dataset, if not provided, validation will be done on train dataset')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(config_file_path=args.config, train_dataset_name=args.train_dataset_name, train_dataset_name2=args.train_dataset_name2,
         shaft_dataset_name=args.shaft_dataset_name, shaft_dataset_name2=args.shaft_dataset_name2,
         valid_dataset_name=args.valid_dataset_name)
