
import argparse
from collections import defaultdict
import yaml
import torch
import pytorch_lightning as pl
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils.import_task import load_model_builder, load_dataset_builder
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from .model_wrapper import ModelWrapper
from .utils.callbacks import (
    get_model_checkpoint_callback, 
    get_learning_rate_monitor_callback, 
    get_early_stopping_callback,
    get_rich_progress_bar_callback
)
from .utils.merge_datasets import merge_datasets
from .utils.utils import hertz_to_mel, mel_to_hertz
from .explainability_module import ExplainabilityModule


def _denorm(x, max_db):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
    y = x * std + mean
    y *= max_db
    y -= max_db
    return y

def parse_slice_times(audio_path):
    audio_path = os.path.basename(audio_path)
    audio_path_no_ext = os.path.splitext(audio_path)[0]
    prefix, slice_part = audio_path_no_ext.rsplit('_slice_', 1)
    start_str, end_str = slice_part.split('-')

    start, end = float(start_str), float(end_str)

    return start, end

def plot_spec_relevance(cfg, recording_id, dataset_name, slices_info, model, dataset, output_dir, unique_label_names, max_db):

    df = dataset.df
    subset = df[df['recording_id'] == recording_id]

    subset["slice_start_tmp"] = subset["path"].apply(lambda p: parse_slice_times(p)[0])
    subset = subset.sort_values(by="slice_start_tmp")

    spectrograms = []
    relevances = []
    time_spans = []
    yticks_pixels = None
    ytickslabels_hz = None
    for _, row in subset.iterrows():
        path = row['path']

        mask = df['path'] == path
        idx = df.index[mask].tolist()[0]

        sample, label = dataset[idx]
        explainability_module = ExplainabilityModule(model.model, dataset, {}, cfg)
        r = explainability_module.generate_single_relevance_map(sample)
        xticks_frames, xticklabels_sec, yticks_pixels, ytickslabels_hz = explainability_module.generate_plotting_axes()

        raw_spectrogram = _denorm(sample, max_db)
        spec = raw_spectrogram[0]

        spectrograms.append(spec)
        relevances.append(r)

    fig, (ax_spec, ax_rel) = plt.subplots(
        2, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    current_offset = 0

    for spec, rel in zip(spectrograms, relevances):
        
        ax_spec.imshow(spec, origin='lower', aspect='auto', cmap='turbo', extent=[current_offset, current_offset + 0.1, 0, spec.shape[0]])
        ax_rel.imshow(rel, origin='lower', aspect='auto', cmap='magma', extent=[current_offset, current_offset + 0.1, 0, rel.shape[0]])

        current_offset += 0.1

    xticks = np.linspace(0, current_offset + 0.1, num=11)

    ax_rel.set_xticks(xticks)
    ax_rel.set_xticklabels([f"{t:.2f}" for t in xticks])
    ax_rel.set_xlabel("Time (s)")

    for ax in [ax_spec, ax_rel]:
        ax.set_yticks(yticks_pixels)
        ax.set_yticklabels([f"{hz:.0f}" for hz in ytickslabels_hz])
        ax.set_ylabel("Frequency (Hz)")

    ax_spec.set_title(f"Recording {recording_id} - Spectrogram with relevance", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(output_dir, f'relevance_spectrogram_{recording_id}.png')
    plt.savefig(out_path)
    plt.close(fig)


def main(run_name, run_number):

    run_root = os.path.join('runs', run_name, 'cnnResnet18', 'cnnSpectrogram', str(run_number))
    hparams_path = os.path.join(run_root, 'fold_0', 'hparams.yaml')

    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)
        cfg = hparams['cfg']

    seed = cfg['seed']
    pl.seed_everything(seed, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = (device == "cuda")

    test_dataset_name = hparams['test_dataset_name']
    dataset_builder = load_dataset_builder(cfg['data']['name'])
    test_dataset = dataset_builder.build_dataset(cfg, augment=cfg['data']['augment_test'], type="test", dataset_name=test_dataset_name)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=pin_memory,
        persistent_workers=cfg['training']['num_workers'] > 0
    )

    df = test_dataset.df
    unique_label_names = test_dataset.unique_label_names
    idx_to_label = test_dataset.idx_to_label
    num_classes = len(unique_label_names)

    model_builder = load_model_builder(cfg['model']['name'])

    recording_wrong_preds = defaultdict(list)

    for fold in sorted(os.listdir(run_root)):

        # For now only for fold_0
        if fold != "fold_0":
            continue

        fold_number = int(fold.split('_')[-1])
        full_fold_path = os.path.join(run_root, fold)
        print(f"Testing fold from {full_fold_path}")
        ckpt_path = os.path.join(full_fold_path, 'checkpoints', f'best_model_fold_{fold_number}.ckpt')
        backbone = model_builder.build_model(cfg=cfg, num_classes=num_classes)
        best_model = ModelWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=backbone,
            cfg=cfg,
            class_weights=None,
            label_names=unique_label_names,
            idx_to_label=idx_to_label,
            train_dataset_name=None,
            test_dataset_name=test_dataset_name,
            strict=False
        )

        best_model.to(device)
        best_model.eval()

        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dataset):
                row = df.iloc[idx]
                audio_path = row["path"]
                recording_id = row["recording_id"]

                start, end = parse_slice_times(audio_path)

                x_batch = x.unsqueeze(0).to(device)  # (1, C, F, T)

                logits = best_model(x_batch)
                probs = F.softmax(logits, dim=1)[0]

                true_idx = int(y) if not isinstance(y, torch.Tensor) else int(y.item())
                pred_idx = int(torch.argmax(probs).item())
                prob_list = probs.cpu().numpy().tolist()

                recording_wrong_preds[recording_id].append({
                    'slice_start': start,
                    'slice_end': end,
                    'true_label': idx_to_label[true_idx],
                    'predicted_label': idx_to_label[pred_idx],
                    'probs': prob_list
                })

    errors_per_recording = {}
    for rec_id, infos in recording_wrong_preds.items():
        errors = sum(1 for info in infos if info["true_label"] != info["predicted_label"])
        errors_per_recording[rec_id] = errors
    
    sorted_recs = sorted(errors_per_recording.items(), key=lambda x: x[1], reverse=True)

    print("Top 10 worst recordings by number of errors:")
    for rec_id, errors in sorted_recs[:10]:
        total = len(recording_wrong_preds[rec_id])
        print(f"Recording {rec_id}: {errors} / {total} errors")

    output_dir = os.path.join("relevance_maps", run_name, str(run_number))

    os.makedirs(output_dir, exist_ok=True)
    
    for rec_id, errors in sorted_recs[:10]:
        slices_info = recording_wrong_preds[rec_id]
        plot_spec_relevance(
            cfg,
            rec_id,
            hparams['test_dataset_name'],
            slices_info,
            best_model,
            test_dataset,
            output_dir,
            unique_label_names,
            cfg['data']['max_db']
        )



def parse_args():
    parser = argparse.ArgumentParser(description="Training script for audio classification using PyTorch Lightning.")
    parser.add_argument('--run-name', required=True, type=str)
    parser.add_argument('--run-number', required=True, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(run_name=args.run_name, run_number=args.run_number)
