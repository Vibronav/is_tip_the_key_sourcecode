
import argparse
import yaml
from collections import defaultdict
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
import glob

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


def parse_slice_times(audio_path):
    audio_path = os.path.basename(audio_path)
    audio_path_no_ext = os.path.splitext(audio_path)[0]
    prefix, slice_part = audio_path_no_ext.rsplit('_slice_', 1)
    start_str, end_str = slice_part.split('-')

    start, end = float(start_str), float(end_str)

    return start, end


def get_recording(rec_id, test_dataset_name, test_dataset):
    recording_path = os.path.join(test_dataset.data_path, test_dataset_name, 'whole', f"{rec_id}.wav")
    signal, sr = torchaudio.load(recording_path)
    signal = signal[0:0+1, :]
    return signal, sr


def reconstruct_recording(rec_id, test_dataset):
    df = test_dataset.df
    subset = df[df['recording_id'] == rec_id]

    end_times = []
    slices_meta = []
    for _, row in subset.iterrows():
        rel_path = row['path']
        start, end = parse_slice_times(rel_path)
        end_times.append(end)
        slices_meta.append((rel_path, start, end))

    slices_meta = sorted(slices_meta, key=lambda x: x[1])

    full_path = os.path.join(test_dataset.data_path, slices_meta[0][0])

    signal, sr = torchaudio.load(full_path)
    signal = signal[0:0+1, :]

    previos_signal_end = slices_meta[0][2]

    for rel_path, start, end in slices_meta[1:]:
        print(f"Reconstructing slice {rel_path} from {start} to {end}")
        full_path = os.path.join(test_dataset.data_path, rel_path)
        slice_signal, _ = torchaudio.load(full_path)
        slice_signal = slice_signal[0:0+1, :]

        offset_sec = previos_signal_end - start
        offset_samples = int(offset_sec * sr)
        signal_to_add = slice_signal[:, offset_samples:]
        signal = torch.cat((signal, signal_to_add), dim=1)
        previos_signal_end = end

    return signal, sr


def plot_spectrogram(
        cfg,
        rec_id,
        test_dataset_name,
        slices_info,
        test_dataset,
        output_dir,
        label_colors,
        label_names,
        model
    ):

    # signal, sr = reconstruct_recording(rec_id, test_dataset)
    signal, sr = get_recording(rec_id, test_dataset_name, test_dataset)

    signal = test_dataset.bandpass(signal)

    signal = test_dataset._normalize_signal(signal)

    specs = test_dataset._convert_to_spectrograms(signal)
    spec = specs[0]

    F, T = spec.shape
    duration_sec = signal.shape[-1] / sr

    n_ffts = cfg["data"]["n_ffts"]
    sr = cfg["data"]["target_sr"]
    hop_length = cfg["data"]["hop_length"]
    n_mels = cfg["data"]["n_mels"]
    spec_type = cfg["data"]['spectrogram_type']
    min_freq = cfg["data"]["min_freq"]
    max_freq = cfg["data"]["max_freq"]

    # Frequency axis calculation for mel
    if spec_type == "mel":
        yticks_pixels = np.linspace(0, n_mels - 1, num=10)
        min_mel = hertz_to_mel(min_freq)
        max_mel = hertz_to_mel(max_freq)
        mel_values = min_mel + (yticks_pixels / (n_mels - 1)) * (max_mel - min_mel)
        ytickslabels_hz = mel_to_hertz(mel_values)

    # Frequency axis calculation for linear
    elif spec_type == "linear":
        plotted_freq_bins = spec.shape[0]

        yticks_real = np.linspace(0, plotted_freq_bins - 1, num=5)
        ytickslabels_hz = min_freq + (yticks_real / (plotted_freq_bins - 1)) * (max_freq - min_freq)
        yticks_pixels = yticks_real

    # Time axis calculation
    time_frames = spec.shape[1]
    xticks_frames = np.linspace(0, time_frames - 1, num=6)
    xticklabels_sec = (xticks_frames * hop_length) / sr

    min_start_time = min(sl['slice_start'] for sl in slices_info)
    max_end_time = max(sl['slice_end'] for sl in slices_info)

    fig, (ax_spec, ax_prob, ax_rel) = plt.subplots(
        3, 1, figsize=(18, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 2]}
    )

    min_time = max(0, min_start_time - 2)
    max_time = min(max_end_time + 2, duration_sec)
    im = ax_spec.imshow(
        spec,
        origin='lower',
        aspect='auto',
        cmap='turbo',
        extent=[0, duration_sec, 0, F]
    )
    ax_spec.set_ylabel('Mel Frequency Bins')
    ax_spec.set_yticks(yticks_pixels)
    ax_spec.set_yticklabels([f"{int(freq)}" for freq in ytickslabels_hz])
    ax_spec.set_xticks(xticks_frames)
    ax_spec.set_xticklabels([f"{t:.2f}" for t in xticklabels_sec])
    ax_spec.set_xlim(min_time, max_time)
    ax_spec.set_title(f'Spectrogram for recording {rec_id}', pad=40)

    for sl in slices_info:
        start = sl['slice_start']
        end = sl['slice_end']
        true_label = sl['true_label']
        predicted_label = sl['predicted_label']

        correct = (true_label == predicted_label)
        color = label_colors[predicted_label]

        ax_spec.axvspan(
            start,
            end,
            ymin=0.9,
            ymax=1.0,
            color=color,
            alpha=0.25 if correct else 0.1
        )

        mid = 0.5 * (start + end)
        ax_spec.text(
            mid, 1.0, predicted_label, ha='center', va='bottom',
            rotation=90, fontsize=8, color=color, transform=ax_spec.get_xaxis_transform()
        )

    times_per_class = {label: [] for label in label_names}
    probs_per_class = {label: [] for label in label_names}
    for sl in slices_info:
        start = sl['slice_start']
        end = sl['slice_end']
        mid_t = 0.5 * (start + end)
        probs = sl['probs']

        for class_idx, prob in enumerate(probs):
            label = label_names[class_idx]
            times_per_class[label].append(mid_t)
            probs_per_class[label].append(prob)

    for label in label_names:
        color = label_colors[label]
        ax_prob.plot(
            times_per_class[label],
            probs_per_class[label],
            marker='o',
            linestyle='-',
            color=color,
            label=label
        )

    ax_prob.set_xlabel('Time (s)')
    ax_prob.set_ylabel('Predicted Probability')
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlim(min_time, max_time)
    ax_prob.legend(loc='upper right', ncol=1)
    ax_prob.grid(alpha=0.3)

    # Relevanse map plotting
    df = test_dataset.df
    subset = df[df['recording_id'] == rec_id]

    subset["slice_start_tmp"] = subset["path"].apply(lambda p: parse_slice_times(p)[0])
    subset = subset.sort_values(by="slice_start_tmp")

    device = next(model.parameters()).device

    yticks_pixels = None
    ytickslabels_hz = None
    for _, row in subset.iterrows():
        path = row['path']
        start, end = parse_slice_times(path)

        mask = df['path'] == path
        idx = df.index[mask].tolist()[0]

        _, _, sample, label = test_dataset[idx]
        sample = sample.to(device)
        explainability_module = ExplainabilityModule(model.model, test_dataset, {}, cfg)
        r = explainability_module.generate_single_relevance_map(sample)
        xticks_frames, xticklabels_sec, yticks_pixels, ytickslabels_hz = explainability_module.generate_plotting_axes()

        ax_rel.imshow(
            r,
            origin='lower',
            aspect='auto',
            cmap='magma',
            extent=[start, end, 0, r.shape[0]]
        )

    ax_rel.set_yticks(yticks_pixels)
    ax_rel.set_yticklabels([f"{hz:.0f}" for hz in ytickslabels_hz])
    ax_rel.set_ylabel('Frequency (Hz)')
    ax_rel.set_xlabel('Time (s)')
    ax_rel.set_xlim(min_time, max_time)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'recording_{rec_id}_spectrogram.png')
    plt.tight_layout()
    plt.savefig(out_path)


def main(run_root, run_name, run_number, fold_number, dataset_name, cfg, folder_suffix):

    seed = cfg['seed']
    pl.seed_everything(seed, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = (device == "cuda")

    dataset_builder = load_dataset_builder(cfg['data']['name'])
    dataset = dataset_builder.build_dataset(
        cfg,
        augment=cfg['data']['augment_test'],
        type="test",
        dataset_name=dataset_name
    )

    if cfg["data"]["normalizing_method"] == "global":
        dataset.update_normalization_params(np.arange(len(dataset)))

    dataset_loader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=pin_memory,
        persistent_workers=cfg['training']['num_workers'] > 0
    )

    unique_label_names = dataset.unique_label_names
    idx_to_label = dataset.idx_to_label
    num_classes = len(unique_label_names)

    model_builder = load_model_builder(cfg['model']['name'])

    recording_wrong_preds = {r: [] for _, r, _, _ in dataset}

    for fold in sorted(os.listdir(run_root)):

        if fold != f"version_{fold_number}":
            continue

        fold_number = int(fold.split('_')[-1])
        full_fold_path = os.path.join(run_root, fold)
        print(f"Testing fold from {full_fold_path}")
        ckpt_pattern = os.path.join(full_fold_path, 'checkpoints', 'best_model-epoch=*.ckpt')
        ckpt_files = glob.glob(ckpt_pattern)
        if ckpt_files:
            ckpt_path = ckpt_files[0]
        else:
            raise ValueError(f"No checkpoint found")
        backbone = model_builder.build_model(cfg=cfg, num_classes=num_classes)
        best_model = ModelWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model=backbone,
            cfg=cfg,
            class_weights=None,
            label_names=unique_label_names,
            idx_to_label=idx_to_label,
            train_dataset_name=None,
            test_dataset_name=dataset_name,
            strict=False
        )

        best_model.to(device)
        best_model.eval()

        with torch.no_grad():
            for audio_paths, recordings_id, batch_x, batch_y in dataset_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = best_model(batch_x)
                preds = torch.argmax(logits, dim=1)
                probs = F.softmax(logits, dim=1)

                for i in range(len(batch_y)):
                    true_idx = batch_y[i].item()
                    pred_idx = preds[i].item()
                    prob = probs[i].cpu().numpy().tolist()

                    start, end = parse_slice_times(audio_paths[i])

                    recording_id = recordings_id[i]
                    recording_wrong_preds[recording_id].append({
                        'slice_start': start,
                        'slice_end': end,
                        'true_label': idx_to_label[true_idx],
                        'predicted_label': idx_to_label[pred_idx],
                        'probs': prob
                    })

    label_colors = {
        "Chicken": 'orange',
        "Zucchini": 'blue',
    }

    errors_per_recording = {}
    for rec_id, infos in recording_wrong_preds.items():
        errors = sum(1 for info in infos if info["true_label"] != info["predicted_label"])
        errors_per_recording[rec_id] = errors

    sorted_recs = sorted(errors_per_recording.items(), key=lambda x: x[1], reverse=True)

    worst_recording_indices = []

    if folder_suffix == 'test_dataset':
        print("Top 10 worst recordings by number of errors:")
        for rec_id, errors in sorted_recs[:10]:
            total = len(recording_wrong_preds[rec_id])
            print(f"Recording {rec_id}: {errors} / {total} errors")
            worst_recording_indices.append(rec_id)
    else:
        i = 0
        print("Top 5 worst recordings for chicken:")
        for rec_id, errors in sorted_recs:
            slices_info = recording_wrong_preds[rec_id]
            if np.all([info["true_label"] == "Chicken" for info in slices_info]):
                total = len(recording_wrong_preds[rec_id])
                print(f"Recording {rec_id}: {errors} / {total} errors")
                worst_recording_indices.append(rec_id)
                i += 1
            
            if i == 5:
                break

        i = 0
        print("Top 5 worst recordings for zucchini:")
        for rec_id, errors in sorted_recs:
            slices_info = recording_wrong_preds[rec_id]
            if np.all([info["true_label"] == "Zucchini" for info in slices_info]):
                total = len(recording_wrong_preds[rec_id])
                print(f"Recording {rec_id}: {errors} / {total} errors")
                worst_recording_indices.append(rec_id)
                i += 1
            
            if i == 5:
                break

    output_dir = os.path.join("relevance_maps_combined", run_name, run_number, f"version_{fold_number}", folder_suffix)

    for rec_id in worst_recording_indices:
        slices_info = recording_wrong_preds[rec_id]
        plot_spectrogram(
            cfg,
            rec_id,
            dataset_name,
            slices_info,
            dataset,
            output_dir,
            label_colors,
            unique_label_names,
            best_model
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for audio classification using PyTorch Lightning.")
    parser.add_argument('--run-name', required=True, type=str)
    parser.add_argument('--run-number', required=True, type=str)
    parser.add_argument('--fold-number', required=True, type=int)
    parser.add_argument('--train-dataset-name1', required=True, type=str)
    parser.add_argument('--train-dataset-name2', required=True, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_name = args.run_name
    run_number = args.run_number
    fold_number = args.fold_number
    train_dataset_name1 = args.train_dataset_name1
    train_dataset_name2 = args.train_dataset_name2

    run_root = os.path.join('runs', run_name, 'cnnResnet18', 'cnnSpectrogram', run_number)
    hparams_path = os.path.join(run_root, f'version_{fold_number}', 'hparams.yaml')

    with open(hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)
        cfg = hparams['cfg']

    merged_train_dataset_name = merge_datasets(cfg, train_dataset_name1, train_dataset_name2)

    main(
        run_root=run_root,
        run_name=run_name, 
        run_number=run_number, 
        fold_number=fold_number, 
        dataset_name=merged_train_dataset_name, 
        cfg=cfg,
        folder_suffix='train_dataset'
    )
    main(
        run_root=run_root,
        run_name=run_name, 
        run_number=run_number, 
        fold_number=fold_number, 
        dataset_name=hparams['test_dataset_name'], 
        cfg=cfg,
        folder_suffix='test_dataset'
    )
