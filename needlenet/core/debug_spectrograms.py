import argparse
import yaml
import os
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from .utils.import_task import load_dataset_builder
from .utils.utils import hertz_to_mel, mel_to_hertz


def pick_indices_for_label(df, label_name):
    hits = df.index[df['label'] == label_name].tolist()
    if len(hits) == 0:
        raise ValueError(f"Label '{label_name}' not found in the dataset.")
    return hits

def save_spectrograms(zucchini_spectrogram3ch, chicken_spectrogram3ch, save_path, cfg, chicken_name, zucchini_name):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    n_ffts = cfg["data"]["n_ffts"]
    sr = cfg["data"]["target_sr"]
    hop_length = cfg["data"]["hop_length"]
    n_mels = cfg["data"]["n_mels"]
    spec_type = cfg["data"]['spectrogram_type']

    def _denorm(x):
        max_db = cfg['data']['max_db']
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
        y = x * std + mean
        y *= max_db
        y -= max_db
        return y
    
    zucchini_spectrogram3ch = _denorm(zucchini_spectrogram3ch)
    chicken_spectrogram3ch = _denorm(chicken_spectrogram3ch)

    for ch in range(zucchini_spectrogram3ch.shape[0]):
        ch_number = ch + 1

        min_freq = cfg["data"]["min_freq"]
        max_freq = cfg["data"]["max_freq"]
        n_fft = n_ffts[ch]
        print(f'Channel {ch_number}: n_fft={n_fft}, freq_bins={zucchini_spectrogram3ch[ch].shape[0]}')

        # Frequency axis calculation for mel
        if spec_type == "mel":
            yticks_pixels = np.linspace(0, n_mels - 1, num=10)
            min_mel = hertz_to_mel(min_freq)
            max_mel = hertz_to_mel(max_freq)
            mel_values = min_mel + (yticks_pixels / (n_mels - 1)) * (max_mel - min_mel)
            ytickslabels_hz = mel_to_hertz(mel_values)

        # Frequency axis calculation for linear
        elif spec_type == "linear":
            plotted_freq_bins = zucchini_spectrogram3ch[ch].shape[0]

            yticks_real = np.linspace(0, plotted_freq_bins - 1, num=5)
            ytickslabels_hz = min_freq + (yticks_real / (plotted_freq_bins - 1)) * (max_freq - min_freq)
            yticks_pixels = yticks_real

        else:
            raise ValueError(f"Unknown spectrogram type: {spec_type}. Supported types: 'mel', 'linear'.") 

        # Time axis calculation
        time_frames = zucchini_spectrogram3ch[ch].shape[1]
        xticks_frames = np.linspace(0, time_frames - 1, num=6)
        xticklabels_sec = (xticks_frames * hop_length) / sr

        axL = axes[ch, 0]
        axL.set_title(f"Zucchini - Channel {ch_number} (n_fft={n_fft})")
        imgL = zucchini_spectrogram3ch[ch].detach().cpu().numpy()
        imL = axL.imshow(imgL, aspect="auto", origin="lower", cmap="turbo")
        fig.colorbar(imL, ax=axL, fraction=0.046, pad=0.04)

        axL.set_yticks(yticks_pixels)
        axL.set_yticklabels([f"{hz:.0f}" for hz in ytickslabels_hz])
        axL.set_ylabel("Frequency (Hz)")

        axL.set_xticks(xticks_frames)
        axL.set_xticklabels([f"{t:.2f}" for t in xticklabels_sec])
        axL.set_xlabel("Time (s)")

        axR = axes[ch, 1]
        axR.set_title(f"Chicken - Channel {ch_number} (n_fft={n_fft})")
        imgR = chicken_spectrogram3ch[ch].detach().cpu().numpy()
        imR = axR.imshow(imgR, aspect="auto", origin="lower", cmap="turbo")
        fig.colorbar(imR, ax=axR, fraction=0.046, pad=0.04)

        axR.set_yticks(yticks_pixels)
        axR.set_yticklabels([f"{hz:.0f}" for hz in ytickslabels_hz])
        axR.set_ylabel("Frequency (Hz)")

        axR.set_xticks(xticks_frames)
        axR.set_xticklabels([f"{t:.2f}" for t in xticklabels_sec])
        axR.set_xlabel("Time (s)")

    plt.suptitle(f"{chicken_name} vs {zucchini_name}", wrap=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(config_file_path, dataset_name, shaft_dataset_name, output_dir, dataset_type):

    with open(config_file_path, 'r') as f:
        cfg = yaml.safe_load(f)

    seed = cfg['seed']
    pl.seed_everything(seed, workers=True)

    dataset_builder = load_dataset_builder(cfg['data']['name'])

    if dataset_type == "train":
        dataset = dataset_builder.build_dataset(cfg, augment=cfg['data']['augment_train'], type="train", dataset_name=dataset_name, dataset_shaft_augment_name=shaft_dataset_name)
    elif dataset_type == "valid":
        dataset = dataset_builder.build_dataset(cfg, augment=cfg['data']['augment_valid'], type="valid", dataset_name=dataset_name)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Supported types: 'train', 'valid'.")

    if cfg["data"]["normalizing_method"] == "global":
        dataset.update_normalization_params(indices=list(range(len(dataset))))
    
    chicken_indices = pick_indices_for_label(dataset.df, 'Chicken')
    zucchini_indices = pick_indices_for_label(dataset.df, 'Zucchini')
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(5):
        chicken_name, _, chicken_spec3ch, _ = dataset[chicken_indices[idx]]
        zucchini_name, _, zucchini_spec3ch, _ = dataset[zucchini_indices[idx]]

        save_path = os.path.join(output_dir, f'Sample_{idx+1}.png')

        save_spectrograms(zucchini_spec3ch, chicken_spec3ch, save_path, cfg, chicken_name, zucchini_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug processed signal to spectrograms.")
    parser.add_argument('--config', required=False, type=str,
                        help='Path to the configuration file')
    parser.add_argument('--dataset-name', required=True, type=str,
                        help='Name of the dataset')
    parser.add_argument('--shaft-dataset-name', required=True, type=str,
                        help='Name of the shaft dataset')
    parser.add_argument('--dataset-type', required=True, type=str,
                        choices=['train', 'valid'],
                        help='Type of the dataset')
    parser.add_argument('--output-dir', required=True, type=str, help='Directory to save the debug spectrograms')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(config_file_path=args.config, dataset_name=args.dataset_name, shaft_dataset_name=args.shaft_dataset_name, output_dir=args.output_dir, dataset_type=args.dataset_type)
