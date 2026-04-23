import pandas as pd
import os
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torchaudio import functional
from torchaudio_augmentations import Gain, PitchShift
from torchvision.transforms.functional import crop
from random import choice
from scipy.signal import butter, sosfiltfilt
import numpy as np
import matplotlib.pyplot as plt

class AudioDataset(Dataset):

    def __init__(
            self, 
            cfg,
            manifest_path,
            data_root,
            augment,
            dataset_type,
            manifest_shaft_augment_path=None
        ):
        
        super().__init__()
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f'Manifest not found: {manifest_path}')
        self.df = pd.read_csv(manifest_path)

        required_columns = {'path', 'label', 'recording_id'}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(f'Manifest file is missing required columns: {missing}')

        # Shaft augment
        self.df_shaft_augment = None
        self.shaft_by_label = {}
        if manifest_shaft_augment_path is not None:
            if not os.path.exists(manifest_shaft_augment_path):
                raise FileNotFoundError(f'Manifest shaft augment not found: {manifest_shaft_augment_path}')
            self.df_shaft_augment = pd.read_csv(manifest_shaft_augment_path)
            for lbl, group in self.df_shaft_augment.groupby('label'):
                self.shaft_by_label[lbl] = group.reset_index(drop=True)
        
        self.data_path = os.path.join(data_root, cfg['data']['name'])

        self.dataset_type = dataset_type
        self.augment = augment
        self.labels_to_augment = cfg['data'].get('labels_to_augment', [])
        self.target_sr = cfg['data']['target_sr']
        self.n_ffts = cfg['data']['n_ffts']
        self.hop_length = cfg['data']['hop_length']
        self.n_mels = cfg['data']['n_mels']
        self.max_db = cfg['data']['max_db']
        self.spec_type = cfg['data']['spectrogram_type']
        self.min_freq = cfg['data']['min_freq']
        self.max_freq = cfg['data']['max_freq']
        self.normalizing_method = cfg['data']['normalizing_method']
        self.power = cfg['data']['power']
        self.spec_normalized = cfg['data']['spectrogram_normalized']

        # Attenuation
        self.attenuation_level = cfg['data']['attenuation_level']
        self.attenuation_curves = {}
        attenuation_data_path = os.path.join(self.data_path, 'attenuation_data.json')
        if os.path.exists(attenuation_data_path):
            with open(attenuation_data_path, 'r') as f:
                raw = json.load(f)
            for tissue_name, freq_dict in raw.items():
                freqs = np.array([float(k) for k in freq_dict.keys()])
                attens = np.array(list(freq_dict.values()))
                self.attenuation_curves[tissue_name.lower()] = (freqs, attens)
        else:
            print("No attenuation data found.")

        labels_sorted = sorted(self.df['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(labels_sorted)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.spec_spectrograms = []
        for i in range(3):
            n_fft = self.n_ffts[i]
            if self.spec_type == "mel":
                self.spec_spectrograms.append(
                    T.MelSpectrogram(
                        sample_rate=self.target_sr,
                        n_fft=n_fft,
                        win_length=n_fft,
                        hop_length=self.hop_length,
                        n_mels=self.n_mels,
                        window_fn=torch.hann_window,
                        power=self.power,
                        normalized=self.spec_normalized,
                        f_min=self.min_freq,
                        f_max=self.max_freq
                    )
                )
            elif self.spec_type == "linear":
                self.spec_spectrograms.append(
                    T.Spectrogram(
                        n_fft=n_fft,
                        win_length=n_fft,
                        hop_length=self.hop_length,
                        window_fn=torch.hann_window,
                        power=self.power,
                        normalized=self.spec_normalized,
                    )
                )
            else:
                raise ValueError(f"Unknown spectrogram type: {self.spec_type}. Supported types: 'mel', 'linear'.")
            
        self.to_db = T.AmplitudeToDB(top_db=self.max_db)

        self.global_mean = None
        self.global_std = None

    @property
    def recording_ids(self):
        return self.df["recording_id"].to_numpy()
    
    @property
    def unique_label_names(self):
        return [self.idx_to_label[i] for i in range(len(self.idx_to_label))]
    
    @property
    def label_numbers(self):
        return self.df['label'].map(self.label_to_idx).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['path']
        label = row['label']
        label_number = self.label_to_idx[label]

        signal, sr = self._load_mono_signal(audio_path)

        if sr != self.target_sr:
            signal = functional.resample(signal, sr, self.target_sr)

        if self.dataset_type == "train":
            if self.augment and self.df_shaft_augment is not None:
                signal = self._mix_shaft_signal(signal, label)
        elif self.dataset_type == "valid":
            if self.augment:
                signal = self._attenuate_signal(signal, label)

        signal =  self.bandpass(signal)

        signal = self._normalize_signal(signal)

        specs = self._convert_to_spectrograms(signal)

        max_F_length = specs[2].shape[0]

        specs = [self._resize(spec, max_F_length) for spec in specs]

        spec_3ch = torch.stack(specs, dim=0)
        spec_3ch = self.normalize_img_with_3_spectrograms(spec_3ch)

        y = torch.tensor(label_number, dtype=torch.long)
        return os.path.basename(audio_path), row['recording_id'], spec_3ch, y
    
    def _load_mono_signal(self, path):
        full_audio_path = os.path.join(self.data_path, path)
        signal, sr = torchaudio.load(full_audio_path)

        return signal, sr
    
    def bandpass(self, signal):
        nyq = 0.5 * self.target_sr

        if self.max_freq > self.target_sr // 2:
            raise ValueError(f"max_freq must be less than or equal to Nyquist frequency (target_sr/2). Got max_freq={self.max_freq}, target_sr={self.target_sr}")

        if self.min_freq == 0 and self.max_freq == self.target_sr // 2:
            return signal

        if self.min_freq != 0 and self.max_freq != self.target_sr // 2:
            lo = max(self.min_freq / nyq, 1e-6)
            hi = min(self.max_freq / nyq, 0.999999)
            sos = butter(32, [lo, hi], btype='bandpass', output='sos')

        if self.min_freq == 0:
            Wn = self.max_freq / nyq
            sos = butter(32, Wn, btype='lowpass', output='sos')

        if self.max_freq == self.target_sr // 2:
            Wn = max(self.min_freq / nyq, 1e-6)
            sos = butter(32, Wn, btype='highpass', output='sos')

        x = signal.detach().cpu().squeeze(0).numpy().astype(np.float64, copy=False)
        y = sosfiltfilt(sos, x)
        y = np.ascontiguousarray(y)
        processed_signal = torch.from_numpy(y).to(device=signal.device, dtype=signal.dtype).unsqueeze(0)
        return processed_signal

    def _normalize_signal(self, signal):
        if self.normalizing_method == "global":
            ## Use global normalization
            if self.global_mean is None or self.global_std is None:
                raise ValueError("Global mean and std not set. Call 'update_normalization_params' with appropriate indices before using the dataset.")

            signal = (signal - self.global_mean) / (self.global_std + 1e-8)

        if self.normalizing_method == "local":
            ## Use local normalization
            mean = signal.mean()
            std = signal.std()
            signal = (signal - mean) / (std + 1e-8)

        if self.normalizing_method == "load":
            # Use load() normalization
            pass

        return signal

    def normalize_img_with_3_spectrograms(self, spec_3ch):
        spec_3ch = (spec_3ch + self.max_db) / self.max_db

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) # It will be broadcasted to (3, F, T)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)  # It will be broadcasted to (3, F, T)
        spec_3ch = (spec_3ch - mean) / std

        return spec_3ch

    def _convert_to_spectrograms(self, signal):
        specs = []
        for spec_transform in self.spec_spectrograms:
            spec = spec_transform(signal)

            if self.spec_type == "linear":
                nyq = 0.5 * self.target_sr
                max_bin = int(self.max_freq / nyq * (spec.shape[1] - 1))
                spec = spec[:, :max_bin + 1, :]

            spec = self.to_db(spec)

            max_db = spec.amax(dim=(-2, -1), keepdim=True)
            spec = spec - max_db
            spec = spec.clamp(min=-self.max_db)

            spec = spec.squeeze(0)
            specs.append(spec)

        return specs

    def _resize(self, spec, max_F_length):
        F_length = spec.shape[0]
        if F_length < max_F_length:
            spec = spec.unsqueeze(0).unsqueeze(0)
            spec = F.interpolate(spec, size=(max_F_length, spec.shape[-1]), mode='nearest')
            spec = spec.squeeze(0).squeeze(0)
            return spec
        else:
            return spec

    def _mix_shaft_signal(self, signal, tip_label):
        if tip_label == "Chicken":
            shaft_df = self.shaft_by_label["Zucchini_shaft"]
        elif tip_label == "Zucchini":
            shaft_df = self.shaft_by_label["Chicken_shaft"]
        else:
            return signal

        shaft_row = shaft_df.sample(n=1).iloc[0]
        shaft_signal, shaft_sr = self._load_mono_signal(shaft_row['path'])

        if shaft_sr != self.target_sr:
            shaft_signal = functional.resample(shaft_signal, shaft_sr, self.target_sr)

        return signal + shaft_signal

    def _attenuate_signal(self, signal, label):
        tissue_key = label.lower()
        if tissue_key not in self.attenuation_curves:
            raise ValueError(f"Attenuation curve for tissue {tissue_key} not found.")

        freqs_curve, attens_db = self.attenuation_curves[tissue_key]

        x = signal.squeeze(0)
        n_samples = x.shape[-1]
        X = torch.fft.rfft(x)
        fft_freqs = torch.fft.rfftfreq(n_samples, d=1.0 / self.target_sr).numpy()

        atten_interp = np.interp(fft_freqs, freqs_curve, attens_db)
        atten_scaled = atten_interp * self.attenuation_level

        gain = torch.tensor(10 ** (atten_scaled / 20.0), dtype=x.dtype, device=x.device)
        X_attenuated = X * gain
        x_attenuated = torch.fft.irfft(X_attenuated, n=n_samples)

        return x_attenuated.unsqueeze(0)

    def update_normalization_params(self, indices):
        subset = self.df.iloc[indices]

        all_means = []
        all_stds = []
        for _, row in subset.iterrows():
            audio_path = row['path']
            signal, sr = self._load_mono_signal(audio_path)

            if sr != self.target_sr:
                signal = functional.resample(signal, sr, self.target_sr)

            all_means.append(signal.mean().item())
            all_stds.append(signal.std().item())

        global_mean = float(torch.tensor(all_means).mean())
        global_std = float(torch.tensor(all_stds).mean())

        self.global_mean = global_mean
        self.global_std = global_std


def build_dataset(cfg, augment, type="train", dataset_name=None, dataset_shaft_augment_name=None):
    data_root = cfg['data']['root']
    if type == "train":
        manifest_path = os.path.join(data_root, f'{cfg["data"]["name"]}/{dataset_name}', 'manifest.csv')
        manifest_shaft_augment_path = os.path.join(data_root, f'{cfg["data"]["name"]}/{dataset_shaft_augment_name}', 'manifest.csv')
        return AudioDataset(cfg, manifest_path, data_root, augment, dataset_type=type, manifest_shaft_augment_path=manifest_shaft_augment_path)
    elif type == "valid":
        manifest_path = os.path.join(data_root, f'{cfg["data"]["name"]}/{dataset_name}', 'manifest.csv')
        return AudioDataset(cfg, manifest_path, data_root, augment, dataset_type=type)
    elif type == "test":
        if dataset_name is not None:
            test_manifest_path = os.path.join(data_root, f'{cfg["data"]["name"]}/{dataset_name}', 'manifest.csv')
            return AudioDataset(cfg, test_manifest_path, data_root, augment, dataset_type=type)
        else:
            return None
    else:
        raise ValueError(f"Unknown dataset type: {type}")
