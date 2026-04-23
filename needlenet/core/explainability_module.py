import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from .lrp.lrp import LRPModel
from .utils.utils import closest_divisible_by_4, hertz_to_mel, mel_to_hertz


class ExplainabilityModule:
    def __init__(self, model, dataset, idx_to_label, cfg):
        self.model = model
        self.dataset = dataset
        self.idx_to_label = idx_to_label
        self.cfg = cfg

        self.device = next(self.model.parameters()).device
        self.target_sr = cfg['data']['target_sr']
        self.spec_type = cfg['data']['spectrogram_type']
        self.hop_length = cfg['data']['hop_length']
        self.min_freq = cfg['data']['min_freq']
        self.max_freq = cfg['data']['max_freq']
        self.top_k = cfg['relevance']['top_k']

        sample = self.dataset[0][2]
        self.freq_resized = closest_divisible_by_4(sample.shape[1])
        self.time_resized = closest_divisible_by_4(sample.shape[2])

        self.lrp_model = LRPModel(self.model, architecture='resnet', top_k=self.top_k)
        self.lrp_model.to(self.device)

        self.transform = transforms.Resize((self.freq_resized, self.time_resized), antialias=True)

    def generate_plotting_axes(self):
        if self.spec_type == "mel":
            n_mels = self.cfg['data']['n_mels']
            yticks_pixels = np.linspace(0, n_mels - 1, num=10)
            min_mel = hertz_to_mel(self.min_freq)
            max_mel = hertz_to_mel(self.max_freq)
            mel_values = min_mel + (yticks_pixels / (n_mels - 1)) * (max_mel - min_mel)
            ytickslabels_hz = mel_to_hertz(mel_values)

        elif self.spec_type == "linear":
            yticks_pixels = np.linspace(0, self.freq_resized - 1, num=5)
            ytickslabels_hz = self.min_freq + (yticks_pixels / (self.freq_resized - 1)) * (self.max_freq - self.min_freq)

        else:
            raise ValueError(f"Unknown spectrogram type: {self.spec_type}. Supported types: 'mel', 'linear'.")

        xticks_frames = np.linspace(0, self.time_resized - 1, num=6)
        xticklabels_sec = (xticks_frames * self.hop_length) / self.target_sr

        return xticks_frames, xticklabels_sec, yticks_pixels, ytickslabels_hz
    
    def generate_single_relevance_map(self, x):
        x_resized = self.transform(x)
        image_tensor = x_resized.unsqueeze(0).to(self.device)

        r = self.lrp_model.forward(image_tensor).detach().cpu().numpy()
        r = (r - r.min()) / (r.max() - r.min())

        return r

    def generate_relevance_maps(self):
        r_by_class = {idx: [] for idx in self.idx_to_label.keys()}
        max_freqs_by_class = {idx: [] for idx in self.idx_to_label.keys()}

        for data in tqdm(self.dataset, desc="Generating Relevance Maps"):
            _, _, x, y = data
            label = int(y)

            r = self.generate_single_relevance_map(x)

            r_by_class[label].append(r)
            max_freq_slice = int(np.argmax(np.sum(r, axis=1)))
            max_freqs_by_class[label].append(max_freq_slice)

        return r_by_class, max_freqs_by_class

    def generate_single_saliency_map(self, x, label):
        self.model.eval()

        x_resized = self.transform(x)
        image_tensor = x_resized.unsqueeze(0).to(self.device)
        image_tensor.requires_grad_(True)

        with torch.set_grad_enabled(True):
            logits = self.model(image_tensor)
            score = logits[0, label]
            self.model.zero_grad(set_to_none=True)
            if image_tensor.grad is not None:
                image_tensor.grad.zero_()
            score.backward()
            grad = image_tensor.grad.detach().cpu().numpy()[0]

        sal = np.abs(grad).sum(axis=0)
        sal = (sal - sal.min()) / (sal.max() - sal.min())

        return sal

    def generate_saliency_maps(self):
        saliency_by_class = {idx: [] for idx in self.idx_to_label.keys()}
        max_freqs_by_class = {idx: [] for idx in self.idx_to_label.keys()}

        for data in tqdm(self.dataset, desc="Generating Saliency Maps"):
            _, _, x, y = data
            label = int(y)

            sal = self.generate_single_saliency_map(x, label)

            saliency_by_class[label].append(sal)
            max_freq_slice = int(np.argmax(np.sum(sal, axis=1)))
            max_freqs_by_class[label].append(max_freq_slice)

        return saliency_by_class, max_freqs_by_class
