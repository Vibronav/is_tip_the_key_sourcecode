import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random
from .utils.seeds import DATA_MODULE_SEED

class AudioDataModule(pl.LightningDataModule):

    def __init__(self, cfg, dataset_builder, train_dataset_name, train_shaft_dataset_name, valid_dataset_name=None):
        super().__init__()
        self.cfg = cfg
        self.dataset_builder = dataset_builder
        self.train_dataset_name = train_dataset_name
        self.train_shaft_dataset_name = train_shaft_dataset_name
        self.valid_dataset_name = valid_dataset_name
        self.batch_size = cfg['training']['batch_size']
        self.num_workers = cfg['training']['num_workers']
        self.seed = cfg['seed']
        self.normalizing_method = cfg['data']['normalizing_method']

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pin_memory = (self.device == "cuda")
        self.persistent_workers = self.num_workers > 0
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed + DATA_MODULE_SEED)

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is not None and self.val_dataset is not None:
            return
        
        self.train_dataset = self.dataset_builder.build_dataset(
            self.cfg, 
            augment=self.cfg['data']['augment_train'], 
            type="train", 
            dataset_name=self.train_dataset_name, 
            dataset_shaft_augment_name=self.train_shaft_dataset_name
        )
        self.val_dataset = self.dataset_builder.build_dataset(
            self.cfg, 
            augment=self.cfg['data']['augment_valid'], 
            type="valid", 
            dataset_name=self.valid_dataset_name if self.valid_dataset_name else self.train_dataset_name
        )

        if self.normalizing_method == "global":
            self.train_dataset.update_normalization_params(np.arange(len(self.train_dataset)))
            self.val_dataset.update_normalization_params(np.arange(len(self.val_dataset)))

        classes = self.train_dataset.label_numbers
        classes_unique = np.unique(classes)
        cw = compute_class_weight(class_weight="balanced", classes=classes_unique, y=classes)
        self.class_weights = torch.tensor(cw, dtype=torch.float)

        print(f"Train Class idx: {self.train_dataset.label_to_idx}")
        print(f"First class numbers: {self.train_dataset.label_numbers[:10]}")
        print(f"Valid Class idx: {self.val_dataset.label_to_idx}")
        print(f"First class numbers: {self.val_dataset.label_numbers[:10]}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            generator=self.generator,
            worker_init_fn=self._seed_worker,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=self._seed_worker
        )
    
    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)