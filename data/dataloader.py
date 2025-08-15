import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import pandas as pd
from PIL import Image
import warnings

from data.dataset import FaceDataset
from data.dataset import HFImageCollator

from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    get_linear_schedule_with_warmup,
)

class Dataloader():
    def __init__(self, ai_dir, real_dir, split, seed, model_id, num_workers=8, batch_size=128):
        self.model_id = str(model_id)
        self.ai_dir = ai_dir
        self.real_dir = real_dir
        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers
        self.seed = seed
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

    def create_loader(self):

        ai_ds   = FaceDataset(root_dir=self.ai_dir,   label=1, processor=self.processor)
        real_ds = FaceDataset(root_dir=self.real_dir, label=0, processor=self.processor)

        full_ds = ai_ds + real_ds

        total = len(full_ds)
        train_size = int(total * self.split[0])
        val_size = int(total * self.split[1])
        test_size = int(total * self.split[2])

        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [train_size, val_size, test_size], generator = torch.Generator().manual_seed(self.seed)
            )
        
        collate = HFImageCollator(self.processor)
        
        train_loader = DataLoader(train_ds, self.batch_size, shuffle=True, num_workers = self.num_workers, pin_memory=True, collate_fn=collate)

        val_loader = DataLoader(val_ds, self.batch_size, shuffle=True, num_workers = self.num_workers, pin_memory=True, collate_fn=collate)

        test_loader = DataLoader(test_ds, self.batch_size, shuffle=True, num_workers = self.num_workers, pin_memory=True, collate_fn=collate)

        return train_loader, val_loader, test_loader