import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import pandas as pd
from PIL import Image
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

class FaceDataset(Dataset):
    def __init__(self, root_dir, label, processor):
        self.root_dir = str(root_dir)
        self.paths = [str(p) for p in Path(self.root_dir).rglob("*") if p.is_file()]
        #self.tranform = transform
        self.label = label
        self.processor = processor

    def __getitem__(self, idx):
        
        image = Image.open(self.paths[idx]).convert("RGB")
        #if self.transform:
            #image = self.transform(image)

        return (image, self.label)

    def __len__(self):
        return len(self.paths)
    
    def collate(self, batch):
        imgs, labels = zip(*batch) 
        enc = self.processor(images=list(imgs), return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
class HFImageCollator:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        imgs, labels = zip(*batch)
        enc = self.processor(images=list(imgs), return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"],                
            "labels": torch.tensor(labels, dtype=torch.long),    
        }
