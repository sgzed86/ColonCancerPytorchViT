import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from torch.utils.data.distributed import DistributedSampler

class HyperKvasirDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        img_size: int = 224,
        transform: Optional[A.Compose] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        # Load labels from CSV
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'labeled-images', 'image-labels.csv'))
        
        # Define class mappings
        self.lesion_classes = {'anatomical-landmarks': 0, 'lesion': 1}
        self.polyp_classes = {
            'hyperplastic polyp': 0,
            'sessile serrated lesion': 1,
            'traditional serrated adenoma': 2,
            'tubular adenoma': 3,
            'unknown': 4
        }
        self.fibrosis_classes = {
            'none': 0,
            'mild': 1,
            'moderate': 2,
            'severe': 3
        }
        
        # Set up transforms
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(height=img_size, width=img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
        else:
            self.transform = transform
            
        # Load dataset
        self.data = self._load_dataset()
        
    def _load_dataset(self) -> list:
        """Load dataset metadata"""
        data = []
        images_dir = os.path.join(self.root_dir, 'labeled-images')
        
        for _, row in self.labels_df.iterrows():
            organ = row['Organ'].strip().lower()  # 'Lower GI' -> 'lower-gi'
            classification = row['Classification'].strip().lower()  # e.g., 'anatomical-landmarks'
            finding = row['Finding'].strip().lower()  # e.g., 'cecum'
            img_name = row['Video file'] + '.jpg'
            
            # Construct the image path based on the actual directory structure
            if organ == 'lower gi':
                img_path = os.path.join(images_dir, 'lower-gi-tract', classification, finding, img_name)
            else:  # upper gi
                img_path = os.path.join(images_dir, 'upper-gi-tract', classification, finding, img_name)
                
            if not os.path.exists(img_path):
                continue
                
            # Lesion label: 1 if not anatomical-landmarks, else 0
            lesion_label = 0 if classification == 'anatomical-landmarks' else 1
            
            # Polyp label: map from Finding if possible, else unknown
            polyp_label = self.polyp_classes.get(finding, 4)
            
            # Fibrosis label: not available, set to 0 (none)
            fibrosis_label = 0
            
            data.append({
                'image_path': img_path,
                'lesion_label': lesion_label,
                'polyp_label': polyp_label,
                'fibrosis_label': fibrosis_label
            })
        
        # Split into train/val
        np.random.seed(42)
        indices = np.random.permutation(len(data))
        split_idx = int(len(data) * 0.8)  # 80% train, 20% val
        
        if self.split == 'train':
            self.data = [data[i] for i in indices[:split_idx]]
        else:
            self.data = [data[i] for i in indices[split_idx:]]
            
        return self.data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Load and transform image
        image = Image.open(item['image_path']).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'lesion_label': torch.tensor(item['lesion_label'], dtype=torch.float32),
            'polyp_label': torch.tensor(item['polyp_label'], dtype=torch.long),
            'fibrosis_label': torch.tensor(item['fibrosis_label'], dtype=torch.long)
        }

def get_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    pin_memory: bool = True,
    prefetch_factor: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with optimized settings"""
    
    train_dataset = HyperKvasirDataset(
        root_dir=root_dir,
        split='train',
        img_size=img_size
    )
    
    val_dataset = HyperKvasirDataset(
        root_dir=root_dir,
        split='val',
        img_size=img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True  # Drop last incomplete batch for better performance
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader 