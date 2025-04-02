import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from typing import Optional, Dict, Tuple, Any, List
import logging
from albumentations.pytorch import ToTensorV2
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientNetDataset(Dataset):
    """Dataset class for EfficientNet-based autism detection model."""
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: bool = True,
        image_size: tuple = (224, 224)
    ):
        """
        Initialize the dataset.
        
        Args:
            data_root: Root directory containing the data
            split: Dataset split ('train', 'val', or 'test')
            transform: Whether to apply data augmentation
            image_size: Size to resize images to
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Get all image paths and labels
        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        
        # Process Autistic class
        autistic_dir = self.data_root / split / 'Autistic'
        if autistic_dir.exists():
            for img_path in autistic_dir.glob('*.jpg'):
                self.image_paths.append(img_path)
                self.labels.append(1)  # 1 for Autistic
        
        # Process Non_Autistic class
        non_autistic_dir = self.data_root / split / 'Non_Autistic'
        if non_autistic_dir.exists():
            for img_path in non_autistic_dir.glob('*.jpg'):
                self.image_paths.append(img_path)
                self.labels.append(0)  # 0 for Non_Autistic
        
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.data_root / split}")
        
        logger.info(f"Found {len(self.image_paths)} images in {split} split")
        logger.info(f"Class distribution: Autistic={sum(self.labels)}, Non_Autistic={len(self.labels)-sum(self.labels)}")
        
        # Set up transforms
        self.train_transforms = self._get_train_transforms()
        self.val_transforms = self._get_val_transforms()
        
        # Calculate class weights for weighted sampling
        self.class_weights = self._calculate_class_weights()
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple:
        try:
            # Get image path and label
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # Check if file exists
            if not image_path.exists():
                logger.warning(f"Missing image for sample {idx}: {image_path}")
                return self._get_default_sample()
            
            # Load image
            try:
                image = Image.open(image_path).convert('RGB')
                image = np.array(image)
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {str(e)}")
                return self._get_default_sample()
            
            # Apply transforms
            if self.transform:
                transformed = self.train_transforms(image=image)
            else:
                transformed = self.val_transforms(image=image)
            
            image = transformed['image']
            
            # Create dummy landmarks (since we don't have them)
            landmarks = np.zeros((478, 3), dtype=np.float32)
            landmarks = torch.from_numpy(landmarks).float()
            landmarks = landmarks / self.image_size[0]  # Normalize to [0, 1] range
            
            return image, landmarks, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            return self._get_default_sample()
    
    def _get_train_transforms(self):
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _get_val_transforms(self):
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def _get_default_sample(self) -> tuple:
        """Return a default sample when image loading fails."""
        # Create blank image
        image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        image = self.val_transforms(image=image)['image']
        
        # Create default landmarks
        landmarks = np.zeros((478, 3), dtype=np.float32)
        landmarks = torch.from_numpy(landmarks).float()
        landmarks = landmarks / self.image_size[0]  # Normalize to [0, 1] range
        
        # Create default label
        label = torch.tensor(0, dtype=torch.long)
        
        return image, landmarks, label
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for weighted sampling."""
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for weighted sampling."""
        return torch.tensor([self.class_weights[label] for label in self.labels]) 