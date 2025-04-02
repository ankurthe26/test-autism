import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ssl
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings
from typing import Dict, Tuple, List, Optional
import traceback
import yaml
import random
import matplotlib.pyplot as plt
from datetime import datetime

# Temporarily bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

from models.efficientnet_model import EfficientNetAutismNet
from datasets.efficient_dataset import EfficientNetDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """Collate function for DataLoader."""
    images, landmarks, labels = zip(*batch)
    images = torch.stack(images)
    landmarks = torch.stack(landmarks)
    labels = torch.stack(labels)
    return images, landmarks, labels

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 20,
        save_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up tensorboard
        self.writer = SummaryWriter(f"runs/autism_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, landmarks, labels) in enumerate(self.train_loader):
            try:
                # Move data to device
                images = images.to(self.device, dtype=torch.float32)
                landmarks = landmarks.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images, landmarks)
                loss = self.criterion(outputs['predictions'], labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = outputs['predictions'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                total_loss += loss.item()
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, landmarks, labels in self.val_loader:
                try:
                    # Move data to device
                    images = images.to(self.device, dtype=torch.float32)
                    landmarks = landmarks.to(self.device, dtype=torch.float32)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images, landmarks)
                    loss = self.criterion(outputs['predictions'], labels)
                    
                    # Calculate accuracy
                    _, predicted = outputs['predictions'].max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    total_loss += loss.item()
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {str(e)}")
                    continue
        
        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                }, os.path.join(self.save_dir, 'best_model.pth'))
            
            # Plot training curves
            self.plot_training_curves()
    
    def plot_training_curves(self):
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()

def main():
    """Main training function."""
    try:
        # Load configuration
        config = load_config('config/model_config.yaml')
        
        # Set device
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Create datasets
        train_dataset = EfficientNetDataset(
            data_root='data',
            split='train',
            transform=True
        )
        
        val_dataset = EfficientNetDataset(
            data_root='data',
            split='valid',
            transform=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        # Create model
        model = EfficientNetAutismNet(
            num_classes=2,
            dropout=0.1,
            device=device
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=config['training']['learning_rate'],
            weight_decay=0.01,
            num_epochs=20
        )
        
        # Train model
        trainer.train()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 