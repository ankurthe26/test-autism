import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from typing import List, Tuple
import logging
from models.efficientnet_model import EfficientNetAutismNet
from datasets.efficient_dataset import EfficientNetDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def plot_training_curves(history: dict, save_path: str = 'training_curves.png'):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_samples(
    dataset: EfficientNetDataset,
    num_samples: int = 5,
    save_dir: str = 'visualizations'
):
    """
    Visualize sample images with landmarks.
    
    Args:
        dataset: Dataset to visualize samples from
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        # Get sample
        image, landmarks, label = dataset[idx]
        
        # Convert image to numpy array
        image = image.permute(1, 2, 0).numpy()
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        image = image.astype(np.uint8)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # Plot landmarks
        landmarks = landmarks.numpy()
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=1)
        
        # Add title
        plt.title(f'Label: {label.item()}')
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'sample_{idx}.png'))
        plt.close()

def visualize_model_predictions(
    model: EfficientNetAutismNet,
    dataset: EfficientNetDataset,
    device: torch.device,
    num_samples: int = 5,
    save_dir: str = 'predictions'
):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained model
        dataset: Dataset to make predictions on
        device: Device to run model on
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to eval mode
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        # Get sample
        image, landmarks, label = dataset[idx]
        
        # Add batch dimension
        image = image.unsqueeze(0).to(device)
        landmarks = landmarks.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image, landmarks)
            prediction = outputs['predictions'].argmax(dim=1).item()
        
        # Convert image to numpy array
        image = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        image = image.astype(np.uint8)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # Plot landmarks
        landmarks = landmarks.squeeze(0).cpu().numpy()
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=1)
        
        # Add title
        plt.title(f'True Label: {label.item()}, Predicted: {prediction}')
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'prediction_{idx}.png'))
        plt.close()

def main():
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = EfficientNetAutismNet(
        num_classes=2,
        dropout=0.1,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataset
    dataset = EfficientNetDataset(
        data_root='data',
        split='valid',
        transform=False
    )
    
    # Visualize samples
    visualize_samples(dataset, num_samples=5)
    
    # Visualize predictions
    visualize_model_predictions(model, dataset, device, num_samples=5)

if __name__ == '__main__':
    main() 