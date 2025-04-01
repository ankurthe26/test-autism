import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MockDataset(Dataset):
    """
    Mock dataset for testing
    """
    def __init__(self, size=10, transform=None):
        self.size = size
        self.transform = transform
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create a random image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        
        # Apply transform if provided
        if self.transform:
            img = self.transform(img)
        
        # Random label (0 or 1)
        label = torch.tensor(np.random.randint(0, 2), dtype=torch.float32)
        
        return img, label

def get_data_loaders(data_root='data', batch_size=32, num_workers=4):
    """
    Get data loaders for training, validation, and testing
    
    Args:
        data_root: Path to data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        dict: Dictionary containing data loaders
    """
    try:
        # Check if data directory exists
        if not os.path.exists(data_root):
            logger.warning(f"Data directory {data_root} not found. Using mock data.")
            return get_mock_data_loaders(batch_size, num_workers)
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create mock datasets for testing
        train_dataset = MockDataset(size=100, transform=transform)
        valid_dataset = MockDataset(size=20, transform=transform)
        test_dataset = MockDataset(size=20, transform=transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        return {
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'test_loader': test_loader,
            'train_dataset': train_dataset,
            'valid_dataset': valid_dataset,
            'test_dataset': test_dataset
        }
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return get_mock_data_loaders(batch_size, num_workers)

def get_mock_data_loaders(batch_size=32, num_workers=4):
    """
    Get mock data loaders for testing
    
    Args:
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        dict: Dictionary containing mock data loaders
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create mock datasets
    train_dataset = MockDataset(size=100, transform=transform)
    valid_dataset = MockDataset(size=20, transform=transform)
    test_dataset = MockDataset(size=20, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'valid_dataset': valid_dataset,
        'test_dataset': test_dataset
    }

def show_samples(dataloader, num_samples=5):
    """
    Visualize samples from a data loader
    
    Args:
        dataloader: PyTorch DataLoader object
        num_samples (int): Number of samples to display
    """
    # Get a batch
    images, labels = next(iter(dataloader))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Convert images back for visualization
    images_np = [img.permute(1, 2, 0).numpy() for img in images]
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images_np = [(img * std + mean).clip(0, 1) for img in images_np]
    
    # Create the figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    class_names = ['Autistic', 'Non_Autistic']
    
    for i, (img, label) in enumerate(zip(images_np, labels)):
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(class_names[label.item()])
        ax.axis('off')
    
    plt.tight_layout()
    plt.show() 