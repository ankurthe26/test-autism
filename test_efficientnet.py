import torch
import numpy as np
import logging
from tqdm import tqdm
from models.efficientnet_model import EfficientNetAutismNet
from datasets.efficient_dataset import EfficientNetDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on the provided dataloader.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, landmarks, labels in tqdm(dataloader, desc="Evaluating"):
            # Move inputs to device
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images, landmarks)
            predictions = outputs['predictions'].argmax(dim=1)
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions) * 100,
        'precision': precision_score(all_labels, all_predictions, average='weighted') * 100,
        'recall': recall_score(all_labels, all_predictions, average='weighted') * 100,
        'f1': f1_score(all_labels, all_predictions, average='weighted') * 100,
        'confusion_matrix': confusion_matrix(all_labels, all_predictions)
    }
    
    return metrics

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
    checkpoint_path = 'checkpoints/best_model.pth'
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create test dataset and dataloader
    test_dataset = EfficientNetDataset(
        data_root='data',
        split='test',
        transform=False  # No augmentation for test data
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    logger.info("===== Test Results =====")
    logger.info(f"Accuracy: {metrics['accuracy']:.2f}%")
    logger.info(f"Precision: {metrics['precision']:.2f}%")
    logger.info(f"Recall: {metrics['recall']:.2f}%")
    logger.info(f"F1 Score: {metrics['f1']:.2f}%")
    logger.info("Confusion Matrix:")
    logger.info(metrics['confusion_matrix'])
    
    # For binary classification, print class-wise results
    if model.num_classes == 2:
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        logger.info(f"True Negatives: {tn}")
        logger.info(f"False Positives: {fp}")
        logger.info(f"False Negatives: {fn}")
        logger.info(f"True Positives: {tp}")
        
        # Calculate specificity
        specificity = tn / (tn + fp) * 100
        logger.info(f"Specificity: {specificity:.2f}%")
    
if __name__ == '__main__':
    main()