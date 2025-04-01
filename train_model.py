import os
import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt

# Import custom modules
import data_loader
from trainer import AutismModelTrainer, evaluate_model

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data = data_loader.get_data_loaders(
        data_root='data', 
        batch_size=32, 
        num_workers=4
    )
    
    train_loader = data['train_loader']
    valid_loader = data['valid_loader']
    test_loader = data['test_loader']
    
    # Print dataset information
    print(f"Number of training samples: {len(data['train_dataset'])}")
    print(f"Number of validation samples: {len(data['valid_dataset'])}")
    print(f"Number of test samples: {len(data['test_dataset'])}")
    
    # Create model - ResNet50 for binary classification
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers (optional)
    # This is transfer learning - we keep the pretrained weights for feature extraction
    # and only train the final layers
    for param in list(model.parameters())[:-20]:  # Freeze all but the last few layers
        param.requires_grad = False
    
    # Modify the final fully connected layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Add dropout with 50% probability
        nn.Linear(num_features, 1)  # 1 output neuron for binary classification
    )
    
    # Create trainer
    trainer = AutismModelTrainer(
        model=model,
        device=device,
        learning_rate=1e-4,
        weight_decay=5e-4
    )
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs=20,
        checkpoint_dir='checkpoints'
    )
    
    # Plot confusion matrix for validation set
    trainer.plot_confusion_matrix(valid_loader)
    
    # Load the best model by F1 score
    try:
        trainer.load_best_model('checkpoints/best_model_f1.pt')
    except FileNotFoundError:
        print("Best model checkpoint not found, using current model.")
    
    # Evaluate on test set
    test_metrics = evaluate_model(
        model=trainer.model,
        test_loader=test_loader,
        device=device
    )
    
    print("Training and evaluation complete!")
    
if __name__ == "__main__":
    main() 