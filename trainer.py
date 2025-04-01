import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import time
from tqdm import tqdm

class AutismModelTrainer:
    def __init__(self, model, device, learning_rate=1e-4, weight_decay=1e-5):
        """
        Initialize the trainer with model and training parameters
        
        Args:
            model: PyTorch model
            device: Device to train on ('cpu', 'cuda', or 'mps')
            learning_rate: Initial learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        
        # Binary classification with 1 output neuron
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        # History for tracking metrics
        self.history = {
            'train_loss': [], 
            'val_loss': [],
            'train_acc': [], 
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.epochs_no_improve = 0
        self.early_stop_patience = 10
        
    def train_epoch(self, train_loader):
        """Run one training epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.float().to(self.device)  # Convert to float for BCE loss
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = self.model(inputs).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Convert outputs to predictions (0 or 1)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.int().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.float().to(self.device)
                
                # Forward pass
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                
                # Convert outputs to predictions (0 or 1)
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.int().cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Calculate additional metrics for detailed evaluation
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        metrics = {
            'val_loss': epoch_loss,
            'val_acc': epoch_acc,
            'val_f1': epoch_f1,
            'val_precision': precision,
            'val_recall': recall
        }
        
        return metrics, all_labels, all_preds
    
    def train(self, train_loader, val_loader, epochs=30, checkpoint_dir='checkpoints'):
        """
        Train the model for the specified number of epochs
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            checkpoint_dir: Directory to save model checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            
            # Validate
            val_metrics, val_labels, val_preds = self.validate(val_loader)
            val_loss = val_metrics['val_loss']
            val_acc = val_metrics['val_acc']
            val_f1 = val_metrics['val_f1']
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Save checkpoint if validation loss improves
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(checkpoint_dir, 'best_model_loss.pt'))
                self.epochs_no_improve = 0
                print("Saved best model by loss!")
            
            # Save checkpoint if validation F1 improves
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, os.path.join(checkpoint_dir, 'best_model_f1.pt'))
                print("Saved best model by F1!")
            else:
                self.epochs_no_improve += 1
            
            # Early stopping
            if self.epochs_no_improve >= self.early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
            
            print("-" * 50)
        
        total_time = time.time() - start_time
        print(f"Training complete in {total_time/60:.2f} minutes")
        
        # Plot training history
        self.plot_training_history()
        
        return self.history
    
    def plot_training_history(self):
        """Plot the training and validation metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        # Plot F1 score
        plt.subplot(1, 3, 3)
        plt.plot(self.history['train_f1'], label='Train F1 Score')
        plt.plot(self.history['val_f1'], label='Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training and Validation F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def plot_confusion_matrix(self, val_loader):
        """Plot confusion matrix for validation data"""
        # Get predictions
        _, labels, preds = self.validate(val_loader)
        
        # Create confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Autistic', 'Non-Autistic'],
                    yticklabels=['Autistic', 'Non-Autistic'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
    
    def load_best_model(self, checkpoint_path):
        """Load the best model from checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']+1} with validation metrics:")
        print(f"Loss: {checkpoint['val_loss']:.4f}, Acc: {checkpoint['val_acc']:.4f}, F1: {checkpoint['val_f1']:.4f}")
        return checkpoint

def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        dict: Dictionary of test metrics
    """
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs)
            
            # Get predictions
            preds = (probs > 0.5).int().cpu().numpy()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Print results
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Autistic', 'Non-Autistic'],
                yticklabels=['Autistic', 'Non-Autistic'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig('test_confusion_matrix.png')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    } 