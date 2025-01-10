import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

class MetricTracker:
    def __init__(self):
        self.metrics = {
            'train_loss': [], 'train_acc': [], 
            'train_recall': [], 'train_f1': [],
            'val_acc': [],  'val_recall': [], 'val_f1': []
        }
    
    def update(self, metrics_dict):
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
    
    def plot_metrics(self, save_path='/kaggle/working/training_metrics.png'):
        epochs = range(1, len(self.metrics['train_loss']) + 1)
        
        # Create a figure with 3x2 subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Training and Validation Metrics', fontsize=16)
        
        # Plot Loss
        axes[0, 0].plot(epochs, self.metrics['train_loss'], label='Training Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot Accuracy
        axes[0, 1].plot(epochs, self.metrics['train_acc'], label='Training')
        axes[0, 1].plot(epochs, self.metrics['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        
        # Plot Recall
        axes[1, 1].plot(epochs, self.metrics['train_recall'], label='Training')
        axes[1, 1].plot(epochs, self.metrics['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot F1 Score
        axes[2, 0].plot(epochs, self.metrics['train_f1'], label='Training')
        axes[2, 0].plot(epochs, self.metrics['val_f1'], label='Validation')
        axes[2, 0].set_title('F1 Score')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('F1 Score')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Remove the empty subplot
        plt.show()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class ModelTrainer:
    def __init__(self, device, num_epochs=10):
        """
        Initializes the ModelTrainer.

        Args:
            device (str): Device to train on ('cpu' or 'cuda').
            num_epochs (int): Number of training epochs.
        """
        self.device = device
        self.num_epochs = num_epochs
        self.metric_tracker = MetricTracker()

    def create_model(self):
        """
        Creates and returns a binary classification model based on MobileNetV3-Large.
        
        Returns:
            model (torch.nn.Module): The binary classification model.
        """
        model = models.mobilenet_v3_large(weights=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        return model

    @staticmethod
    def calculate_metrics(all_predictions, all_labels):
        """
        Calculates recall and F1 score.

        Args:
            all_predictions (list): Predicted labels.
            all_labels (list): True labels.

        Returns:
            tuple: Recall and F1 score.
        """
        recall = recall_score(all_labels, all_predictions, average='binary')
        f1 = f1_score(all_labels, all_predictions, average='binary')
        return recall, f1

    def train_model(self, model, train_loader, val_loader):
        """
        Trains the model and validates it after each epoch.

        Args:
            model (torch.nn.Module): Model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

        best_val_acc = 0.0

        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            all_predictions = []
            all_labels_list = []

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} Training", unit="batch") as pbar:
                for images, labels in pbar:
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels_list.extend(labels.cpu().numpy())

                    pbar.set_postfix(loss=running_loss / (pbar.n + 1), accuracy=100. * correct / total)

            train_acc = 100. * correct / total
            train_recall, train_f1 = self.calculate_metrics(all_predictions, all_labels_list)

            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_labels_list = []

            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} Validation", unit="batch") as pbar:
                    for images, labels in pbar:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                        val_predictions.extend(predicted.cpu().numpy())
                        val_labels_list.extend(labels.cpu().numpy())

                        pbar.set_postfix(accuracy=100. * val_correct / val_total)

            val_acc = 100. * val_correct / val_total
            val_recall, val_f1 = self.calculate_metrics(val_predictions, val_labels_list)

            scheduler.step(val_acc)

            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            print(f'Training Loss: {running_loss/len(train_loader):.4f}')
            print(f'Training Metrics:')
            print(f'  Accuracy: {train_acc:.2f}%')
            print(f'  Recall: {train_recall:.4f}')
            print(f'  F1 Score: {train_f1:.4f}')
            print(f'Validation Metrics:')
            print(f'  Accuracy: {val_acc:.2f}%')
            print(f'  Recall: {val_recall:.4f}')
            print(f'  F1 Score: {val_f1:.4f}')

            self.metric_tracker.update({
                'train_loss': running_loss/len(train_loader),
                'train_acc': train_acc,
                'train_recall': train_recall,
                'train_f1': train_f1,
                'val_acc': val_acc,
                'val_recall': val_recall,
                'val_f1': val_f1
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')

            print('-' * 60)

        self.metric_tracker.plot_metrics()


def classify_images(model, image_paths, transform, device):
    model.to(device)
    
    # Placeholder for keeping non-empty images
    valid_images = []
    
    # Wrap image_paths with tqdm for the progress bar
    for image_path in tqdm(image_paths, desc="Classifying images", unit="image"):
        # Load the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
        image = image.to(device)

        # Predict the class (empty or not)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)  # 0 = empty, 1 = not empty
            
            # If predicted label is 1 (not empty), keep this image
            if predicted.item() == 1:
                valid_images.append(image_path)
    
    return valid_images