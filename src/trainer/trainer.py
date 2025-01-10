import torch
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall, F1Score
import wandb
import numpy as np
from tqdm import tqdm
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


class Trainer:
    def __init__(self, model, criterion, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion
        self.config = config 

        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=2,
            factor=0.5,
            min_lr=1e-6
        )

        # Initialize metrics with macro averaging
        self.metrics = {
            'accuracy': Accuracy(task='multiclass', num_classes=self.config.num_classes, average='macro').to(self.device),
            'precision': Precision(task='multiclass', num_classes=self.config.num_classes, average='macro').to(self.device),
            'recall': Recall(task='multiclass', num_classes=self.config.num_classes, average='macro').to(self.device),
            'f1': F1Score(task='multiclass', num_classes=self.config.num_classes, average='macro').to(self.device)
        }

        # Initialize training state
        self.best_model = None
        self.best_metric = float('inf') if self.config.early_stop_metric == 'loss' else float('-inf')
        self.history = {'train': [], 'val': []}
        self.epochs_without_improve = 0

        # Initialize W&B
        self.run = wandb.init(
            project=config.project_name,
            config={
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "model": model.__class__.__name__
            }
        )
        wandb.watch(model)

    def reset_metrics(self):
        """Reset all metrics at the start of each epoch"""
        for metric in self.metrics.values():
            metric.reset()

    def validate(self, val_loader):
        """Validate the model on the validation set"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        self.reset_metrics()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                batch_size = inputs.size(0)
                total_samples += batch_size

                outputs = self.model(inputs)
                one_hot = nn.functional.one_hot(labels, self.config.num_classes).float()
                total_loss += self.criterion(outputs, one_hot).item() * batch_size

                preds = outputs.argmax(1)
                for metric in self.metrics.values():
                    metric.update(preds, labels)

        metrics_dict = {
            'loss': total_loss / total_samples,
            'accuracy': self.metrics['accuracy'].compute().item(),
            'precision': self.metrics['precision'].compute().item(),
            'recall': self.metrics['recall'].compute().item(),
            'f1': self.metrics['f1'].compute().item()
        }
        wandb.log({f"val_{k}": v for k, v in metrics_dict.items()})
        return metrics_dict

    def train_epoch(self, train_loader, epoch_idx):
        """Train for one epoch and log metrics against batches."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        self.reset_metrics()

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch_idx + 1}")):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            one_hot = nn.functional.one_hot(labels, self.config.num_classes).float()
            loss = self.criterion(outputs, one_hot)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            preds = outputs.argmax(1)
            for metric in self.metrics.values():
                metric.update(preds, labels)

            # Compute batch metrics
            batch_metrics = {
                'loss': loss.item(),
                'accuracy': self.metrics['accuracy'].compute().item(),
                'precision': self.metrics['precision'].compute().item(),
                'recall': self.metrics['recall'].compute().item(),
                'f1': self.metrics['f1'].compute().item()
            }

            # Log metrics to W&B
            wandb.log({f"train_batch_{k}": v for k, v in batch_metrics.items()}, step=epoch_idx * len(train_loader) + batch_idx)

        # Compute epoch-level metrics
        epoch_metrics = {
            'loss': total_loss / total_samples,
            'accuracy': self.metrics['accuracy'].compute().item(),
            'precision': self.metrics['precision'].compute().item(),
            'recall': self.metrics['recall'].compute().item(),
            'f1': self.metrics['f1'].compute().item()
        }
        wandb.log({f"train_epoch_{k}": v for k, v in epoch_metrics.items()}, step=epoch_idx + 1)

        return epoch_metrics

    def train(self, train_loader, val_loader, epochs=10):
        """Main training loop"""
        print(f"Starting training on device: {self.device}")
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validation phase
            val_metrics = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])

            # Store history
            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)

            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s")
            print("Train:", " ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))
            print("Val:", " ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items()))

            # Check for improvement
            current = val_metrics[self.config.early_stop_metric]
            improved = (self.config.early_stop_metric == 'loss' and 
                       current < self.best_metric - self.config.min_delta) or \
                      (self.config.early_stop_metric != 'loss' and 
                       current > self.best_metric + self.config.min_delta)

            if improved:
                self.best_metric = current
                self.best_model = self.model.state_dict().copy()
                self.epochs_without_improve = 0
                print(f"Best {self.config.early_stop_metric}: {self.best_metric:.4f}")
                torch.save(self.model.state_dict(), f"best_model_epoch_{epoch + 1}.pth")
            else:
                self.epochs_without_improve += 1
                if self.epochs_without_improve >= self.config.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")

        # Restore best model
        self.model.load_state_dict(self.best_model)
        wandb.save("best_model.pth")

        return self.model


    def plot_metrics(self, final=False):
        """Plot training metrics"""
        metrics = ['loss', 'accuracy', 'precision', 'f1']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for ax, metric in zip(axes, metrics):
            train_metric = [epoch[metric] for epoch in self.history['train']]
            val_metric = [epoch[metric] for epoch in self.history['val']]
            ax.plot(train_metric, label='Train')
            ax.plot(val_metric, label='Val')
            
            ax.set_title(f'{metric.capitalize()}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if final:
            plt.savefig('training_metrics.png')
            plt.show()
        else:
            plt.draw()
            plt.pause(0.1)
        plt.close(fig)

    def plot_confusion_matrix(self, loader, title="Confusion Matrix"):
        """Plot confusion matrix on a given dataset (train/val/test)"""
        self.model.eval()
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                preds = outputs.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(self.config.num_classes))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(self.config.num_classes))
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, cmap="Blues", values_format='d')
        plt.title(title)
        plt.savefig(f"{title.replace(' ', '_').lower()}.png")
        plt.show()
    
        # Log confusion matrix to W&B
        wandb.log({title: wandb.Image(fig)})
        plt.close(fig)
