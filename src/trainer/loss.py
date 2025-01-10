import torch
from torch import nn

class WeightedMultiClassLogLoss(nn.Module):
    def __init__(self, class_weights=None, device=None):
        """
        Initializes the Weighted Multi-Class Log Loss (WMCLL).
        
        Args:
            class_weights (torch.Tensor): A tensor of shape (num_classes,) containing weights for each class.
            device (torch.device): The device to perform the computation on.
        """
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None

    def forward(self, logits, targets):
        """
        Computes the Weighted Multi-Class Log Loss.
        
        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth of shape (batch_size, num_classes) as one-hot encoded.
        
        Returns:
            torch.Tensor: The computed weighted multi-class log loss.
        """
        logits, targets = logits.to(self.device), targets.to(self.device)
        
        if self.class_weights is not None:
            weights = self.class_weights.to(self.device)
        else:
            weights = torch.ones(logits.size(1), device=self.device)

        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        weighted_log_loss = -torch.sum(weights * targets * log_probs, dim=1)
        loss = torch.mean(weighted_log_loss)

        return loss
    

def calculate_class_weights(df, label_column):
    """
    Calculates class weights based on the inverse frequency of labels in the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the label data.
        label_column (str): Name of the column containing class labels.

    Returns:
        tuple: 
            - class_to_weight (dict): Mapping of class indices to weights.
            - class_weights_tensor (torch.Tensor): Class weights as a PyTorch tensor.
    """
    # Count the occurrences of each class
    class_counts = df[label_column].value_counts()

    # Total number of samples
    total_samples = len(df)

    # Calculate weights as inverse frequency
    class_weights = total_samples / class_counts

    # Convert to a PyTorch tensor
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)

    # Map weights to class indices
    class_to_weight = dict(zip(class_counts.index, class_weights))

    return class_to_weight, class_weights_tensor
