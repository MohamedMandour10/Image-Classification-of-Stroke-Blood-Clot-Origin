from timm import create_model
import torch.nn as nn

class OptimizedPoolformer:
    """
    A class to encapsulate the creation and customization of an optimized Poolformer model.
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.3, trainable_backbone_layers=10):
        """
        Initializes the OptimizedPoolformer class.

        Args:
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout rate for the classifier head.
            trainable_backbone_layers (int): Number of trainable backbone layers from the end.
        """
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.trainable_backbone_layers = trainable_backbone_layers
        self.model = self.create_model()

    def create_model(self):
        """
        Creates a Poolformer model with the specified optimizations.

        Returns:
            nn.Module: Optimized Poolformer model ready for fine-tuning.
        """
        # Create the base Poolformer model
        model = create_model('poolformer_s36', pretrained=True)

        # Freeze all layers by default
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the last `trainable_backbone_layers` stages in the backbone
        if hasattr(model, "stages"):
            for stage in model.stages[-self.trainable_backbone_layers:]:
                for param in stage.parameters():
                    param.requires_grad = True
        else:
            raise AttributeError("The Poolformer model does not have the expected 'stages' attribute.")

        # Modify the classifier head
        if hasattr(model.head, "fc"):  # Access the fully connected layer in the head
            in_features = model.head.fc.in_features
            model.head.fc = nn.Sequential(
                nn.BatchNorm1d(in_features),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features, in_features // 2),  # Intermediate dense layer
                nn.ReLU(),
                nn.BatchNorm1d(in_features // 2),
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(in_features // 2, self.num_classes)
            )
        else:
            raise AttributeError("The Poolformer model does not have the expected 'fc' in the head.")

        # Ensure classifier head is trainable
        for param in model.head.parameters():
            param.requires_grad = True

        return model

    def get_model(self):
        """
        Returns the fully configured and optimized Poolformer model.

        Returns:
            nn.Module: Optimized Poolformer model.
        """
        return self.model
