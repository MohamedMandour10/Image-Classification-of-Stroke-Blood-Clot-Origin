from torch import nn

class BinaryCCTWrapper(nn.Module):
    def __init__(self, base_model, freeze_layers=True, dropout_rate=0.3, trainble_layers=5):
        super().__init__()
        self.base_model = base_model
        
        if freeze_layers:
            for param in self.base_model.parameters():
                param.requires_grad = False
        else:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
            for block in self.base_model.classifier.blocks[-trainble_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
            
            for param in self.base_model.classifier.norm.parameters():
                param.requires_grad = True
        
        in_features = self.base_model.classifier.fc.in_features
        
        # Replace base model's classifier head
        self.base_model.classifier.fc = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features * 2, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.base_model(x)