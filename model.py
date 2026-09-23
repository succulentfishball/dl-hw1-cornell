import torch
from torch import nn
from d2l import torch as d2l


SETTINGS = ("baseline", "dropout", "weight_decay", "batch_norm")

def init_cnn(module):
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

class LeNet(d2l.Classifier):
    """The LeNet-5 model."""
    def __init__(self, setting="baseline", dropout=0.3, weight_decay=1e-4, lr=0.1, num_classes=10):
        super().__init__()
        if setting not in SETTINGS:
            raise ValueError(f"Unknown setting: {setting}")
        bn = setting == "batch_norm"
        use_dropout = setting == "dropout"

        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), 
            nn.BatchNorm2d(6) if bn else nn.Identity(),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), 
            nn.BatchNorm2d(16) if bn else nn.Identity(),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), 
            nn.BatchNorm1d(120) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout) if use_dropout else nn.Identity(),
            nn.LazyLinear(84), 
            nn.BatchNorm1d(84) if bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout) if use_dropout else nn.Identity(),
            nn.LazyLinear(num_classes))
        
    
    def configure_optimizers(self):
        decay = self.weight_decay if self.setting == "weight_decay" else 0.0

        return torch.optim.AdamW(
            [
                {
                    "params": [
                        p for p in self.parameters() if p.ndim > 1
                    ],
                    "weight_decay": decay,
                },
                {
                    "params": [
                        p for p in self.parameters() if p.ndim <= 1
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=self.lr,
        )