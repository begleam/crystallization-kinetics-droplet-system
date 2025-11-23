import torch
import torch.nn as nn
from torchvision import models

class Regressor(nn.Module):
    def __init__(self, emb_dim=1024):
        super(Regressor, self).__init__()
        
        # Load a pre-trained EfficientNet-B4
        self.efficientnet = models.efficientnet_b4(pretrained=True)
        
        # Modify the first convolutional layer to accept one-channel input
        original_first_conv = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(1, original_first_conv.out_channels,
                                                     kernel_size=original_first_conv.kernel_size,
                                                     stride=original_first_conv.stride,
                                                     padding=original_first_conv.padding,
                                                     bias=False)
        
        # Initialize the modified first convolutional layer with the pretrained weights for one channel
        with torch.no_grad():
            original_first_conv_weights = original_first_conv.weight.mean(dim=1, keepdim=True)
            self.efficientnet.features[0][0].weight = nn.Parameter(original_first_conv_weights)
        
        # Replace the classifier layer with a new one
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.1, inplace=True),
            nn.ReLU(),
        )
        self.regressor0 = nn.Sequential(
            nn.Linear(num_features, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(p=0.1, inplace=True),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )
        self.regressor1 = nn.Sequential(
            nn.Linear(num_features, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(p=0.1, inplace=True),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )
        self.regressor2 = nn.Sequential(
            nn.Linear(num_features, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(p=0.1, inplace=True),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )
        self.regressor3 = nn.Sequential(
            nn.Linear(num_features, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(p=0.1, inplace=True),
            nn.ReLU(),
            nn.Linear(emb_dim, 2),
        )
        self.regressor4 = nn.Sequential(
            nn.Linear(num_features, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(p=0.1, inplace=True),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )
    
    def forward(self, x):
        z = self.efficientnet(x)
        return self.regressor0(z), self.regressor1(z), self.regressor2(z), self.regressor3(z), self.regressor4(z)
