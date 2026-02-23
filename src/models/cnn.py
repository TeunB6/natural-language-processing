import torch.nn as nn
import torch

class CNNClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, num_filters: int=100, filter_sizes: list[int]=[3, 4, 5]):
        super(CNNClassifier, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        conv_outputs = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_outputs = [torch.max(conv_output, dim=2)[0] for conv_output in conv_outputs]
        cat = torch.cat(pooled_outputs, dim=1)
        drop = self.dropout(cat)
        fc = self.fc(drop)
        return self.softmax(fc)