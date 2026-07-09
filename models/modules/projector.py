import torch
import torch.nn as nn

class ProjectionHead(nn.Module):

    def __init__(
        self, input_dim = 2048, hidden_dim = 4096, output_dim = 256
    ):
        super(ProjectionHead, self).__init__()
        self.linear1 = nn.Linear(input_dim,  hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.act     = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
    
class PredictionHead(nn.Module):

    def __init__(
        self, input_dim = 2048, hidden_dim = 4096, output_dim = 256
    ):
        super(PredictionHead, self).__init__()
        self.linear1 = nn.Linear(input_dim,  hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.act     = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
    
