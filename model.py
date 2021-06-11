import torch.nn as nn


class PGN(nn.Module):
    def __init__(self, state_space, action_space, dropout=0.0, n_dim=128):
        super(PGN, self).__init__()
        self.in_dim = state_space
        self.out_dim = action_space
        self.dropout = dropout
        self.n_dim = n_dim
        
        self.layers = self._generate_layers()
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def _generate_layers(self):
        layers = []
        layers.append(nn.Linear(self.in_dim, self.n_dim, bias=False))
        layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.n_dim, self.out_dim, bias=False))
        layers.append(nn.Softmax(dim=-1))
        return layers
