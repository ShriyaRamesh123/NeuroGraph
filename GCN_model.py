#Libraries
import torch
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import ChebConv




n1=7
dout=0.2

#DeepGCN module
relu=nn.ReLU()
#from torch_geometric.utils import dropout_edge

class DeepChebNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, K, dropout, dropedge_prob=0.3):
    
        #Args:
         #   input_dim: Input feature dimension
          #  hidden_dims: List of hidden layer dimensions (length 9 for 10 total layers)
           # output_dim: Output dimension
            #K: Chebyshev polynomial order
            #dropout: Dropout probability

        super().__init__()

        self.edge_drop_prob = dropedge_prob

        # Input layer
        self.conv_input = ChebConv(input_dim, hidden_dims[0], K)

        # Hidden layers (8 total)
        self.hidden_convs = nn.ModuleList()
        for i in range(1, n1):
            self.hidden_convs.append(ChebConv(hidden_dims[i-1], hidden_dims[i], K))

        # Output layer
        self.conv_output = ChebConv(hidden_dims[-1], output_dim, K)


        # MLP classifier (1x1 convolutions)
        self.mlp = nn.Sequential(
             nn.Dropout(dropout),
            nn.Linear(output_dim, 128),  # First layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1)
        )

        self.dropout = dropout
        self.activation = nn.ReLU()

        # Initialization
        self.reset_parameters()

    def reset_parameters(self):
        for conv in [self.conv_input, *self.hidden_convs, self.conv_output]:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        # Input layer
        '''if self.training and self.edge_drop_prob > 0:
            edge_index, edge_weight = dropout_edge(
                edge_index,
                edge_weight,
                p=self.edge_drop_prob,
                force_undirected=False  # For undirected graphs
            )'''
        x = self.activation(self.conv_input(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Hidden layers
        for conv in self.hidden_convs:
            x = self.activation(conv(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer (no activation)
        x = self.conv_output(x, edge_index, edge_weight)
        x=self.mlp(x)
        return torch.sigmoid(x)