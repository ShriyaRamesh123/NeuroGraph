import scipy.io
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import wandb
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import ChebConv
from sklearn.metrics import f1_score, roc_auc_score

# Load data
mat = scipy.io.loadmat('all_data.mat')
X = torch.tensor(mat['X'], dtype=torch.float)
y = torch.tensor(mat['y'][0], dtype=torch.float)
edge_indices = torch.tensor(mat['edge_indices'], dtype=torch.long)
edge_weights = torch.tensor(mat['edge_weights'][0], dtype=torch.float)

# Build graph
data = Data(x=X[:, :2052], y=y, edge_index=edge_indices, edge_attr=edge_weights)
data = RandomNodeSplit(num_val=0.1, num_test=0.0)(data)

# Define model
class DeepChebNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, K, dropout):
        super().__init__()
        self.conv_input = ChebConv(input_dim, hidden_dims[0], K)
        self.hidden_convs = nn.ModuleList([
            ChebConv(hidden_dims[i], hidden_dims[i+1], K)
            for i in range(len(hidden_dims) - 1)
        ])
        self.conv_output = ChebConv(hidden_dims[-1], output_dim, K)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.dropout = dropout
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.activation(self.conv_input(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.hidden_convs:
            x = self.activation(conv(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_output(x, edge_index, edge_weight)
        x = self.mlp(x)
        return torch.sigmoid(x)

# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'min': 1e-4, 'max': 1e-1},
        'weight_decay': {'min': 1e-6, 'max': 1e-3},
        'dropout': {'min': 0.1, 'max': 0.5},
        'num_hidden_layers': {'values': [3, 5, 7, 9]},
        'hidden_dim': {'values': [32, 64, 128]},
        'K': {'values': [2, 3, 5]}
    }
}

# Training function
def train():
    with wandb.init() as run:
        config = run.config
        hidden_dims = [config.hidden_dim] * config.num_hidden_layers
        model = DeepChebNet(
            input_dim=data.x.shape[1],
            hidden_dims=hidden_dims,
            output_dim=128,
            K=config.K,
            dropout=config.dropout
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.BCELoss()

        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            outputs = model(data.x, data.edge_index, data.edge_attr).squeeze()
            train_loss = criterion(outputs[data.train_mask], data.y[data.train_mask])
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # Training metrics
            train_probs = outputs[data.train_mask].detach().cpu().numpy()
            train_labels = data.y[data.train_mask].cpu().numpy()
            train_preds = (train_probs > 0.5).astype(float)
            train_acc = (train_preds == train_labels).mean()

            # Validation metrics
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index, data.edge_attr).squeeze()
                val_probs = val_out[data.val_mask].cpu().numpy()
                val_labels = data.y[data.val_mask].cpu().numpy()
                val_preds = (val_probs > 0.5).astype(float)

                val_acc = (val_preds == val_labels).mean()
                val_f1 = f1_score(val_labels, val_preds, zero_division=0)
                try:
                    val_auc = roc_auc_score(val_labels, val_probs)
                except ValueError:
                    val_auc = float('nan')

            wandb.log({
                #'epoch': epoch,
                'loss': train_loss.item(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
            })

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Loss: {train_loss.item():.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                      f"F1: {val_f1:.4f} | AUROC: {val_auc:.4f}")

# Launch sweep
if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project='deepchebnet_sweep2')
    wandb.agent(sweep_id, function=train, count=25)
