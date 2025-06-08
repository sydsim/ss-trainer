import torch.nn as nn



class BasicModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim = 64,
        num_layers = 2,
    ):
        super().__init__()

        self.input_size = input_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_dim

        self.module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[
                layer
                for _ in range(num_layers)
                for layer in [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ] 
            ],
            nn.Linear(hidden_dim, 3)
        )
    def forward(self, x):
        res = self.module(x[:, -1])
        return res
