import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_nonlinear_func, expand_tensor 



class MLP(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=8,
        output_dim=2,
        nonlinearity="relu",
        num_hidden_layers=1,
        use_nonlinearity_output=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        self.act = get_nonlinear_func(nonlinearity)

        layers = []
        for i in range(num_hidden_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)

        final_in_dim = input_dim if num_hidden_layers == 0 else hidden_dim
        self.fc = nn.Linear(final_in_dim, output_dim)

    def forward(self, input):
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)

        hidden = x

        for layer in self.layers:
            hidden = self.act(layer(hidden))

        output = self.fc(hidden)

        if self.use_nonlinearity_output:
            output = self.act(output)

        return output