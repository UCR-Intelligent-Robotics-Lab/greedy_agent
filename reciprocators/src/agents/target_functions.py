import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentScalarPredictor(nn.Module):
    def __init__(self, input_dims: int, n_output_dims: int, n_latent_var: int):
        """
        Similar to the RecurrentCritic class but with a linear instead of convolutional layer.
        :param input_dims: Dims of the flattened of the input space.
        :param n_output_dims: Number of output dimensions.
        :param n_latent_var: Number of hidden units in the hidden linear layer.
        """
        super(RecurrentScalarPredictor, self).__init__()
        # Set up critic architecture
        self.input_dims = input_dims
        self.output_dims = n_output_dims
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dims, n_latent_var),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(n_latent_var, n_latent_var, num_layers=1, batch_first=False)
        self.critic = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_latent_var, n_output_dims),
        )
        self.hidden_state = None

    def reset(self):
        self.hidden_state = None

    def forward(self, state_bs: torch.Tensor, time_dim: bool = True):
        if time_dim:
            T, bsz = state_bs.shape[:2]
            state_bs = state_bs.flatten(end_dim=1)
            state_encoding = self.state_encoder(state_bs).view(T, bsz, -1)  # (T, bsz, n_latent_var)
            out, _ = self.rnn(state_encoding)
        else:
            state_encoding = self.state_encoder(state_bs).unsqueeze(dim=0)  # (1, bsz, n_latent_var)
            if self.hidden_state is None:
                out, self.hidden_state = self.rnn(state_encoding)
            else:
                out, self.hidden_state = self.rnn(state_encoding, self.hidden_state)

        return self.critic(out)


class ScalarPredictor(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, hidden_layer_size: int = 32):
        super(ScalarPredictor, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_layer_size = hidden_layer_size

        self.model = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.output_dims)
        )

    def forward(self, x):
        return self.model(x)


class DiscreteClassifier(nn.Module):
    def __init__(self, num_classes: int, input_dims: int, hidden_layer_size: int = 32, num_output_dims: int = 1):
        super(DiscreteClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_dims = input_dims
        self.hidden_layer_size = hidden_layer_size
        self.num_output_dims = num_output_dims

        self.model = self.init_model()

    def init_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, self.num_classes * self.num_output_dims),
        )
        return model

    def forward(self, x):
        """
        Can have distributions over multiple output dimensions. The output tensor is then of shape
        (bsz * num_output_dims, num_classes), softmaxed over the last dimension - it is assumed each output
        dimension has the same number of possible classes.
        :param x: A tensor of shape (bsz, input_dims) representing the input.
        """
        logits = self.model(x)
        if self.num_output_dims > 1:
            logits = logits.view(x.size(0) * self.num_output_dims, self.num_classes)

        return F.log_softmax(logits, dim=-1)
