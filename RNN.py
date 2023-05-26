import torch
from torch import nn

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("Using device: {}".format(device))
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim= hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        x.to(device)
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # get RNN outputs
        out, hn = self.rnn(x, h0)
        # get final output
        output = self.linear(out[:, -1, :])
        return output
