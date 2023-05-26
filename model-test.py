import pandas as pd
import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cpu')
print("Using device: {}".format(device))
class RNN(nn.Module):
    def __init__(self, hidden_dim=10, num_layers=1):
        super(RNN, self).__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.num_layers = num_layers
        self.hidden_dim= hidden_dim
        self.rnn = nn.RNN(self.input_dim, hidden_dim, self.num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        x.to(device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # get RNN outputs
        out, hn = self.rnn(x, h0)
        # get final output
        output = self.linear(out[:, -1, :])
        return output
class LSTM(nn.Module):
    def __init__(self,hidden_dim=10, num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = 1
        self.output_dim = 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        x.to(device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out
def create_dataset(seq, window_size):
    x = []
    y = []
    for i in range(len(seq) - window_size):
        # windows are the training features
        window = seq[i : i+ window_size]
        # The RNN is many-to-many architecture, therefore the output size
        label = seq[i + window_size: i + window_size + 1]
        x.append(window)
        y.append(label)

    # Convert to numpy array for easier indexing
    x = np.array(x)
    y = np.array(y)

    # Split data into training and testing set
    train_size = int(0.8*len(seq))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Convert to tensor
    x_train = torch.from_numpy(x_train).type(torch.Tensor).to(device)
    x_test = torch.from_numpy(x_test).type(torch.Tensor).to(device)
    y_train = torch.from_numpy(y_train).type(torch.Tensor).to(device)
    y_test = torch.from_numpy(y_test).type(torch.Tensor).to(device)

    x_train = x_train.unsqueeze(2)
    x_test = x_test.unsqueeze(2)
    # return processed data
    return x_train, x_test, y_train, y_test

scaler = MinMaxScaler(feature_range=(-1, 1))

data = pd.read_csv("sp500_index.csv", index_col='Date')
data['S&P500'] = scaler.fit_transform(data['S&P500'].values.reshape(-1, 1))
X_train, X_test, y_train, y_test = create_dataset(data['S&P500'].values, 12)

best_lstm = LSTM(hidden_dim=10, num_layers=1)
best_rnn =
#%%
