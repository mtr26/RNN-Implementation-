import torch as th
import torch.nn as nn
import torch.nn.functional as F



class RNNLayer(nn.Module):
    """
    RNN layer with tanh activation function
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state):
        combined = th.cat((x, hidden_state), dim=1)
        hidden_state = th.tanh(self.i2h(combined))
        output = self.h2o(hidden_state)
        return output, hidden_state

    def init_hidden(self, batch_size, device):
        return nn.init.kaiming_uniform_(th.empty(batch_size, self.hidden_size, device=device))



class RNN(nn.Module):
    """
    RNN model with multiple layers
    """
    def __init__(self, input_size, hidden_size, output_size, num_layer = 1):
        super(RNN, self).__init__()
        self.layers = nn.ModuleList(
            [RNNLayer(input_size, hidden_size, output_size)] +
            [RNNLayer(hidden_size, hidden_size, output_size) for _ in range(num_layer - 1)]
        )

    def forward(self, x, hidden_states=None):
        seq_len, batch_size, _ = x.size()
        if hidden_states is None:
            hidden_states = [layer.init_hidden(batch_size, x.device)  for layer in self.layers]
        outputs = []
        for t in range(seq_len):
            input_x = x[t]
            new_hidden_states = []
            for i, layer in enumerate(self.layers):
                out, hidden_state = layer(input_x, hidden_states[i])
                new_hidden_states.append(hidden_state)
                input_x = out
            hidden_states = new_hidden_states
            outputs.append(out)
        return th.stack(outputs), hidden_states






