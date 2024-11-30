import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)

class DecoderWithAttention(nn.Module):
    def __init__(self, output_size, embedding_matrix, hidden_size, num_layers=1):
        super(DecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.attention = Attention(hidden_size)
        self.rnn = nn.LSTM(hidden_size + embedding_matrix.size(1), hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)

        attention_weights = self.attention(hidden[-1], encoder_outputs)
        attention_weights = attention_weights.unsqueeze(1)
        context = torch.bmm(attention_weights, encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        prediction = self.fc(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))
        return prediction, hidden, cell

# Make sure the file is named 'decoder.py' and saved in the 'utils' directory.
