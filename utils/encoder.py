import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_matrix, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.rnn = nn.LSTM(embedding_matrix.size(1), hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # embedded: [batch_size, seq_len, embedding_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

# Make sure the file is named 'encoder.py' and saved in the 'utils' directory.
