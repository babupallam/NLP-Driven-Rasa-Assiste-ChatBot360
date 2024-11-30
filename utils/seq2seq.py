import torch
import torch.nn as nn

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        output_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, output_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(source)

        input = target[:, 0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            top1 = output.argmax(1)
            input = target[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

# Make sure the file is named 'seq2seq.py' and saved in the 'utils' directory.
