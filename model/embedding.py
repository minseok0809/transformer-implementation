import torch
from torch import nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):   
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        return x 
     
class PositionalEmbedding(nn.Module):
    def __init__(self, batch_size, max_seq_length,
                 embedding_dim, sinusoidal_wave):   
        super(PositionalEmbedding, self).__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.sinusoidal_wave = sinusoidal_wave

    def forward(self, x):
        position = torch.zeros_like(x)

        for sentence in range(self.batch_size):
            for kth_word in range(self.max_seq_length):
                for i in range(self.embedding_dim // 2):
                    ith_value = 2 * i
                    rational_number_input =  torch.tensor(kth_word / \
                                                          (self.sinusoidal_wave ** ((ith_value) / self.embedding_dim)))

                    position[sentence, kth_word, ith_value] = torch.sin(rational_number_input)
                    position[sentence, kth_word, ith_value + 1] = torch.cos(rational_number_input)

        x = x + position

        return x


class WordEmbedding(nn.Module):
    def __init__(self, batch_size, max_seq_length, vocab_size, embedding_dim, sinusoidal_wave):    
        super(WordEmbedding, self).__init__()
        self.embedding_layer = InputEmbedding(vocab_size, embedding_dim)
        self.positional_embeddings = PositionalEmbedding(batch_size, max_seq_length,
                                                        embedding_dim, sinusoidal_wave)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.positional_embeddings(x)

        return x 