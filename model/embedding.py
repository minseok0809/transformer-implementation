import torch
import torch.nn as nn
import numpy as np


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):   
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)

    def forward(self, x):
        x = self.embedding(x)
        return x 

   
class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length,
                 embedding_dim, sinusoidal_wave):   
        super(PositionalEmbedding, self).__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.sinusoidal_wave = sinusoidal_wave

    def forward(self, x):
        
        seq_len = x.size()[1]
        positions = np.arange(seq_len)[:, np.newaxis]
        dimentions = np.arange(self.embedding_dim)[np.newaxis,:]
        angles = positions/ np.power(self.sinusoidal_wave, 2*(dimentions//2)/self.embedding_dim)

        pos_encoding = np.zeros(angles.shape)
        pos_encoding[:,0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:,1::2] = np.cos(angles[:, 1::2])
        
        pos_encoding = torch.FloatTensor(pos_encoding).cuda()

        x = x + pos_encoding

        """
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
        """
        return x


class WordEmbedding(nn.Module):
    def __init__(self, max_seq_length, vocab_size, embedding_dim, sinusoidal_wave):    
        super(WordEmbedding, self).__init__()
        self.embedding_layer = InputEmbedding(vocab_size, embedding_dim)
        self.positional_embeddings = PositionalEmbedding(max_seq_length, embedding_dim, sinusoidal_wave)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.positional_embeddings(x)

        return x 

 

