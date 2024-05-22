import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
import pdb

from .sublayer import MultiHeadAttention, ResidualConnection, FeedForwardNetwork
from .embedding import WordEmbedding

class Encoder(nn.Module):
    def __init__(self, embedding_dim, feed_forward_size, num_attention_heads, attention_dropout_prob):    
        super(Encoder, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_attention_heads)
        self.residual_connection = ResidualConnection(attention_dropout_prob, embedding_dim)
        self.feed_forward_network = FeedForwardNetwork(embedding_dim, feed_forward_size)

    def forward(self, encoder_input, max_seq_length):

        first_encoder_output = self.multi_head_attention(encoder_input, encoder_input, max_seq_length)
        first_encoder_output = self.residual_connection(encoder_input, first_encoder_output)

        encoder_output = self.feed_forward_network(first_encoder_output)
        encoder_output = self.residual_connection(first_encoder_output, encoder_output)
        
        return encoder_output 


class Decoder(nn.Module):
    def __init__(self, embedding_dim, feed_forward_size, num_attention_heads, attention_dropout_prob): 
        super(Decoder, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_attention_heads)
        self.masked_multi_head_attention = MultiHeadAttention(embedding_dim, num_attention_heads, is_mask=True)
        self.residual_connection = ResidualConnection(attention_dropout_prob, embedding_dim)
        self.feed_forward_network = FeedForwardNetwork(embedding_dim, feed_forward_size)


    def forward(self, encoder_output, decoder_input, max_seq_length):

        first_decoder_output = self.masked_multi_head_attention(decoder_input, decoder_input, max_seq_length)
        first_decoder_output = self.residual_connection(decoder_input, first_decoder_output)

        second_decoder_output = self.multi_head_attention(first_decoder_output, encoder_output, max_seq_length)
        second_decoder_output = self.residual_connection(first_decoder_output, second_decoder_output)

        decoder_output = self.feed_forward_network(second_decoder_output)
        decoder_output = self.residual_connection(second_decoder_output, decoder_output)

        return decoder_output

class TransformerOutput(nn.Module):
    def __init__(self, embedding_dim, vocab_size):    
        super(TransformerOutput, self).__init__()

        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
   
    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)

        return x  
    
    def prob(self,decoded_output):
        x = self.linear(decoded_output)
        output_probabilities = self.softmax(x)
        # LOGGER.info("output probabilities size={}".format(output_probabilities.size()))
        
        return output_probabilities


class TransformerForTranslation(nn.Module):
    def __init__(self, max_seq_length, vocab_size,
                 embedding_dim, sinusoidal_wave,
                 num_sub_layer, feed_forward_size, num_attention_heads,
                 attention_dropout_prob):

        super(TransformerForTranslation,self).__init__()

        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sinusoidal_wave = sinusoidal_wave
        self.num_sub_layer = num_sub_layer
        self.feed_forward_size = feed_forward_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout_prob = attention_dropout_prob
        
        self.embeddings = WordEmbedding(max_seq_length, vocab_size,
                                        embedding_dim, sinusoidal_wave)

        self.encoder = Encoder(embedding_dim, feed_forward_size, num_attention_heads,
                               attention_dropout_prob)
        self.decoder = Decoder(embedding_dim, feed_forward_size, num_attention_heads,
                               attention_dropout_prob)
        
        self.dropout = nn.Dropout(attention_dropout_prob)

        self.output = TransformerOutput(embedding_dim, vocab_size)

    def forward(self, src_text, tgt_text):

        src_embedding_output = self.embeddings(src_text)
        src_embedding_output = self.dropout(src_embedding_output)
        encoder_output = src_embedding_output

        for sub_layer_idx in range(self.num_sub_layer):
            encoder_output = self.encoder(encoder_output, self.max_seq_length)

        tgt_embedding_output = self.embeddings(tgt_text)
        tgt_embedding_output = self.dropout(tgt_embedding_output)
        decoder_output = tgt_embedding_output

        for sub_layer_idx in range(self.num_sub_layer):
            decoder_output = self.decoder(encoder_output, decoder_output, self.max_seq_length)

        output = self.output(decoder_output)
        
        return output


