import torch
from torch import nn
from .embedding import InputEmbedding, PositionalEmbedding, WordEmbedding
from .sublayer import SelfAttention, ResidualConnection, FeedForwardNetwork

class Encoder(nn.Module):
    def __init__(self, embedding_dim, feed_forward_size, num_attention_heads,
                 attention_dropout_prob):    
        super(Encoder, self).__init__()

        self.self_attention = SelfAttention(embedding_dim, num_attention_heads)
        self.residual_connection = ResidualConnection(attention_dropout_prob, embedding_dim)
        self.feed_forward_network = FeedForwardNetwork(embedding_dim, feed_forward_size)
    
    def forward(self, embedding_output, encoder_output, src_mask, sub_layer_idx):

        if sub_layer_idx == 0:
            encoder_output = self.self_attention(embedding_output, embedding_output, src_mask)
        elif sub_layer_idx > 0:
            encoder_output = self.self_attention(encoder_output, encoder_output, src_mask)

        encoder_output = self.residual_connection(embedding_output, encoder_output)

        encoder_output = self.feed_forward_network(encoder_output)
        encoder_output = self.residual_connection(embedding_output, encoder_output)
        
        return encoder_output 

class Decoder(nn.Module):
    def __init__(self, embedding_dim, feed_forward_size, num_attention_heads,
                 attention_dropout_prob):    
        super(Decoder, self).__init__()

        self.self_attention = SelfAttention(embedding_dim, num_attention_heads)
        self.residual_connection = ResidualConnection(attention_dropout_prob, embedding_dim)
        self.feed_forward_network = FeedForwardNetwork(embedding_dim, feed_forward_size)
        
    def forward(self, embedding_output, decoder_output, encoder_output, src_mask, tgt_mask, sub_layer_idx):
        if sub_layer_idx == 0:
            decoder_output = self.self_attention(embedding_output, embedding_output, tgt_mask)
        elif sub_layer_idx > 0:
            decoder_output = self.self_attention(decoder_output, decoder_output, tgt_mask)

        decoder_output = self.residual_connection(embedding_output, decoder_output)

        decoder_output = self.self_attention(decoder_output, encoder_output, src_mask)
        decoder_output = self.residual_connection(embedding_output, decoder_output)

        decoder_output = self.feed_forward_network(decoder_output)
        decoder_output = self.residual_connection(embedding_output, decoder_output)
        
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
     
class TransformerForTranslation(nn.Module):
     
    def __init__(self, batch_size, max_seq_length, vocab_size,
                 embedding_dim, sinusoidal_wave,
                 n_sub_layer, feed_forward_size, num_attention_heads,
                 attention_dropout_prob):    
        super(TransformerForTranslation, self).__init__()

        self.embeddings = WordEmbedding(batch_size, max_seq_length, vocab_size,
                                        embedding_dim, sinusoidal_wave)
        self.n_sub_layer = n_sub_layer
        self.encoder = Encoder(embedding_dim, feed_forward_size, num_attention_heads,
                               attention_dropout_prob)
        self.decoder = Decoder(embedding_dim, feed_forward_size, num_attention_heads,
                               attention_dropout_prob)
        self.output_layer = TransformerOutput(embedding_dim, vocab_size)

        
    def forward(self, src_text, tgt_text, src_mask, tgt_mask):
        src_embedding_output = self.embeddings(src_text)

        for sub_layer_idx in range(0, self.n_sub_layer):
            if sub_layer_idx == 0:
                encoder_output = self.encoder(src_embedding_output, src_embedding_output, 
                                              src_mask, sub_layer_idx)
            elif sub_layer_idx > 0:
                encoder_output = self.encoder(src_embedding_output, encoder_output, 
                                              src_mask, sub_layer_idx)

        tgt_embedding_output = self.embeddings(tgt_text)
     
        for sub_layer_idx in range(0, self.n_sub_layer):
            if sub_layer_idx == 0:
                decoder_output = self.decoder(tgt_embedding_output, tgt_embedding_output, tgt_embedding_output,
                                              src_mask, tgt_mask, sub_layer_idx)
            
            elif sub_layer_idx > 0:
                decoder_output = self.decoder(tgt_embedding_output, decoder_output, encoder_output,
                                              src_mask, tgt_mask, sub_layer_idx)
 
        output = self.output_layer(decoder_output)

        return output


