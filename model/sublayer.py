import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, is_mask=False):    
        super(MultiHeadAttention, self).__init__()

        self.is_mask = is_mask
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(embedding_dim / num_attention_heads)
        self.all_head_size = self.attention_head_size * num_attention_heads

        self.query = nn.Linear(embedding_dim, self.attention_head_size)
        self.key = nn.Linear(embedding_dim, self.attention_head_size)
        self.value = nn.Linear(embedding_dim, self.attention_head_size)

        self.linear = nn.Linear(self.all_head_size, embedding_dim)


    def single_head_attention(self, query_layer, key_layer, value_layer, max_seq_length):

        dot_product_attention = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scaled_dot_product_attention = dot_product_attention / math.sqrt(key_layer.size(-1))

        if self.is_mask:
            seq_len = query_layer.size()[-2]
            scaled_dot_product_attention = self.mask(scaled_dot_product_attention, seq_len)

        attention_probablity = nn.Softmax(dim=-1)(scaled_dot_product_attention)
        attention_output = torch.matmul(attention_probablity, value_layer)

        return attention_output

    def mask(self, scaled_dot_product_attention, max_seq_length):
        masked_input = torch.triu(torch.ones(max_seq_length, max_seq_length),diagonal=1)*(-1.0e9)
        masked_input = masked_input.cuda() + scaled_dot_product_attention
        return masked_input

    def forward(self, x, y, max_seq_length):
        
        if torch.equal(x, y) == True:
            query_layer = self.query(x)
            key_layer = self.key(x)
            value_layer = self.value(x)

        elif torch.equal(x, y) != True:
            query_layer = self.query(x)
            key_layer = self.key(y)
            value_layer = self.value(y)

        multi_head_attention_output = []
        for head in range(self.num_attention_heads):
            single_head_attention_output = self.single_head_attention(query_layer, key_layer, value_layer, max_seq_length)
            multi_head_attention_output.append(single_head_attention_output)

        attention_output = torch.cat(multi_head_attention_output, dim=2)

        attention_output = self.linear(attention_output)

        return attention_output
    
class ResidualConnection(nn.Module):
    def __init__(self, attention_dropout_prob, embedding_dim):    
        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(attention_dropout_prob)
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, encoder_input, encoder_output):
        
        encoder_output = self.dropout(encoder_output)
        residual_connection = encoder_input + encoder_output 
        residual_connection = self.normalization(residual_connection)

        return residual_connection


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, feed_forward_size):    
        super(FeedForwardNetwork, self).__init__()

        self.first_linear = nn.Linear(embedding_dim, feed_forward_size)
        self.activation_function = nn.ReLU()
        self.second_linear = nn.Linear(feed_forward_size, embedding_dim)
       
    def forward(self, x):
        x = self.first_linear(x)
        x = self.activation_function(x)
        x = self.second_linear(x)

        return x  