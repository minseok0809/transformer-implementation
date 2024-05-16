import math
import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads):    
        super(SelfAttention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(embedding_dim / num_attention_heads)
        self.all_head_size = self.attention_head_size * num_attention_heads

        self.query = nn.Linear(embedding_dim, self.all_head_size)
        self.key = nn.Linear(embedding_dim, self.all_head_size)
        self.value = nn.Linear(embedding_dim, self.all_head_size)
        self.embedding_dim = embedding_dim

        self.linear = nn.Linear(self.all_head_size, embedding_dim)

    def multi_head_attention(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3)
        return x

    def multi_head_mask(self, x):
        x = torch.stack([x] * self.num_attention_heads, dim=1)
        return x
       
    def put_heads_together(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (self.all_head_size, )
        x = x.view(*new_x_shape)
        return x


    def forward(self, x, y, mask):
        if torch.equal(x, y) == True:
            query_layer = self.query(x)
            key_layer = self.key(x)
            value_layer = self.value(x)

        elif torch.equal(x, y) != True:
            query_layer = self.query(x)
            key_layer = self.key(y)
            value_layer = self.value(y)
            
        query_layer = self.multi_head_attention(query_layer)
        key_layer = self.multi_head_attention(key_layer)
        value_layer = self.multi_head_attention(value_layer)

        mask = self.multi_head_mask(mask)
 
        dot_product_attention = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scaled_dot_product_attention = dot_product_attention / math.sqrt(key_layer.size(-1))

        if mask is not None:
            scaled_dot_product_attention.masked_fill_(mask == 0, -1e9)

        attention_probablity = nn.Softmax(dim=-1)(scaled_dot_product_attention)
        attention_output = torch.matmul(attention_probablity, value_layer)

        attention_output = self.put_heads_together(attention_output)
        attention_output = self.linear(attention_output)

        return attention_output


class ResidualConnection(nn.Module):
    def __init__(self, attention_dropout_prob, embedding_dim):    
        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(attention_dropout_prob)
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, embedding_output, encoder_output):
        
        encoder_output = self.dropout(encoder_output)
        residual_connection = embedding_output + encoder_output 
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