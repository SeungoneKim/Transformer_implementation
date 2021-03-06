import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        
    def forward(self, query, key, value, mask=None):
        _1, _2, query_sequence_length, _3 = query.size()
        batch_size, num_head, key_sequence_length, size_per_head = key.size()
        
        # matmul between query and key
        query = query.view(batch_size, num_head, query_sequence_length, size_per_head)
        key =  key.view(batch_size, num_head, size_per_head, key_sequence_length)
        attention_score = torch.einsum('abcd, abde -> abce', query, key) # [batch_size, num_head, query_sl, key_sl]
        
        # scale
        attention_score = attention_score / math.sqrt(size_per_head)
        
        # applying mask(opt) : 0s are where we apply masking
        if mask is not None:
            # mask= (batch_size, 1, query_sequence_length, key_sequence_length)
            attention_score = attention_score.masked_fill(mask==0,-1e9)
        
        # applying softmax
        attention_score = F.softmax(attention_score, dim=-1)
        
        # matmul between attention_score and value
        result = attention_score@value

        return result, attention_score

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, key_dim, value_dim, num_head):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        # query_dim is same with key_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_head = num_head
        
        self.Wq = nn.Linear(model_dim, key_dim)
        self.Wk = nn.Linear(model_dim, key_dim)
        self.Wv = nn.Linear(model_dim, value_dim)
        self.attention = ScaledDotProductAttention()
        self.Wo = nn.Linear(value_dim, model_dim)
        
    def forward(self, query, key, value, mask=None):
        # linearly project queries, key and values
        prj_query = self.Wq(query)
        prj_key = self.Wk(key)
        prj_value = self.Wv(value)
        
        # split prj_query, prj_key, prj_value into multi head
        multihead_query = self.multihead_split(prj_query)
        multihead_key = self.multihead_split(prj_key)
        multihead_value = self.multihead_split(prj_value)
        
        # perform Scaled Dot Product Attention
        attention_output, attention_score = self.attention(multihead_query, multihead_key, multihead_value, mask=mask)
        
        # concat output back to 3-dimensional tensor of (batch_size, sequence_length, hidden_size)
        output = self.multihead_concat(attention_output)
        #print(output.size())
        output = self.Wo(output)
        
        return output
    
    def multihead_split(self, tensor):
        batch_size, sequence_length, hidden_size = tensor.size()
        
        size_per_head = hidden_size // self.num_head
        # (batch_size, num_head, sequence_length, size_per_head)
        return tensor.view(batch_size, self.num_head, sequence_length, size_per_head)
    
    def multihead_concat(self, tensor):
        #print(tensor.size())
        batch_size, num_head, sequence_length, size_per_head = tensor.size()
        
        hidden_size = num_head * size_per_head
        return tensor.view(batch_size,sequence_length,hidden_size)

class FeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, drop_prob):
        super(FeedForward,self).__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        
        self.linearlayer1 = nn.Linear(model_dim, hidden_dim)
        self.linearlayer2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, tensor):
        tensor = self.dropout(self.relu(self.linearlayer1(tensor)))
        return self.linearlayer2(tensor)