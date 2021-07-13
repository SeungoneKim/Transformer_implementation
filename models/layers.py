import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ScaledDotProductAttention, MultiHeadAttention, FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_head, drop_prob):
        super(EncoderLayer,self).__init__()
        
        self.attention = MultiHeadAttention(model_dim, model_dim, model_dim, num_head)
        self.normalization1 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.ffn = FeedForward(model_dim, hidden_dim, drop_prob)
        self.normalization2 = nn.LayerNorm(model_dim)
        self.dropout2 = nn.Dropout(drop_prob)
        
    def forward(self, tensor, source_mask):
        residual = tensor
        tensor = self.attention(query=tensor,key=tensor,value=tensor,mask=source_mask)
        tensor = self.dropout1(self.normalization1(tensor+residual))
        
        residual = tensor
        tensor = self.ffn(tensor)
        tensor = self.dropout2(self.normalization2(tensor+residual))
        
        return tensor

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_head, drop_prob):
        super(DecoderLayer,self).__init__()
        
        self.self_attention = MultiHeadAttention(model_dim, model_dim, model_dim, num_head)
        self.normalization1 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.enc_dec_attention = MultiHeadAttention(model_dim, model_dim, model_dim, num_head)
        self.normalization2 = nn.LayerNorm(model_dim)
        self.dropout2 = nn.Dropout(drop_prob)
        
        self.ffn = FeedForward(model_dim, hidden_dim, drop_prob)
        self.normalization = nn.LayerNorm(model_dim)
        self.dropout3 = nn.Dropout(drop_prob)
        
    def forward(self, enc_tensor, dec_tensor, source_mask, target_mask):
        residual = tgt_tensor
        tensor = self.self_attention(query=dec_tensor,key=dec_tensor,value=dec_tensor,mask=target_mask)
        tensor = self.dropout1(self.normalization1(tensor+residual))
        
        if enc_tensor is not None:
            residual = tensor
            tensor = self.enc_dec_attention(query=tensor, key=enc_tensor, value=enc_tensor, mask=source_mask)
            tensor = self.dropout2(self.normalization2(tensor+residual))
        
        residual = tensor
        tensor = self.ffn(tensor)
        tensor = self.dropout3(self.normalization3(tensor+residual))
        
        return tensor