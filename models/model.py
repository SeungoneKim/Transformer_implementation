import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import TokenEmbedding, PositionalEncoding, TransformerEmbedding
from attention import ScaledDotProductAttention, MultiHeadAttention, FeedForward
from layers import EncoderLayer, DecoderLayer

class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, src_max_len, 
                 model_dim, key_dim, value_dim, hidden_dim, 
                 num_head, num_layer, drop_prob, device):
        super(Encoder,self).__init__()
        self.embedding = TransformerEmbedding(enc_vocab_size, model_dim, src_max_len, drop_prob, device)
        
        self.layers = nn.ModuleList([EncoderLayer(model_dim, key_dim, value_dim, 
                                                  hidden_dim, num_head, 
                                                  drop_prob) for _ in range(num_layer)])
        
    def forward(self, tensor, src_mask):
        input_emb = self.embedding(tensor)
        encoder_output = input_emb
        
        for layer in self.layers:
            encoder_output = layer(encoder_output, src_mask)
        
        return encoder_output

class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, tgt_max_len,
                model_dim, key_dim, value_dim, hidden_dim, 
                 num_head, num_layer, drop_prob, device):
        super(Decoder,self).__init__()
        self.embedding = TransformerEmbedding(dec_vocab_size, model_dim, tgt_max_len, drop_prob, device)
        
        self.layers = nn.ModuleList([DecoderLayer(model_dim, key_dim, value_dim, 
                                                  hidden_dim, num_head,
                                                 drop_prob) for _ in range(num_layer)])

    def forward(self, tensor, encoder_output, src_mask, tgt_mask):
        tgt_emb = self.embedding(tensor)
        decoder_output = tgt_emb
        
        for layer in self.layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
            
        return decoder_output

class LangaugeModelingHead(nn.Module):
    def __init__(self, dec_vocab_size, model_dim):
        super(LangaugeModelingHead,self).__init__()
        self.linearlayer = nn.Linear(model_dim, dec_vocab_size)
        
    def forward(self, decoder_output):
        return self.linearlayer(decoder_output)

class TransformersModel(nn.Module):
    def __init__(self, src_pad_idx, tgt_pad_idx, tgt_sos_idx, 
                enc_vocab_size, dec_vocab_size, 
                model_dim, key_dim, value_dim, hidden_dim, 
                num_head, num_layer, max_len, drop_prob, device):
        super(TransformersModel, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.device = device
        
        self.Encoder = Encoder(enc_vocab_size, max_len, 
                 model_dim, key_dim, value_dim, hidden_dim, num_head, num_layer, drop_prob, device)
        self.Decoder = Decoder(dec_vocab_size, max_len,
                 model_dim, key_dim, value_dim, hidden_dim, num_head, num_layer, drop_prob, device)
        self.LMHead = LangaugeModelingHead(model_dim, dec_vocab_size)
        
    def forward(self, src_tensor, tgt_tensor):
        enc_mask = self.generate_padding_mask(src_tensor, src_tensor, "src","src")
        
        enc_dec_mask = self.generate_padding_mask(tgt_tensor, src_tensor, "src","tgt")
        
        dec_mask = self.generate_padding_mask(tgt_tensor, tgt_tensor,"tgt","tgt") * \
                    self.generate_triangular_mask(tgt_tensor, tgt_tensor)
        
        encoder_output = self.Encoder(src_tensor, enc_mask)
        
        decoder_output = self.Decoder(tgt_tensor, encoder_output, enc_dec_mask, dec_mask)
    
        output = self.LMHead(decoder_output)
        
        return output
    
    # applying mask(opt) : 0s are where we apply masking
    # pad_type =["src". "tgt"]
    def generate_padding_mask(self, query, key, query_pad_type=None, key_pad_type=None):
        # query = (batch_size, query_length)
        # key = (batch_size, key_length)
        query_length = query.size(1)
        key_length = key.size(1)
        
        # decide query_pad_idx based on query_pad_type
        if query_pad_type == "src":
            query_pad_idx = self.src_pad_idx
        elif query_pad_type == "tgt":
            query_pad_idx = self.tgt_pad_idx
        else:
            assert "query_pad_type should be either src or tgt"
        
        # decide key_pad_idx based on key_pad_type
        if key_pad_type == "src":
            key_pad_idx = self.src_pad_idx
        elif key_pad_type == "tgt":
            key_pad_idx = self.tgt_pad_idx
        else:
            assert "key_pad_type should be either src or tgt"
        
        # convert query and key into 4-dimensional tensor
        # query = (batch_size, 1, query_length, 1) -> (batch_size, 1, query_length, key_length)
        # key = (batch_size, 1, 1, key_length) -> (batch_size, 1, query_length, key_length)
        query = query.ne(query_pad_idx).unsqueeze(1).unsqueeze(3)
        query = query.repeat(1,1,1,key_length)
        key = key.ne(key_pad_idx).unsqueeze(1).unsqueeze(2)
        key = key.repeat(1,1,query_length,1)
        
        # create padding mask with key and query
        mask = key & query
        
        return mask
    
    # applying mask(opt) : 0s are where we apply masking
    def generate_triangular_mask(self, query, key):
        # query = (batch_size, query_length)
        # key = (batch_size, key_length)
        query_length = query.size(1)
        key_length = key.size(1)
        
        # create triangular mask
        mask = torch.tril(torch.ones(query_length,key_length)).type(torch.BoolTensor).to(device)
        
        return mask
"""
model = build_model(src_pad_idx=0,tgt_pad_idx=0,tgt_sos_idx=1,
                    enc_vocab_size=37000,dec_vocab_size=37000,
                    model_dim=512, key_dim=64, value_dim=64, hidden_dim=2048,
                    num_head=8,num_layer=6,
                    max_len=256,drop_prob=0.1)

params = list(model.parameters())
print("The number of parameters:",sum([p.numel() for p in model.parameters() if p.requires_grad]), "elements")

The number of parameters: 88597000 elements
"""
def build_model(src_pad_idx, tgt_pad_idx, tgt_sos_idx, 
                enc_vocab_size, dec_vocab_size, 
                model_dim, key_dim, value_dim, hidden_dim, 
                num_head, num_layer, max_len, drop_prob):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TransformersModel(src_pad_idx, tgt_pad_idx, tgt_sos_idx, 
                enc_vocab_size, dec_vocab_size, 
                model_dim, key_dim, value_dim, hidden_dim, 
                num_head, num_layer, max_len, drop_prob,device)
    
    return model.cuda() if torch.cuda.is_available() else model