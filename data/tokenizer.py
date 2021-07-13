import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer

class Tokenizer(nn.Module):
    def __init__(self):
        super(Tokenizer,self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.unk_token = self.tokenizer.cls_token
        self.pad_token = self.tokenizer.pad_token
        self.mask_token = self.tokenizer.mask_token
        self.vocab_size = self.tokenizer.vocab_size
    
    def encode(self, batch_sentences):
        return self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")

    def encode_multiple(self, batch_sentences1, batch_sentences2):
        return self.tokenizer(batch_sentences1, batch_sentences2, padding=True, truncation=True, return_tensors="pt")
    
    def encode_into_input_ids(self, batch_sentence):
        return self.encode(batch_sentences)['input_ids']
    
    def encode_multiple_into_input_ids(self, batch_sentences1, batch_sentences2):
        return self.encode_multiple(batch_sentences1, batch_sentences2)['input_ids']
    
    def encode_into_token_type_ids(self, batch_sentence):
        return self.encode(batch_sentences)['token_type_ids']
    
    def encode_multiple_into_token_type_ids(self, batch_sentences1, batch_sentences2):
        return self.encode_multiple(batch_sentences1, batch_sentences2)['token_type_ids']
    
    def encode_into_attention_mask(self, batch_sentence):
        return self.encode(batch_sentences)['attention_mask']
    
    def encode_multiple_into_attention_mask(self, batch_sentences1, batch_sentences2):
        return self.encode_multiple(batch_sentences1, batch_sentences2)['attention_mask']
    
    def decode(self, encoded_inputs):
        decoded_output=[]
        for ids in encoded_inputs["input_ids"]:
            decoded_output.append( [self.tokenizer.decode(ids)] )
        return decoded_output
    
    def decode_input_ids(self, encoded_input_ids):
        decoded_output=[]
        for ids in encoded_inputs:
            decoded_output.append( [self.tokenizer.decode(ids)] )
        return decoded_output
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def add_word(self, word):
        self.tokenizer.add_tokens([word])
        self.vocab_size = len(self.tokenizer)
        return self.vocab_size
    
    def add_words(self, list_of_words):
        self.tokenizer.add_tokens(list_of_words)
        self.vocab_size = len(self.tokenizer)
        return self.vocab_size