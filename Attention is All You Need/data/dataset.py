from datasets import load_dataset, list_datasets # huggingface library
from data.tokenizer import Tokenizer
from torch.utils.data import Dataset

class TransformerDataset(Dataset):
    def __init__(self, enc_language, dec_language, enc_max_len, dec_max_len, 
                dataset_name, dataset_type, split_type, category_type, 
                x_name, y_name, percentage=None):
        
        if dataset_name not in list_datasets():
            assert('Not available in HuggingFace datasets')
        
        if percentage is None:
            data = load_dataset(dataset_name, dataset_type, split=split_type)
        else:
            data = load_dataset(dataset_name, dataset_type, split=f'{split_type}[:{percentage}%]')
        
        self.data = data[category_type]
        self.x_name = x_name
        self.y_name = y_name
        self.data_len = len(data) # number of data
        self.enc_language = enc_language
        self.dec_language = dec_language
        self.encoder_max_len = enc_max_len # max sequence length
        self.decoder_max_len = dec_max_len # max sequence length
        self.enc_tokenizer = Tokenizer(enc_language,enc_max_len)
        self.dec_tokenizer = Tokenizer(dec_language,dec_max_len)
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        encoded_datax = self.enc_tokenizer.encode(self.data[index][self.enc_language])
        encoded_datay = self.dec_tokenizer.encode(self.data[index][self.dec_language])
        
        # batch of data that the dataloader will provide during training
        batch ={}
        batch['encoder_input_ids'] = encoded_datax.input_ids
        batch['encoder_attention_mask'] = encoded_datax.attention_mask # will be generated in model as well
        batch['decoder_input_ids'] = encoded_datay.input_ids
        batch['labels'] = encoded_datay.input_ids.clone()
        batch['decoder_attention_mask'] = encoded_datay.attention_mask # will be generated in model as well
        
        
        # check if length is fixed to max_len
        #assert all([x.size() == self.encoder_max_len for x in batch['encoder_input_ids']])
        #assert all([x.size() == self.decoder_max_len for x in batch['decoder_input_ids']])


        return batch


class TransformerDataset_total():
    def __init__(self, enc_language, dec_language, enc_max_len, dec_max_len,
                dataset_name, dataset_type, category_type, 
                x_name, y_name, percentage=None):
        self.traindata = TransformerDataset(enc_language, dec_language, enc_max_len, dec_max_len,
                                    dataset_name, dataset_type, 'train', 
                                    category_type, x_name, y_name, percentage)
        self.valdata = TransformerDataset(enc_language, dec_language, enc_max_len, dec_max_len,
                                    dataset_name, dataset_type, 'validation', 
                                    category_type, x_name, y_name, percentage)
        self.testdata = TransformerDataset(enc_language, dec_language, enc_max_len, dec_max_len,
                                    dataset_name, dataset_type, 'test', 
                                    category_type, x_name, y_name, percentage)
    
    def getTrainData(self):
        return self.traindata
    
    def getValData(self):
        return self.valdata
    
    def getTestData(self):
        return self.testdata