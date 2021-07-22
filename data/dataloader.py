from dataset import TransformerDataset_total

def get_dataloader(train_batch_size, val_batch_size, test_batch_size,
                enc_language, dec_language, enc_max_len, dec_max_len,
                dataset_name, dataset_type, category_type, 
                x_name, y_name, percentage=None):
    
    # e.g. 
    # total_dataset = TransformerDataset_total('en','de',256,256,'wmt14','de-en',
    #                                     'translation','en','de')
    dataset = TransformerDataset_total(enc_language, dec_language, enc_max_len, dec_max_len,
                dataset_name, dataset_type, category_type, 
                x_name, y_name, percentage=None)
    
    train_dataloader = DataLoader(dataset=dataset.getTrainData(),
                            batch_size=train_batch_size,
                            shuffle=True)

    val_dataloader = DataLoader(dataset=dataset.getValData(),
                            batch_size=val_batch_size,
                            shuffle=True)      
    
    test_dataloader = DataLoader(dataset=dataset.getTestData(),
                            batch_size=test_batch_size,
                            shuffle=True)      
    
    return train_dataloader, val_dataloader, test_dataloader