def get_pretrain_dataloader(dataset_name, category_name, batch_size, percentage=None):
    dataset = PretrainDataset(dataset_name,category_name,percentage)
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True)
    return dataloader

def get_finetune_dataloader(dataset_name, x_name, y_name, batch_size, percentage=None):
    dataset = FineTuneDataset_total(dataset_name, x_name, y_name, percentage)

    train_dataloader = DataLoader(dataset=dataset.getTrainData(),
                                 batch_size=batch_size,
                                 shuffle=True)
    val_dataloader = DataLoader(dataset=dataset.getValData(),
                               batch_size=batch_size,
                               shuffle=True)
    test_dataloader = DataLoader(dataset=dataset.getTestData(),
                                batch_size=batch_size,
                                shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader