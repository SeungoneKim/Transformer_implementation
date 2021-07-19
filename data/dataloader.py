def get_dataloader(dataset_name, category_name, batch_size, percentage=None):
    dataset = TrainDataset(dataset_name,category_name,percentage)
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True)
    return dataloader