from datasets import load_dataset, list_datasets

class PretrainDataset(Dataset):
    def __init__(self, dataset_name, category_name, percentage=None):
        
        if dataset_name not in list_datasets():
            assert('Not available in HuggingFace datasets')
        
        if percentage is None:
            data = load_dataset(dataset_name, split='train')
        else:
            data = load_dataset(dataset_name, split=f'train[:{percentage}%]')
        
        self.data = data[category_name]
        self.data_len = len(data)
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        return self.data[index]

class FineTuneDataset_train(Dataset):
    def __init__(self, dataset_name, x_name, y_name, percentage=None):
        
        if dataset_name not in list_datasets():
            assert('Not available in HuggingFace datasets')
        
        if percentage is None:
            train_data = load_dataset(dataset_name, split='train')
        else:
            train_data = load_dataset(dataset_name, split=f'train[:{percentage}%]')
        
        self.data_len = len(train_data)
        
        self.trainX = train_data[x_name]
        self.trainY = train_data[y_name]
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        return self.trainX[index], self.trainY[index]

class FineTuneDataset_val(Dataset):
    def __init__(self, dataset_name, x_name, y_name, percentage=None):
        
        if dataset_name not in list_datasets():
            assert('Not available in HuggingFace datasets')
        
        if percentage is None:
            val_data = load_dataset(dataset_name, split='validation')
        else:
            val_data = load_dataset(dataset_name, split=f'validation[:{percentage}%]')
        
        self.data_len = len(val_data)
        
        self.valX = val_data[x_name]
        self.valY = val_data[y_name]
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        return self.valX[index], self.valY[index]

class FineTuneDataset_test(Dataset):
    def __init__(self, dataset_name, x_name, y_name, percentage=None):
        
        if dataset_name not in list_datasets():
            assert('Not available in HuggingFace datasets')
        
        if percentage is None:
            test_data = load_dataset(dataset_name, split='test')
        else:
            test_data = load_dataset(dataset_name, split=f'test[:{percentage}%]')
        
        self.data_len = len(test_data)
        
        self.testX = test_data[x_name]
        self.testY = test_data[y_name]
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        return self.testX[index], self.testY[index]

class FineTuneDataset_total():
    def __init__(self, dataset_name, x_name, y_name, percentage=None):
        self.traindata = FineTuneDataset_train(dataset_name, x_name, y_name, percentage)
        self.valdata = FineTuneDataset_val(dataset_name, x_name, y_name, percentage)
        self.testdata = FineTuneDataset_test(dataset_name, x_name, y_name, percentage)
    
    def getTrainData(self):
        return self.traindata
    
    def getValData(self):
        return self.valdata
    
    def getTestData(self):
        return self.testdata