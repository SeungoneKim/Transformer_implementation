import torch
import torch.nn as nn
import torch.optim as optim
from datasets import list_metrics, load_metric
import sklearn.metrics as skm

sklearn_metrics_list = ['accuracy_score','f1_score','precision_score','recall_score',
                        'roc_auc_score','mean_squared_error','mean_absolute_error']

lossfn_list = ['BCELoss','CrossEntropyLoss','KLDivLoss','BCEWithLogitsLoss',
                'L1Loss','MSELoss','NLLLoss']

def load_metric(metric_type):
    metric= None
    if metric_type in list_metrics():
        metric= load_metric(metric_type)
    elif metric_type in sklearn_metrics_list:
        if metric_type == 'accuracy_score':
            metric = skm.accuracy_score()
        elif metric_type == 'f1_score':
            metric = skm.f1_score()
        elif metric_type == 'precision_score':
            metric = skm.precision_score()
        elif metric_type == 'recall_score':
            metric = skm.recall_score()
        elif metric_type == 'roc_auc_score':
            metric = skm.roc_auc_score()
        elif metric_type == 'mean_squared_error':
            metric = skm.mean_squared_error()
        elif metric_type == 'mean_absolute_error':
            metric = skm.mean_absolute_error()
    else:
        assert "You typed a metric that doesn't exist or is not supported"

    return metric

def load_optimizer(model, learning_rate, weight_decay):
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def load_scheduler(optimizer, decay_epoch):
    return optim.lr_scheduler.StepLR(optimizer, decay_epoch)

def load_lossfn(lossfn_type):
    lossfn= None
    if lossfn_type in lossfn_list:
        if lossfn_type == 'BCELoss':
            lossfn = nn.BCELoss()
        elif lossfn_type == 'CrossEntropyLoss':
            lossfn = nn.CrossEntropyLoss()
        elif lossfn_type == 'KLDivLoss':
            lossfn = nn.KLDivLoss()
        elif lossfn_type == 'BCEWithLogitsLoss':
            lossfn = nn.BCEWithLogitsLoss()
        elif lossfn_type == 'L1Loss':
            lossfn = nn.L1Loss()
        elif lossfn_type == 'MSELoss':
            lossfn = nn.MSELoss()
        elif lossfn_type == 'NLLLoss':
            lossfn = nn.NLLLoss()
    
    return lossfn
        



