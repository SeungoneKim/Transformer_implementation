import os
import sys
import argparse

parser = argparse.ArgumentParser()

# gpu
parser.add_argument('--device', type=str, default='cuda:0,1,2,3')
# hyperparameters
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model_dim', type=int, default=512)
parser.add_argument('--key_dim',type=int, default = 64)
parser.add_argument('--value_dim',type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=2048)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--drop_prob',type=float, default=0.1)
parser.add_argument('--init_lr',type=float, default=1e-5)
parser.add_argument('--warm_up',type=int, default=100)
parser.add_argument('--adam_eps',type=float, default=5e-9)
parser.add_argument('--adam_beta1',type=float, default=0.9)
parser.add_argument('--adam_beta2',type=float, default=0.98)
parser.add_argument('--enc_max_len', type=int, default=512)
parser.add_argument('--dec_max_len',type=int, default=256)
parser.add_argument('--enc_vocab_size',type=int, default=28996)
parser.add_argument('--dec_vocab_size',type=int, default=30000)
parser.add_argument('--weight_decay',type=float, default=5e-4)
parser.add_argument('--decay_epoch',type=int, default=100)
# tokenizer
### encoder(en)
parser.add_argument('--enc_pad_idx',type=int, default=0) # [PAD]
parser.add_argument('--enc_bos_idx',type=int, default=1) # <s>
parser.add_argument('--enc_eos_idx',type=int, default=2) # </s>
parser.add_argument('--enc_unk_idx',type=int, default=100) # [UNK]
parser.add_argument('--enc_cls_idx',type=int, default=101)  # [CLS]
parser.add_argument('--enc_sep_idx',type=int, default=102)  # [SEP]
parser.add_argument('--enc_mask_idx',type=int,default=103) # [MASK]
### decoder(de)
parser.add_argument('--dec_pad_idx',type=int, default=0) # [PAD]
parser.add_argument('--dec_bos_idx',type=int, default=1) # <s>
parser.add_argument('--dec_eos_idx',type=int, default=2) # </s>
parser.add_argument('--dec_unk_idx',type=int, default=3) # [UNK]
parser.add_argument('--dec_cls_idx',type=int, default=)  # [CLS]
parser.add_argument('--dec_sep_idx',type=int, default=)  # [SEP]
parser.add_argument('--dec_mask_idx',type=int,default=4) # [MASK]
# trainer
parser.add_argument('--pretrain_or_finetune',type=str, default='finetune')
parser.add_argument('--metric',type=str, default='accuracy_score')
parser.add_argument('--lossfn',type=str, default= 'CrossEntropyLoss')
# dataloader
parser.add_argument('--dataset_name',type=str, default=None)
parser.add_argument('--category_name',type=str, default=None)
parser.add_argument('--x_name',type=str, default=None)
parser.add_argument('--y_name',type=str, default=None)
parser.add_argument('--percentage',type=int, default=100)
# set path
parser.add_argument('--cur_path',type=str, default=os.getcwd())
parser.add_argument('--weight_path',type=str,default = os.path.join(os.getcwd(),'weights'))

def get_config():
    return parser

def set_random_fixed(seed_num):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    np.random.seed(seed_num)