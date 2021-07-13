parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_len', type=int, default=256)
parser.add_argument('--model_dim', type=int, default=512)
parser.add_argument('--hidden_dim', type=int, default=2048)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--drop_prob',type=float, default=0.1)
parser.add_argument('--init_lr',type=float, default=1e-5)
parser.add_argument('--warm_up',type=int, default=100)
parser.add_argument('--adam_eps',type=float, default=5e-9)
parser.add_argument('--adam_beta1',type=float, default=0.9)
parser.add_argument('--adam_beta2',type=float, default=0.98)
parser.add_argument('--vocab_size',type=int, default=37000)
parser.add_argument('--weight_decay',type=float, default=5e-4)
parser.add_argument('--pad_idx',type=int, default=0)
parser.add_argument('--bos_idx',type=int, default=1)
parser.add_argument('--eos_idx',type=int, default=2)
parser.add_argument('--unk_idx',type=int, default=3)

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