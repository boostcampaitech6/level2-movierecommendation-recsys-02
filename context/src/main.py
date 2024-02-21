import os
import torch
import torch.nn as nn
import wandb

import argparse
from module.trainer import trainer
from module.utils import set_seed, parse_args_boolean, logging_conf, get_logger

logger = get_logger(logger_conf=logging_conf)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="../../data/train/", type=str, help="Data path")
    parser.add_argument("--output_dir", default="outputs/", type=str, help="Submission path")
    
    parser.add_argument("--model", default="fm", type=str, help="Model name")
    parser.add_argument("--dropout_rate", default=0.01, type=float, help="Dropout rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight_decay")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
    
    parser.add_argument("--num_nag_samples", default=50, type=int, help="number of nagative samples")
    parser.add_argument("--embed_dim", default=8, type=int, help="embed feature`s dimension")
    parser.add_argument('--mlp_dims', type=parse_args, default=(30, 20, 10), help='DeepFM의 MLP Network의 차원을 조정할 수 있습니다.')
    
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    
    parser.add_argument('--wandb', type=parse_args_boolean, default=True, help='WandB 사용 여부를 설정할 수 있습니다.')
    args = parser.parse_args()

    return args


def main():
    # os.makedirs(args.model_dir, exist_ok=True)
    
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    trainer(args)
    
if __name__ == "__main__":
    main()
    