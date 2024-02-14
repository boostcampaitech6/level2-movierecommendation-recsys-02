import os
import torch
# import wandb

from module.trainer import trainer
from module.args import parse_args
from module.utils import set_seed


args = parse_args()

def main():
    # os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    trainer(config=args)
    
if __name__ == "__main__":
    main()