from recbole.config import Config
from recbole.quick_start import run_recbole
import torch
print(torch.cuda.is_available())
config_file_SASREC=[
    './configs/Training/train.yaml',
    './configs/Model/Multi_VAE.yaml',
    './configs/EVAL/eval.yaml',
    './configs/Dataset/ML_sequence.yaml',
    './configs/ENV/env.yaml',
    ]
config_file_list2=['./configs/EVAL/eval2.yaml','./configs/ENV/env.yaml']
config = Config(config_file_list=config_file_SASREC)

run_recbole(dataset=config['dataset'], model=config['model'], config_file_list = config_file_list2)
