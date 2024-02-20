import os
import torch
import numpy as np
import random
import time
import argparse

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    
    
def get_logger(logger_conf: dict):
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}

def get_expname(args):
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    expname = save_time + '_' + args.model
    return expname

# 입력값을 소문자로 변환
def parse_args_boolean(value):
    lower_value = value.lower()
    if lower_value == 'true':
        return True
    elif lower_value == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError(f'--해당 인자에는 [true/false] 만 입력 가능합니다: {value}')
    
def parse_args(value):
    # 사용자 정의 함수로 리스트로 파싱
    return [int(dim) for dim in value.split(',')]