from recbole.config import Config
from recbole.quick_start import run_recbole
import torch
import argparse

def get_config_combination_list(model_name):
    '''
    txt 파일의 combination 조합을 읽어오는 함수
    '''
    comb_list = None
    model_name_lower = model_name.lower()
    with open("./configs/config_combination.txt", "r") as file:
        for line in file:
            splited_line = line.split(' ')
            if splited_line[0] == model_name_lower:
                comb_list = splited_line[1:]
    if comb_list == None:
        raise Exception("모델 config comb가 존재하지 않습니다.")
    return comb_list

if __name__ == "__main__":
    #arg parser
    parser = argparse.ArgumentParser(description='Input Model Name.')
    parser.add_argument('model', type=str)
    parser.add_argument('--dataset',default='ML',type=str)
    args = parser.parse_args()    
    
    # run
    run_recbole(dataset=args.dataset, model=args.model, config_file_list = get_config_combination_list(args.model))
    