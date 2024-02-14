import torch
import numpy as np
import pandas as pd


def inference(model, user_train, user_valid, max_len, make_sequence_dataset, exp_name):
    model.eval()

    NDCG = 0.0 # NDCG@10
    HIT = 0.0 # HIT@10


    users = [user for user in range(make_sequence_dataset.num_user)]
    result = []
    
    for user in users:
        seq = (user_train[user] + [make_sequence_dataset.num_item])[-max_len:]
        rated = set(user_train[user] + user_valid[user])
        item_idx = [i+1 for i in range(make_sequence_dataset.num_item)]
        with torch.no_grad():
            predictions = -model(np.array([seq]))
            predictions = predictions[0][-1][item_idx] # sampling
            predictions[[i-1 for i in rated]] = np.inf    # 여기서 이미 본 아이템 제외
            rank = predictions.argsort().argsort().cpu().numpy()
            result.append(np.where(rank < 10)[0] + 1)   # 아이템 범위는 1~6807, item
    
    submit_user = []
    submit_item = []
    for i, r in enumerate(result):
        submit_user.append([make_sequence_dataset.user2idx.index[i]] * 10)
        for _r in r:
            submit_item.append(make_sequence_dataset.item2idx.index[make_sequence_dataset.item2idx == _r].values[0])
    submit_user = np.concatenate(submit_user)
    
    submit_df = pd.DataFrame(data={'user': submit_user, 'item': submit_item}, columns=['user', 'item'])
    return submit_df

