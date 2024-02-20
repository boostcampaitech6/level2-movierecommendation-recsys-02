import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def inference(model, items_dict, users_dict, raw_genre_df, user_group_dfs, n_item, device):
    # 모든 유저-아이템을 인풋으로 넣어서 결과 생성 후 랭킹 (31360 x 6807)
    u_list = []
    i_list = []
    ritems_dict = {v:k for k,v in items_dict.items()}
    for u, u_items in tqdm(user_group_dfs):

        # 인코딩하기 전에 유저id 저장
        u_list.append([u]*10)

        # user incoding
        u = users_dict[u]
        u_items = set(u_items.map(lambda x : items_dict[x])) # incoding된 유저의 시청 아이템

        # user, item, genre 데이터를 인코딩하여 학습한 모델에 맞는 값으로 변환
        i_user_col = torch.tensor([u] * n_item)
        i_item_col = torch.tensor(raw_genre_df['item'].map(lambda x : items_dict[x]).values)
        i_genre_col = torch.tensor(raw_genre_df['genre'].values)
        
        x = torch.cat([i_user_col.unsqueeze(1), i_item_col.unsqueeze(1), i_genre_col.unsqueeze(1)], dim=1)
        x = x.to(device)

        model.eval()
        output_batch = model(x)
        output_batch = output_batch.cpu().detach().numpy()

        output_batch[list(u_items)] = -1    # 이미 본 아이템 제외
        result_batch = np.argsort(output_batch)[-10:][::-1] # Top 10 item_id 추출
        i_list.append(list(map(lambda x : ritems_dict[x], result_batch)))   # item decoding


    u_list = np.concatenate(u_list)
    i_list = np.concatenate(i_list)
    submit_df = pd.DataFrame(data={'user': u_list, 'item': i_list}, columns=['user', 'item'])
    return submit_df
    