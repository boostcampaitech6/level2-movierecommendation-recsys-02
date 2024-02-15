import os
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class MakeSequenceDataSet():
    """
    SequenceData 생성
    """
    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(os.path.join(self.config.data_path, 'train_ratings.csv'))

        self.item_encoder, self.item_decoder, self.item_ids = self.generate_encoder_decoder('item')
        self.user_encoder, self.user_decoder, self.user_ids = self.generate_encoder_decoder('user')
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df['item_idx'] = self.df['item'].apply(lambda x : self.item_encoder[x] + 1)
        self.df['user_idx'] = self.df['user'].apply(lambda x : self.user_encoder[x])
        self.df = self.df.sort_values(['user_idx', 'time']) # 시간에 따라 정렬
        self.user_train, self.user_valid = self.generate_sequence_data()
        
        self.item2idx = pd.Series(data=np.arange(len(self.item_ids))+1, index=self.item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
        self.user2idx = pd.Series(data=np.arange(len(self.user_ids)), index=self.user_ids) # user re-indexing (0~num_user-1)

    def generate_encoder_decoder(self, col : str) -> dict:
        """
        encoder, decoder 생성

        Args:
            col (str): 생성할 columns 명
        Returns:
            dict: 생성된 user encoder, decoder
        """

        encoder = {}
        decoder = {}
        ids = self.df[col].unique()

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder, ids

    def generate_sequence_data(self) -> dict:
        """
        sequence_data 생성

        Returns:
            dict: train user sequence / valid user sequence
        """
        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        group_df = self.df.groupby('user_idx')
        for user, item in group_df:
            users[user].extend(item['item_idx'].tolist())

        for user in users:
            user_train[user] = users[user][:-1]
            user_valid[user] = [users[user][-1]] # 마지막 아이템을 예측

        return user_train, user_valid

    def get_train_valid_data(self):
        return self.user_train, self.user_valid

    
    
class BERTRecDataSet(Dataset):
    def __init__(self, user_train, max_len, num_user, num_item, mask_prob):
        self.user_train = user_train
        self.max_len = max_len
        self.num_user = num_user
        self.num_item = num_item
        self.mask_prob = mask_prob
        self._all_items = set([i for i in range(1, self.num_item + 1)])

    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user):

        user_seq = self.user_train[user]
        tokens = []
        labels = []
        for s in user_seq[-self.max_len:]:
            prob = np.random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    # masking
                    tokens.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    # noise
                    tokens.extend(self.random_neg_sampling(rated_item = user_seq, num_item_sample = 1))  # item random sampling
                else:
                    tokens.append(s)
                labels.append(s) # 학습에 사용 O
            else:
                tokens.append(s)
                labels.append(0) # 학습에 사용 X

        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def random_neg_sampling(self, rated_item : list, num_item_sample : int):
        nge_samples = random.sample(list(self._all_items - set(rated_item)), num_item_sample)
        return nge_samples
    

class SASRecDataSet(Dataset):
    def __init__(self, user_train, max_len, num_user, num_item):
        self.user_train = user_train
        self.max_len = max_len
        self.num_user = num_user
        self.num_item = num_item
        self._all_items = set([i for i in range(1, self.num_item + 1)])

    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user):

        user_seq = self.user_train[user]
        user_seq_len = len(user_seq)

        seq = user_seq[-(user_seq_len) : -1]
        seq = seq[-self.max_len :]

        pos = user_seq[-(user_seq_len - 1) : ]
        pos = pos[-self.max_len :]

        neg = random.sample(list(self._all_items - set(user_seq)), len(pos))

        seq = [0] * (self.max_len - len(seq)) + seq
        pos = [0] * (self.max_len - len(pos)) + pos
        neg = [0] * (self.max_len - len(neg)) + neg

        return np.array(seq, dtype=np.int32), np.array(pos, dtype=np.int32), np.array(neg, dtype=np.int32)

    def random_neg_sampling(self, rated_item : list, num_item_sample : int):
        nge_samples = random.sample(list(self._all_items - set(rated_item)), num_item_sample)
        return nge_samples