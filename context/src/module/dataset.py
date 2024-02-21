import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

class RatingDataset(Dataset):
    def __init__(self, input_tensor, target_tensor):
        self.input_tensor = input_tensor.long()
        self.target_tensor = target_tensor.long()

    def __getitem__(self, index):
        return self.input_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.target_tensor.size(0)
    

class MakeDataset:
    def __init__(self, rating_data_path, genre_data_path, num_negative=50):
        self.rating_data_path = rating_data_path
        self.genre_data_path = genre_data_path
        self.users = None
        self.items = None
        self.genres = None
        self.data = None
        self.num_negative = num_negative
        self.user_dict = None
        self.item_dict = None
        self.genre_dict = None
        self.raw_rating_df = None
        self.user_group_dfs = None
        self.raw_genre_df = None

    def load_data(self):
        # Load rating data
        self.raw_rating_df = pd.read_csv(self.rating_data_path)
        self.raw_rating_df['rating'] = 1.0  # implicit feedback
        self.raw_rating_df.drop(['time'], axis=1, inplace=True)

        # Load genre data
        self.raw_genre_df = pd.read_csv(self.genre_data_path, sep='\t')
        self.raw_genre_df = self.raw_genre_df.drop_duplicates(subset=['item'])

        # Map genre to ids
        genre_dict = {genre: i for i, genre in enumerate(set(self.raw_genre_df['genre']))}
        self.raw_genre_df['genre'] = self.raw_genre_df['genre'].map(lambda x: genre_dict[x])

        self.users = set(self.raw_rating_df['user'])
        self.items = set(self.raw_rating_df['item'])
        self.genres = set(self.raw_genre_df['genre'])

        self.data = self.raw_rating_df.merge(self.raw_genre_df, on='item', how='inner')

    def create_negative_instances(self):
        self.user_group_dfs = list(self.data.groupby('user')['item'])
        neg_instances = []

        for u, u_items in tqdm(self.user_group_dfs):
            u_items = set(u_items)
            i_user_neg_item = np.random.choice(list(self.items - u_items), self.num_negative, replace=False)
            neg_instances.extend([(u, item, 0) for item in i_user_neg_item])

        neg_df = pd.DataFrame(neg_instances, columns=['user', 'item', 'rating'])
        self.data = pd.concat([self.data, neg_df], ignore_index=True)

    def index_mapping(self):
        self.users = sorted(list(self.users))
        self.items = sorted(list(self.items))
        self.genres = sorted(list(self.genres))

        self.user_dict = {user: i for i, user in enumerate(self.users)}
        self.item_dict = {item: i for i, item in enumerate(self.items)}
        self.genre_dict = {genre: i for i, genre in enumerate(self.genres)}

        self.data['user'] = self.data['user'].map(self.user_dict)
        self.data['item'] = self.data['item'].map(self.item_dict)
        self.data['genre'] = self.data['genre'].map(self.genre_dict)

    def preprocess(self):
        self.load_data()
        self.create_negative_instances()
        self.index_mapping()
        return self.raw_rating_df, self.user_group_dfs, self.raw_genre_df

    def get_statistics(self):
        n_data = len(self.data)
        n_user = len(self.users)
        n_item = len(self.items)
        n_genre = len(self.genres)

        return n_data, n_user, n_item, n_genre, self.user_dict, self.item_dict, self.genre_dict