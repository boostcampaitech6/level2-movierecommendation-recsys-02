import os
import pandas as pd
import numpy as np
import argparse
import implicit
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix


def save_recommendations_to_csv(args, model, ratings_data, sparse_user_item, user_id_to_user_map, item_id_to_item_map):
    lst = []

    for user_id in ratings_data["user_id"].unique():
        item, score = model.recommend(user_id, sparse_user_item[user_id], 10)
        user = user_id_to_user_map[user_id]

        for rec_item in item:
            item = item_id_to_item_map[rec_item]

            dic = dict()
            dic["user"] = user
            dic["item"] = item
            lst.append(dic)

    submission_df = pd.DataFrame(l)
    submission_df.to_csv(f"{args.output_dir}submission.csv", index=False)
            
                
def main():
    
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data_path = args.data_path
    
    # data load
    print("######## Data Loading ########")
    ratings_data = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))

    ratings_data['user'] = ratings_data['user'].astype("category")
    ratings_data['item'] = ratings_data['item'].astype("category")
    ratings_data['user_id'] = ratings_data['user'].cat.codes # 새로운 user_id 할당
    ratings_data['item_id'] = ratings_data['item'].cat.codes # 새로운 item_id 할당

    user_id_to_user_map = dict(enumerate(ratings_data['user'].cat.categories)) # 새로운 user_id => 기존 user
    item_id_to_item_map = dict(enumerate(ratings_data['item'].cat.categories))  # 새로운 item_id => 기존 item
    
    # generate pivot table
    df = ratings_data.pivot_table(
        ["time"], 
        index=ratings_data["user_id"],
        columns=ratings_data["item_id"], 
        aggfunc="count",
        fill_value=0
    )

    sparse_user_item = sparse.csr_matrix(df)
    
    
    # ALS Model Bulid
    
    print("######## Bulid Model ########")
    model = implicit.als.AlternatingLeastSquares(
        factors=args.factors, # The number of latent factors to compute
        regularization = args.regularization,
        iterations = args.iterations, #
        calculate_training_loss=False,
        use_gpu = True
    )
    
    # Training
    print("######## Start Training ########")
    model.fit(sparse_user_item)
    
    # Inference and submission file
    print("######## Bulid Submisison file ########")
    save_recommendations_to_csv(args, model, ratings_data, sparse_user_item, user_id_to_user_map, item_id_to_item_map)
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="../data/train/", type=str, help="Data path")
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument("--factors", default=20, type=int, help="latent factors")
    parser.add_argument("--regularization", default=0.1, type=int, help="regularizaton")
    parser.add_argument("--iterations", default=100, type=int, help="iterations")
    parser.add_argument("--output_dir", default="outputs/", type=str, help="Submission path")
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()


