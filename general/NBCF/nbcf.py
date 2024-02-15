import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import argparse

def compute_cos_similarity(v1, v2):
    norm1 = np.sqrt(np.sum(np.square(v1)))
    norm2 = np.sqrt(np.sum(np.square(v2)))
    dot = np.dot(v1, v2)
    return dot / (norm1 * norm2)

def movie_recommend(adj_matrix, user_id_to_user_map, item_id_to_item_map):
    recommendations = {}
    for my_id, my_vector in tqdm(enumerate(adj_matrix), total=adj_matrix.shape[0]):
        best_match, best_match_vector = -1, []
        for user_id, user_vector in enumerate(adj_matrix):
            if my_id != user_id:
                cos_similarity = compute_cos_similarity(my_vector, user_vector)
                if cos_similarity > best_match:
                    best_match = cos_similarity
                    best_match_vector = user_vector

        recommend_list = np.where((my_vector < 1.) & (best_match_vector > 0.))[0][:10]
        user = user_id_to_user_map[my_id]
        recommendations[user] = [item_id_to_item_map[item_id] for item_id in recommend_list]
        
    return recommendations

def save_recommendations_to_csv(recommendations, args):
    output_path = os.path.join(args.output_dir, "submission.csv")
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['user', 'item']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for user, items in recommendations.items():
            for item in items:
                writer.writerow({'user': user, 'item': item})
                         
                
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
    adj_matrix = df.to_numpy()

    # recommendation
    print("######## Start Recommendation ########")
    recommendations = movie_recommend(adj_matrix, user_id_to_user_map, item_id_to_item_map)
    
    # submission file
    print("######## Bulid Submisison file ########")
    save_recommendations_to_csv(recommendations, args)
    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="../../data/train/", type=str, help="Data path")
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument("--output_dir", default="outputs/", type=str, help="Submission path")
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()


