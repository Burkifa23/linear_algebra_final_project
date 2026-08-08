import numpy as np
import pandas as pd

def compute_euclidean_distance(user_u, user_v):
    return np.linalg.norm(user_u - user_v)

def get_k_nearest_neighbours(target_user_id, ratings_df, k=5):
    target_vector = ratings_df.loc[target_user_id].to_numpy()
    distances = []
    
    for user_id in ratings_df.index:
        if user_id != target_user_id:
            peer_vector = ratings_df.loc[user_id].to_numpy()
            dist = compute_euclidean_distance(target_vector, peer_vector)
            distances.append((user_id, dist))
            
    distances.sort(key=lambda x: x[1])
    return distances[:k]

def generate_recommendations(target_user_id, neighbours, ratings_df):
    target_ratings = ratings_df.loc[target_user_id]
    neighbour_ids = [n[0] for n in neighbours]
    neighbour_ratings = ratings_df.loc[neighbour_ids]
    
    recommendations = []
    
    for movie in ratings_df.columns:
        if target_ratings[movie] == 0.0:
            scores = neighbour_ratings[movie].to_numpy()
            avg_score = np.mean(scores)
            if avg_score > 0: 
                recommendations.append((movie, avg_score))
                
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations