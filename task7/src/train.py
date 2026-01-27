import pandas as pd
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict

def get_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""
    user_est_true = defaultdict(list)
    for uid, true_r, est in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here treat it as 0.
        precisions[uid] = n_rel_and_rec_k / k

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here treat it as 0.
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

def train_content_based(train_df, test_df, movies_df):
    """
    Train a simple Content-Based model using Movie Genres.
    Strategy: 
    1. Build User Profile = Weighted Average of Genres of movies they liked.
    2. Suggest movies similar to User Profile (or just similar to movies they rated highly).
    
    Simplified Implementation for Rating Prediction:
    predict(user, item) = Weighted avg of user's ratings for similar items.
    Similarity based on TF-IDF of genres.
    """
    print("Training Content-Based Model...")
    
    # 1. Create TF-IDF matrix for movies
    # Ensure genres are string
    movies_df['genres'] = movies_df['genres'].fillna('')
    tfidf = TfidfVectorizer(token_pattern='[a-zA-Z0-9\-]+')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    
    # Map movieId to index
    movie_idx_map = pd.Series(movies_df.index, index=movies_df['movieId']).to_dict()
    
    # Calculate Cosine Similarity Matrix (Too large for full dataset? Small dataset is fine: ~9000 movies)
    # 9000x9000 is approx 81M floats -> ~324MB. Feasible.
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Predict for Test Set
    preds = []
    # Optimization: Pre-compute user histories
    user_history = train_df.groupby('userId')[['movieId', 'rating']].apply(lambda x: list(zip(x['movieId'], x['rating']))).to_dict()
    
    # Limit test set for speed if needed, but 20k is fine
    test_subset = test_df #.sample(1000) 
    
    predictions = [] # List of (uid, true_r, est)
    
    for idx, row in test_subset.iterrows():
        user = row['userId']
        movie = row['movieId']
        true_rating = row['rating']
        
        if movie not in movie_idx_map:
            est = 2.5 # Default
        else:
            movie_idx = movie_idx_map[movie]
            sim_scores = cosine_sim[movie_idx]
            
            # Weighted average of ratings from this user for OTHER movies, weighted by similarity
            weighted_sum = 0
            sim_sum = 0
            
            if user in user_history:
                for past_movie, past_rating in user_history[user]:
                    if past_movie in movie_idx_map:
                        past_idx = movie_idx_map[past_movie]
                        sim = sim_scores[past_idx]
                        if sim > 0:
                            weighted_sum += sim * past_rating
                            sim_sum += sim
            
            if sim_sum > 0:
                est = weighted_sum / sim_sum
            else:
                est = train_df['rating'].mean() # Global mean fallback
                
        preds.append(est)
        predictions.append((user, true_rating, est))
        
    rmse, mae = get_metrics(test_subset['rating'], preds)
    
    k_values = [5, 10, 15, 20]
    metrics_k = {}
    
    for k in k_values:
        precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=3.5)
        mean_prec = sum(prec for prec in precisions.values()) / len(precisions)
        mean_rec = sum(rec for rec in recalls.values()) / len(recalls)
        metrics_k[f"Precision@{k}"] = mean_prec
        metrics_k[f"Recall@{k}"] = mean_rec
    
    results = {"RMSE": rmse, "MAE": mae, "Predictions": predictions}
    results.update(metrics_k)
    return results

def train_collaborative_filtering(train_df, test_df):
    """
    Train Collaborative Filtering using SVD (Matrix Factorization) with Mean Centering.
    """
    print("Training Collaborative Filtering (SVD)...")
    
    # Create User-Item Matrix
    user_item_matrix = train_df.pivot(index='userId', columns='movieId', values='rating')
    
    # Calculate user means
    user_means = user_item_matrix.mean(axis=1)
    
    # Center the matrix (subtract mean), fill missing with 0
    matrix_centered = user_item_matrix.sub(user_means, axis=0).fillna(0)
    
    # Decompose
    X = matrix_centered.values
    # Use fewer components for small data to avoid overfitting 0s
    SVD = TruncatedSVD(n_components=20, random_state=42)
    matrix_reduced = SVD.fit_transform(X)
    matrix_reconstructed = SVD.inverse_transform(matrix_reduced)
    
    # Convert back to DataFrame
    matrix_reconstructed_df = pd.DataFrame(matrix_reconstructed, index=matrix_centered.index, columns=matrix_centered.columns)
    
    preds = []
    predictions = []
    
    global_mean = train_df['rating'].mean()
    
    for idx, row in test_df.iterrows():
        user = row['userId']
        movie = row['movieId']
        true_rating = row['rating']
        
        est = global_mean # Fallback
        
        if user in matrix_reconstructed_df.index and movie in matrix_reconstructed_df.columns:
            deviation = matrix_reconstructed_df.loc[user, movie]
            u_mean = user_means.loc[user]
            est = u_mean + deviation
            
        # Clip
        est = min(5.0, max(0.5, est))
        
        preds.append(est)
        predictions.append((user, true_rating, est))
        
    rmse, mae = get_metrics(test_df['rating'], preds)
    
    k_values = [5, 10, 15, 20]
    metrics_k = {}
    
    for k in k_values:
        precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=3.5)
        mean_prec = sum(prec for prec in precisions.values()) / len(precisions)
        mean_rec = sum(rec for rec in recalls.values()) / len(recalls)
        metrics_k[f"Precision@{k}"] = mean_prec
        metrics_k[f"Recall@{k}"] = mean_rec
    
    results = {"RMSE": rmse, "MAE": mae, "Predictions": predictions}
    results.update(metrics_k)
    return results


def main(data_dir='data'):
    processed_dir = os.path.join(data_dir, 'processed')
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))
    movies_df = pd.read_csv(os.path.join(processed_dir, 'movies.csv'))
    
    metrics = {}
    
    # Content Based
    cb_results = train_content_based(train_df, test_df, movies_df)
    metrics['Content-Based'] = {k:v for k,v in cb_results.items() if k != 'Predictions'}
    print(f"Content-Based Results: {metrics['Content-Based']}")
    
    # Collaborative Filtering
    cf_results = train_collaborative_filtering(train_df, test_df)
    metrics['Collaborative-Filtering'] = {k:v for k,v in cf_results.items() if k != 'Predictions'}
    print(f"Collaborative Filtering Results: {metrics['Collaborative-Filtering']}")
    
    # Save Metrics
    with open(os.path.join(data_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Training complete. Metrics saved.")

if __name__ == "__main__":
    main('../data')
