import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

class PopularityRecommender:
    def __init__(self):
        self.popular_items = []
    
    def fit(self, train_df):
        print("Training Popularity Model...")
        # Count frequency of each item
        counts = train_df['asin'].value_counts()
        self.popular_items = counts.index.tolist()
        print(f"Learned {len(self.popular_items)} items.")
        
    def recommend(self, user_id, k=10, exclude=None):
        preds = []
        for item in self.popular_items:
            if exclude and item in exclude:
                continue
            preds.append(item)
            if len(preds) == k:
                break
        return preds

class ItemKNNRecommender:
    def __init__(self, k_neighbors=20):
        self.k_neighbors = k_neighbors
        self.sim_matrix = None
        self.item_mapper = {}
        self.reverse_mapper = {}
        self.user_mapper = {}
        self.train_mat = None
        
    def fit(self, train_df):
        print(f"Training ItemKNN (k={self.k_neighbors})...")
        
        # Create mapping
        unique_users = train_df['reviewerID'].unique()
        unique_items = train_df['asin'].unique()
        
        self.user_mapper = {u: i for i, u in enumerate(unique_users)}
        self.item_mapper = {i: idx for idx, i in enumerate(unique_items)}
        self.reverse_mapper = {idx: i for idx, i in enumerate(unique_items)}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Create Sparse Matrix (Users x Items)
        rows = train_df['reviewerID'].map(self.user_mapper)
        cols = train_df['asin'].map(self.item_mapper)
        
        # Implicit feedback = 1
        data = np.ones(len(train_df))
        
        self.train_mat = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # Compute Item-Item Similarity (Cosine)
        # Transpose to get Items x Users
        item_user_mat = self.train_mat.T.tocsr()
        
        # Compute similarity (can be large, cautious)
        # If too large, we compute on the fly or using TruncatedSVD (but this is pure memory-based KNN)
        print("Computing similarity matrix...")
        self.sim_matrix = cosine_similarity(item_user_mat, dense_output=False)
        print("Similarity computed.")
        
    def recommend(self, user_id, k=10, exclude=None):
        if user_id not in self.user_mapper:
            # Cold user: fallback to something or empty
            return []
        
        u_idx = self.user_mapper[user_id]
        
        # Get user interactions (sparse vector 1 x n_items)
        user_vector = self.train_mat[u_idx]
        
        # Scores = user_vector * sim_matrix (1 x n_items)
        # This gives score for all items based on sum of similarities to items user interacted with
        scores = user_vector.dot(self.sim_matrix)
        
        # Convert to dense for sorting
        if sp.issparse(scores):
            scores = scores.toarray().flatten()
        
        # Sort indices
        # We need to filter excludes
        
        # Get candidate items (all items)
        # Set seen items score to -1
        if exclude:
            for item in exclude:
                if item in self.item_mapper:
                    idx = self.item_mapper[item]
                    scores[idx] = -1.0
        
        # Also exclude items user already interacted with (if not handled by 'exclude' passed arg)
        # Usually exclude=seen_items.
        
        # Top-K
        # argpartition is faster than argsort for top-k
        top_indices = np.argpartition(scores, -k)[-k:]
        
        # Sort the top k
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        recs = [self.reverse_mapper[idx] for idx in top_indices if scores[idx] > -1.0]
        return recs
