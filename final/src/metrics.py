import numpy as np

def precision_at_k(recommended_indices, true_indices, k):
    if len(true_indices) == 0:
        return 0.0
    recommended_k = recommended_indices[:k]
    hits = len(set(recommended_k) & set(true_indices))
    return hits / k

def recall_at_k(recommended_indices, true_indices, k):
    if len(true_indices) == 0:
        return 0.0
    recommended_k = recommended_indices[:k]
    hits = len(set(recommended_k) & set(true_indices))
    return hits / len(true_indices)

def dcg_at_k(recommended_indices, true_indices, k):
    recommended_k = recommended_indices[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in true_indices:
            dcg += 1.0 / np.log2(i + 2)
    return dcg

def ndcg_at_k(recommended_indices, true_indices, k):
    dcg = dcg_at_k(recommended_indices, true_indices, k)
    # IDCG: Best possible DCG (all true items at top)
    ideal_k = min(len(true_indices), k)
    idcg = 0.0
    for i in range(ideal_k):
        idcg += 1.0 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_models(model, test_df, train_df, k_list=[5, 10]):
    """
    Evaluates a model on test data.
    model: object with .recommend(user_id, k) method
    test_df: dataframe with user-item interactions
    train_df: dataframe used for training (to exclude seen items)
    """
    users = test_df['reviewerID'].unique()
    metrics = {f'P@{k}': [] for k in k_list}
    metrics.update({f'R@{k}': [] for k in k_list})
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    
    # Group truth by user
    ground_truth = test_df.groupby('reviewerID')['asin'].apply(set).to_dict()
    seen_items = train_df.groupby('reviewerID')['asin'].apply(set).to_dict()
    
    print(f"Evaluating on {len(users)} users...")
    
    for i, user in enumerate(users):
        if user not in ground_truth:
            continue
            
        true_items = ground_truth[user]
        seen = seen_items.get(user, set())
        
        # Request max K recommendations
        max_k = max(k_list)
        recs = model.recommend(user, k=max_k, exclude=seen)
        
        for k in k_list:
            metrics[f'P@{k}'].append(precision_at_k(recs, true_items, k))
            metrics[f'R@{k}'].append(recall_at_k(recs, true_items, k))
            metrics[f'NDCG@{k}'].append(ndcg_at_k(recs, true_items, k))
            
    # Average
    results = {k: np.mean(v) for k, v in metrics.items()}
    return results
