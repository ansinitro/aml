import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.baselines import PopularityRecommender, ItemKNNRecommender
from models.mf import MatrixFactorization
from models.ncf import NeuMF
from data_loader import ImplicitFeedbackDataset
from metrics import evaluate_models
import os

class TorchModelWrapper:
    def __init__(self, model, user_map, item_map, reverse_item_map, device):
        self.model = model
        self.user_map = user_map
        self.item_map = item_map
        self.reverse_item_map = reverse_item_map
        self.device = device
        
    def recommend(self, user_id, k=10, exclude=None):
        # Handle unknown user
        if user_id not in self.user_map:
            return []
        
        u_idx = self.user_map[user_id]
        
        # Map exclude items (strings) to indices
        mapped_exclude = None
        if exclude:
            mapped_exclude = [self.item_map[i] for i in exclude if i in self.item_map]
            
        # Get recommendations (indices)
        rec_indices = self.model.recommend(u_idx, k=k, exclude=mapped_exclude, device=self.device)
        
        # Map back to item strings
        return [self.reverse_item_map[i] for i in rec_indices]

def run_experiment(args):
    print(f"Loading data from {args.data_dir}...")
    train_df = pd.read_parquet(os.path.join(args.data_dir, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(args.data_dir, 'val.parquet')) 
    test_df = pd.read_parquet(os.path.join(args.data_dir, 'test.parquet'))
    
    # Concatenate to find all unique IDs
    all_df = pd.concat([train_df, val_df, test_df])
    unique_users = all_df['reviewerID'].unique()
    unique_items = all_df['asin'].unique()
    
    user_map = {u: i for i, u in enumerate(unique_users)}
    item_map = {i: idx for idx, i in enumerate(unique_items)}
    reverse_item_map = {idx: i for idx, i in enumerate(unique_items)}
    
    n_users = len(user_map)
    n_items = len(item_map)
    
    # Apply mapping for training
    train_df['reviewerID_idx'] = train_df['reviewerID'].map(user_map)
    train_df['asin_idx'] = train_df['asin'].map(item_map)
    
    print(f"Data Loaded: {len(train_df)} train, {len(test_df)} test.")
    print(f"Users: {n_users}, Items: {n_items}")
    
    model = None
    if args.model == 'pop':
        model = PopularityRecommender()
        model.fit(train_df)
        
    elif args.model == 'knn':
        model = ItemKNNRecommender(k_neighbors=args.k)
        model.fit(train_df)
        
    elif args.model in ['mf', 'ncf']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on {device}...")
        
        if args.model == 'mf':
            model = MatrixFactorization(n_users, n_items, n_factors=args.embedding_dim).to(device)
        else:
            model = NeuMF(n_users, n_items, factor_num=args.embedding_dim).to(device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        dataset = ImplicitFeedbackDataset(train_df, n_users, n_items, num_negatives=args.negatives)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for u, i, negs in loader:
                u = u.to(device)
                i = i.to(device)
                negs = negs.to(device) 
                
                # Positive
                pos_pred = model(u, i)
                pos_label = torch.ones_like(pos_pred)
                loss = loss_fn(pos_pred, pos_label)
                
                # Negatives
                for n_idx in range(negs.shape[1]):
                    n_item = negs[:, n_idx]
                    neg_pred = model(u, n_item)
                    neg_label = torch.zeros_like(neg_pred)
                    loss += loss_fn(neg_pred, neg_label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} Loss: {total_loss:.4f}")
            
        model.eval()
        # Wrap the model for evaluation
        model = TorchModelWrapper(model, user_map, item_map, reverse_item_map, device)
        
    # Evaluate
    print("Evaluating...")
    results = evaluate_models(model, test_df, train_df, k_list=[5, 10])
    
    print("-" * 30)
    print(f"Model: {args.model}")
    print(results)
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['pop', 'knn', 'mf', 'ncf'])
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--negatives', type=int, default=4)
    parser.add_argument('--k', type=int, default=20) # for KNN
    
    args = parser.parse_args()
    run_experiment(args)
