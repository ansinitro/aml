import torch
from torch.utils.data import Dataset
import numpy as np

class ImplicitFeedbackDataset(Dataset):
    def __init__(self, df, n_users, n_items, num_negatives=4):
        self.users = torch.tensor(df['reviewerID_idx'].values, dtype=torch.long)
        self.items = torch.tensor(df['asin_idx'].values, dtype=torch.long)
        self.n_users = n_users
        self.n_items = n_items
        self.num_negatives = num_negatives
        
        # Build set of interacted items for each user for fast negative sampling
        self.user_item_set = set(zip(df['reviewerID_idx'].values, df['asin_idx'].values))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        
        # Negative Sampling
        neg_items = []
        for _ in range(self.num_negatives):
            while True:
                neg_item = np.random.randint(0, self.n_items)
                if (user.item(), neg_item) not in self.user_item_set:
                    neg_items.append(neg_item)
                    break
        
        # Return user, positive_item, negative_item(s)
        # For BPR we usually take pairs (u, i, j).
        # For Pointwise BCE we take (u, i, 1) and (u, j, 0).
        # Let's stick to simple triplet (u, i, j) for BPR usually, 
        # but here let's return neg_items as list
        return user, item, torch.tensor(neg_items, dtype=torch.long)
