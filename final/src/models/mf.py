import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=32):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # Initialize
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        
    def forward(self, user, item):
        u_emb = self.user_factors(user)
        i_emb = self.item_factors(item)
        return (u_emb * i_emb).sum(1)
    
    def recommend(self, user_idx, k=10, exclude=None, device='cpu'):
        # Predict for all items for this user
        u_emb = self.user_factors(torch.tensor([user_idx], device=device)) # 1 x F
        scores = torch.matmul(u_emb, self.item_factors.weight.t()).squeeze() # 1 x I
        
        if exclude:
            scores[list(exclude)] = -float('inf')
        
        top_k = torch.topk(scores, k)
        return top_k.indices.cpu().numpy()
