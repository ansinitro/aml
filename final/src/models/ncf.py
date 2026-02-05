import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, factor_num=32, layers=[64, 32, 16], dropout=0.0):
        super(NeuMF, self).__init__()
        
        # GMF Part
        self.embed_user_GMF = nn.Embedding(n_users, factor_num)
        self.embed_item_GMF = nn.Embedding(n_items, factor_num)
        
        # MLP Part
        self.embed_user_MLP = nn.Embedding(n_users, int(layers[0]/2))
        self.embed_item_MLP = nn.Embedding(n_items, int(layers[0]/2))
        
        self.mlp_modules = []
        for i, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.mlp_modules.append(nn.Linear(in_size, out_size))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        
        self.mlp_layers = nn.Sequential(*self.mlp_modules)
        
        # Final Prediction Layer
        # Concatenation of GMF (factor_num) and MLP (layers[-1])
        predict_size = factor_num + layers[-1]
        self.predict_layer = nn.Linear(predict_size, 1)
        
        # Init weights (LeCun uniform or Xavier)
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user, item):
        # GMF
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        output_GMF = embed_user_GMF * embed_item_GMF
        
        # MLP
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.mlp_layers(interaction)
        
        # Concatenate
        concat = torch.cat((output_GMF, output_MLP), -1)
        
        # Prediction
        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def recommend(self, user_idx, k=10, exclude=None, device='cpu'):
        # For NCF, we need to correct forward pass for all items
        # efficient way: embedding lookup once
        all_items = torch.arange(self.embed_item_GMF.num_embeddings, device=device)
        user_repeated = torch.tensor([user_idx], device=device).repeat(len(all_items))
        
        with torch.no_grad():
            scores = self.forward(user_repeated, all_items)
            
        if exclude:
            scores[list(exclude)] = -float('inf')
        
        top_k = torch.topk(scores, k)
        return top_k.indices.cpu().numpy()
