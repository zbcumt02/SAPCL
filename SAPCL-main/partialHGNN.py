import torch
import torch.nn as nn
import torch.nn.functional as F


class SAPCL_GNN(nn.Module):
    def __init__(self, name='SAPCL_GNN', head='mlp', feat_dim=128, num_class=5, input_feats=768,
                 hidden_feats=768, dropout=0.5, pretrained=False):
        super(SAPCL_GNN, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(input_feats, num_class)

        self.mlp1 = nn.Linear(input_feats, input_feats)
        self.mlp2 = nn.Linear(input_feats, input_feats)

        if head == 'linear':
            self.head = nn.Linear(hidden_feats, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(input_feats, hidden_feats),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_feats, feat_dim)
            )

        self.register_buffer("prototypes", torch.zeros(num_class, feat_dim))

    def forward(self, x):
        feat = x

        feat = self.dropout(feat)
        # 分类头
        logits = self.dense(feat)
        # 投影头
        # feat = self.mlp1(self.mlp2(feat))
        feat = self.mlp1(feat)
        feat_c = self.head(feat)  # ([batch_size, in_feats / 2])

        return logits, F.normalize(feat_c, dim=1)
