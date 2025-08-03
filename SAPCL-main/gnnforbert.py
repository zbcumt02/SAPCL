"""

更适合bert宝宝体质的GNN

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import  torch.nn.init as init


class GNNForBERT(nn.Module):
    def __init__(self, name='gnnforbert', head='mlp', feat_dim=128, num_class=5, input_feats=768,
                 hidden_feats=768, heads=4, dropout=0.5, pretrained=False):
        super(GNNForBERT, self).__init__()

        # 第一层 GAT，输入 768 维，输出 hidden_dim * heads
        self.gat1 = GATConv(input_feats, hidden_feats, heads=heads, dropout=dropout)

        # 第二层 GAT，将维度转换回 768
        self.gat2 = GATConv(hidden_feats * heads, input_feats, heads=1, dropout=dropout)

        # 增加残差连接，提升信息传播
        self.residual1 = nn.Linear(input_feats, hidden_feats * heads)
        self.residual2 = nn.Linear(hidden_feats * heads, input_feats)

        # 使用xavier初始化
        init.xavier_normal_(self.residual1.weight)
        init.xavier_normal_(self.residual2.weight)

        self.dropout = dropout
        # 分类头
        self.fc = nn.Linear(hidden_feats, num_class)

        # 保持原来的投影头结构
        if head == 'linear':
            self.head = nn.Linear(hidden_feats, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(hidden_feats, hidden_feats),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_feats, feat_dim)
            )

        # 原型缓冲区
        self.register_buffer("prototypes", torch.zeros(num_class, feat_dim))

    def forward(self, x):
        # 构建图结构
        edge_index = self.create_graph_data(x)

        # 第一层 GAT，加入残差连接
        x_res = self.residual1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gat1(x, edge_index)) + x_res  # 残差连接

        # 第二层 GAT，输出维度恢复为 768，加入残差连接
        x_res = self.residual2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gat2(x, edge_index)) + x_res  # 残差连接

        # x: [batch_size, in_feats] - 输入节点特征矩阵 (batch_size, 特征维度, 特征维度)([64, 768])

        # 图编码器处理
        feat = x  # ([batch_size, in_feats])
        # 投影头
        feat_c = self.head(feat)  # ([batch_size, in_feats / 2])
        # 分类头
        logits = self.fc(feat)  # ([64, 5])
        # 对节点特征进行归一化
        return logits, F.normalize(feat_c, dim=1)

    # 创建图数据的辅助函数
    def create_graph_data(self, x, k=5):
        """
        根据输入 x (batch_size, 768) 创建图结构，基于 K-近邻
        :param x: 输入的 BERT 向量
        :param k: 每个节点的邻居数
        :return: edge_index (图的边索引)
        """

        batch_size = x.size(0)

        # 计算余弦相似度
        x_norm = x / x.norm(dim=1, keepdim=True)
        similarity = torch.mm(x_norm, x_norm.t())  # (batch_size, batch_size)

        # 获取每个节点的 k 个最近邻
        _, topk_indices = similarity.topk(k + 1, dim=1)  # +1 因为包括自身
        topk_indices = topk_indices[:, 1:]  # 去掉自身

        # 构建 edge_index
        device = x.device
        src = torch.arange(batch_size).repeat_interleave(k).to(device)
        dst = topk_indices.reshape(-1).to(device)
        edge_index = torch.stack([src, dst], dim=0)

        return edge_index

    def contrastive_loss(self, z_i, z_j):
        """
        对比损失函数，基于相似度的对比损失，使用温度参数
        :param z_i: 输入的正样本表示
        :param z_j: 输入的负样本表示
        :return: 对比损失值
        """
        # 计算余弦相似度
        sim_ij = F.cosine_similarity(z_i, z_j, dim=1)

        # 计算对比损失 (InfoNCE 损失)
        loss = -torch.log(torch.exp(sim_ij / self.temperature) / torch.sum(torch.exp(sim_ij / self.temperature), dim=0))

        return loss.mean()

