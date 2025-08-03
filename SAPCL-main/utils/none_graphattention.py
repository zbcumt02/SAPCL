import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(in_features, out_features))  # 节点特征线性变换的权重
        self.a = nn.Parameter(torch.zeros(2*out_features, 1))  # 注意力机制的参数

        # 初始化权重
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, H, adj):
        # H是节点特征矩阵, adj是邻接矩阵
        N = H.size(0)
        H_prime = torch.matmul(H, self.W)  # 对节点特征进行线性变换
        attention = torch.zeros(N, N)

        # 计算注意力系数
        for i in range(N):
            for j in range(N):
                if adj[i][j] != 0:  # 如果i和j之间有连接
                    attention[i][j] = torch.matmul(torch.cat([H_prime[i], H_prime[j]], dim=0), self.a).squeeze()

        # 归一化注意力系数
        attention = F.leaky_relu(attention, negative_slope=0.2)
        attention = torch.softmax(attention, dim=1)

        # 通过邻接矩阵聚合邻居信息
        output = torch.matmul(attention, H_prime)
        return output


class GAT(nn.Module):
    def __init__(self, in_features, out_features):
        super(GAT, self).__init__()
        self.gat_layer = GraphAttentionLayer(in_features, out_features)

    def forward(self, H, adj):
        return self.gat_layer(H, adj)


# 假设BERT生成的节点特征是一个 (1, 768) 的张量
node_features = torch.randn(1, 768)

# 邻接矩阵，假设图有5个节点
N = 5
adjacency_matrix = torch.rand((N, N))  # 假设的邻接矩阵，值在0到1之间

# 将特征矩阵扩展到符合节点数的维度 (N, 768)
node_features_expanded = node_features.expand(N, -1)

# 定义GAT模型
gat = GAT(in_features=768, out_features=128)

# 运行模型
output = gat(node_features_expanded, adjacency_matrix)[0]

# 打印输出
print("输出矩阵形状:", output.shape)
print("输出矩阵内容:", output)
