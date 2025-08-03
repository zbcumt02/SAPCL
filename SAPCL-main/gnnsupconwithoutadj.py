"""

for treeLSTM
it has failed

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        in_feats = int(in_feats ** 0.5)
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.dropout = dropout
        self.alpha = alpha

        # 定义注意力层的权重
        self.W = nn.Linear(in_feats * in_feats, out_feats, bias=False)
        self.a = nn.Parameter(torch.Tensor(1, out_feats * 2))  # 注意力权重
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)

        # 初始化参数
        nn.init.xavier_uniform_(self.W.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, x):
        # x: [batch_size, in_feats, in_feats] - 输入节点特征矩阵 (batch_size, 特征维度, 特征维度)
        batch_size, feature1, feature2 = x.size()

        # 对每个节点的特征矩阵进行展平处理
        x_flat = x.view(batch_size, -1)  # [batch_size, in_feats * in_feats]
        # 线性变换得到新的特征
        h = self.W(x_flat)  # [batch_size, out_feats]
        # 注意力计算，假设节点之间有连接关系
        attention = torch.zeros(batch_size, batch_size).to(x.device)
        for i in range(batch_size):
            for j in range(batch_size):
                # 对每对节点计算注意力系数
                attention[i, j] = self.leakyrelu(torch.sum(torch.cat([h[i], h[j]], dim=0) @ self.a.T, dim=0))

        # 归一化处理，避免梯度爆炸
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)

        # feature1 = int(feature1 * 2)
        # feature2 = int(feature2 / 2)

        # 根据注意力系数加权求和邻居节点的特征
        out = torch.mm(attention, h)  # [batch_size, out_feats]
        out = out.reshape(batch_size, feature1, feature2)
        return out


# GraphSAGE编码器更新
class GATEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2):
        super(GATEncoder, self).__init__()
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(GraphAttentionLayer(in_feats, hidden_feats))

        # 隐藏层
        # for _ in range(num_layers - 2):
        #     self.layers.append(GraphAttentionLayer(hidden_feats, hidden_feats))
        #
        # 输出层
        self.layers.append(GraphAttentionLayer(hidden_feats, out_feats))

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        batch_size, feature1, feature2 = h.size()
        h = h.reshape(batch_size, int(feature1 * feature2))
        return h


class SupConGNN2(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='supconGNN2', head='mlp', feat_dim=128, num_class=5, in_feats=256, hidden_feats=256,
                 num_layers=2, pretrained=False):
        super(SupConGNN2, self).__init__()

        # 使用GAT替换原来的图像编码器
        self.encoder = GATEncoder(
            in_feats=in_feats,
            hidden_feats=hidden_feats,  # 隐藏层维度，与原来ResNet的512对齐
            out_feats=hidden_feats,  # 输出维度保持一致
            num_layers=num_layers
        )

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
        # x: [batch_size, in_feats, in_feats] - 输入节点特征矩阵 (batch_size, 特征维度, 特征维度)([64, 16, 16])

        # 图编码器处理
        feat = self.encoder(x)  # ([64, 256])
        # 投影头
        feat_c = self.head(feat)  # ([64, 128])
        # 分类头
        logits = self.fc(feat)  # ([64, 5])
        # 对节点特征进行归一化
        return logits, F.normalize(feat_c, dim=1)
