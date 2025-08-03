import torch
import torch.nn as nn
import torch.nn.functional as F


# GraphSAGE层定义
class GraphSAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats, aggr='mean'):
        super(GraphSAGEConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggr = aggr

        # 线性变换层：自身节点特征
        self.self_linear = nn.Linear(in_feats, out_feats, bias=False)
        # 线性变换层：邻居聚合特征
        self.neigh_linear = nn.Linear(in_feats, out_feats, bias=False)
        self.bn = nn.BatchNorm1d(out_feats)

    def forward(self, x, adj):
        # x: [N, in_feats] - 节点特征矩阵
        # adj: [N, N] - 邻接矩阵

        # 自身节点特征变换
        self_feats = self.self_linear(x)

        # 邻居特征聚合
        if self.aggr == 'mean':
            neigh_feats = torch.mm(adj, x)  # 邻居特征加权和
            neigh_feats = self.neigh_linear(neigh_feats)
        else:
            raise NotImplementedError(f'Aggregation {self.aggr} not supported')

        # 组合自身和邻居特征
        out = self_feats + neigh_feats
        out = self.bn(out)
        return F.relu(out)


# GraphSAGE编码器
class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2):
        super(GraphSAGEEncoder, self).__init__()
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(GraphSAGEConv(in_feats, hidden_feats))

        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGEConv(hidden_feats, hidden_feats))

        # 输出层
        self.layers.append(GraphSAGEConv(hidden_feats, out_feats))

    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        return h


class SupConGNN1(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='supconGNN1', head='mlp', feat_dim=128, num_class=2, in_feats=16, hidden_feats=512,
                 num_layers=2, pretrained=False):
        super(SupConGNN1, self).__init__()

        # 使用GraphSAGE替换原来的图像编码器
        self.encoder = GraphSAGEEncoder(
            in_feats=in_feats,  # 输入特征维度（例如BERT的768）
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

    def forward(self, x, adj):
        # x: [N, in_feats] - 节点特征矩阵
        # adj: [N, N] - 邻接矩阵

        # 图编码器处理
        feat = self.encoder(x, adj)

        # 投影头
        feat_c = self.head(feat)

        # 分类头
        logits = self.fc(feat)

        return logits, F.normalize(feat_c, dim=1)

# 使用示例
# def create_model(in_feats=768, num_class=10):
#     model = SupConGNN1(
#         in_feats=in_feats,  # 输入特征维度
#         hidden_feats=512,  # 隐藏层维度
#         feat_dim=128,  # 投影特征维度
#         num_class=num_class,  # 分类数量
#         num_layers=2  # GNN层数
#     )
#     return model
#
#
# # 测试
# if __name__ == "__main__":
#     # 模拟输入
#     batch_size = 32
#     num_nodes = 100
#     in_feats = 768
#
#     x = torch.randn(num_nodes, in_feats)  # 节点特征
#     adj = torch.ones(num_nodes, num_nodes)  # 全连接图作为示例
#
#     model = create_model(in_feats=in_feats, num_class=10)
#     logits, features = model(x, adj)
#     print(f"Logits shape: {logits.shape}")  # [num_nodes, num_class]
#     print(f"Features shape: {features.shape}")  # [num_nodes, feat_dim]
