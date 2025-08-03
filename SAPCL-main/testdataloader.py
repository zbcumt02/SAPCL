import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.data import DataLoader
import random


class SimpleGraphDataset(Dataset):
    def __init__(self, num_graphs=1000, num_nodes_range=(5, 20), feature_dim=768):
        self.num_graphs = num_graphs
        self.num_nodes_range = num_nodes_range
        self.feature_dim = feature_dim

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        # 随机生成节点数
        num_nodes = random.randint(*self.num_nodes_range)

        # 生成节点特征 (num_nodes, feature_dim)
        x = torch.rand((num_nodes, self.feature_dim), dtype=torch.float32)

        # 随机生成边 (无向图)
        num_edges = random.randint(num_nodes, num_nodes * (num_nodes - 1) // 2)  # 随机生成边数
        edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

        # 为了简单起见，y 设为图的标签
        y = torch.tensor([random.randint(0, 1)], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y, idx=idx)


# 创建数据集
dataset = SimpleGraphDataset(num_graphs=20, num_nodes_range=(5, 10), feature_dim=768)

# 使用 DataLoader 加载数据
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 打印每个批次的图的信息
for batch in dataloader:
    print(f"Batch x shape: {batch.x.shape}")  # 查看每个批次中所有图的节点特征矩阵形状
    print(f"Batch edge_index shape: {batch.edge_index.shape}")  # 查看每个批次中所有图的边连接信息形状
    print(f"Batch y: {batch.y}")  # 查看每个批次中所有图的标签
    print(f"Batch batch: {batch.batch}")  # 查看每个节点属于哪个图
    print(f"Batch idx: {batch.idx}")

