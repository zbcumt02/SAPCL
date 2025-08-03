import random
import math
import torch
import numpy as np
from torch_geometric.data import Data


def augment_graph(x, edge_index, edge_attr=None, mask_node_ratio=0.25, mask_edge_ratio=0.25):
    """
    对图进行增强，随机掩码节点和边。

    :param x: 节点特征矩阵 (N, F)
    :param edge_index: 边连接矩阵 (2, E)
    :param edge_attr: 可选的边特征 (E, F)，如果没有则传入 None
    :param mask_node_ratio: 随机掩码节点的比例 (default: 0.25)
    :param mask_edge_ratio: 随机掩码边的比例 (default: 0.25)

    :return: 掩码后的图数据（Data对象）
    """
    N = x.shape[0]  # 节点数量
    M = edge_index.shape[1]  # 边数量

    # 计算需要掩码的节点和边数量
    num_mask_nodes = max(1, math.floor(mask_node_ratio * N))
    num_mask_edges = max(0, math.floor(mask_edge_ratio * M))

    # 随机选择掩码的节点和边
    mask_nodes_i = random.sample(range(N), num_mask_nodes)
    mask_nodes_j = random.sample(range(N), num_mask_nodes)
    mask_edges_i_single = random.sample(range(M), num_mask_edges)

    # 对节点进行掩码：设置掩码节点的特征为0或其他特定值
    augmented_x = x.clone()
    augmented_x[mask_nodes_i] = 0  # 将掩码节点的特征设为0（也可以是其他的掩码值）

    # 对边进行掩码：删除掩码的边
    augmented_edge_index = edge_index.clone()
    augmented_edge_attr = edge_attr.clone() if edge_attr is not None else None

    augmented_edge_index = np.delete(augmented_edge_index, mask_edges_i_single, axis=1)

    if augmented_edge_attr is not None:
        augmented_edge_attr = np.delete(augmented_edge_attr, mask_edges_i_single, axis=0)

    # 创建处理后的图数据
    augmented_data = Data(x=augmented_x, edge_index=torch.tensor(augmented_edge_index, dtype=torch.long))

    if augmented_edge_attr is not None:
        augmented_data.edge_attr = torch.tensor(augmented_edge_attr, dtype=torch.long)

    return augmented_data


# # 假设你有预处理过的图数据
# x = torch.tensor(node_features)  # 节点特征矩阵 (N, F)
# edge_index = torch.tensor(edge_list)  # 边连接矩阵 (2, E)
#
# # 对图进行增强
# augmented_data = augment_graph(x, edge_index)
#
# # 获取增强后的图的特征和边
# augmented_x = augmented_data.x  # 增强后的节点特征
# augmented_edge_index = augmented_data.edge_index  # 增强后的边连接矩阵
