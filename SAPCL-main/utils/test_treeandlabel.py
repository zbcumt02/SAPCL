import json
import torch
import numpy as np


# 将树结构平展，获取每个节点和它的邻接关系
def get_neighbors(tree):
    """递归解析树形结构，获取每个节点的邻接关系"""
    neighbors = {}

    def traverse(node, tree):
        if isinstance(tree, dict):
            for child in tree:
                # 如果节点是字典，就记录这个节点与子节点的关系
                if node not in neighbors:
                    neighbors[node] = []
                if child != node:
                    neighbors[node].append(child)  # 保存子节点作为邻接节点
                traverse(child, tree[child])  # 递归遍历子节点
    for root in tree:
        traverse(root, tree)

    return neighbors


# 获取某个节点的邻接一跳和二跳标签
def get_labels_for_node(example_node, neighbors, labels):
    # 获取一跳邻居的标签
    one_hop_neighbors = neighbors.get(example_node, [])
    # 将 whole_tree 的键转换为列表来找到邻居的索引
    one_hop_labels = [labels[int(neighbor)] for neighbor in one_hop_neighbors]
    # 获取二跳邻居的标签
    two_hop_neighbors = []
    for one_hop in one_hop_neighbors:
        two_hop_neighbors.extend(neighbors.get(one_hop, []))

    two_hop_labels = [labels[int(neighbor)] for neighbor in two_hop_neighbors if
                      neighbor != example_node]

    three_hop_neighbors = []
    for two_hop in two_hop_neighbors:
        three_hop_neighbors.extend(neighbors.get(two_hop, []))

    three_hop_labels = [labels[int(neighbor)] for neighbor in three_hop_neighbors if
                        neighbor != example_node]

    return one_hop_labels, two_hop_labels, three_hop_labels


five_events = ['ch', 'fg', 'gc', 'ow', 'ss']
# 训练集预处理
whole_tree = []
label_set = []
label_arr = []

for eve_index in range(0, 5):
    train_tree_in = open('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/'+five_events[eve_index] + '_tree.json', 'r')
    train_label_in = open('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/'+five_events[eve_index] + '_label.txt', 'r')
    for line in train_tree_in:
        line = line.strip()
        whole_tree.append(json.loads(line))
    for line in train_label_in:
        label_set.append([int(x) for x in line.strip().split("\t")])
        label_arr += [int(x) for x in line.strip().split("\t")]
    train_tree_in.close()
    train_label_in.close()

    # len(whole_tree) 274
    # len(label_set) 274
    # len(label_arr) 4612

total_count_in_one_hop = 0

for i in range(len(whole_tree)):
    # 遍历每一棵树
    tree = whole_tree[i]  # 取出其中一个树结构（假设是全局树结构，也可以调整为不同树）
    label = label_set[i]
    neighbors = get_neighbors(tree)

    # for node, adj_nodes in neighbors.items():
    #     print(f"Node {node} has neighbors: {adj_nodes}")
    #
    # for example_node in neighbors:
    #     one_hop_labels, two_hop_labels, three_hop_labels = get_labels_for_node(example_node, neighbors, label)
    #     print(f"Node {example_node}:")
    #     print(f"One-hop labels: {one_hop_labels}")
    #     print(f"Two-hop labels: {two_hop_labels}")
    #     print(f"Three-hop labels: {three_hop_labels}")

    parent_label = 4
    child_label = 4
    # 找到所有标签为x的节点
    zero_label_nodes = [i for i in range(len(label)) if label[i] == parent_label]

    # 遍历所有标签为x的节点
    for example_node in zero_label_nodes:
        # print(example_node, neighbors, label)
        one_hop_labels, two_hop_labels, three_hop_labels = get_labels_for_node(str(example_node), neighbors, label)
        # count_zero_in_neighbors = 0
        # for hop_index in range(len(one_hop_labels)):
        #     print(hop_index, one_hop_labels)
        #     if one_hop_labels[hop_index] == 0:
        #         print(hop_index)
        #         count_zero_in_neighbors += 1
        count_zero_in_neighbors = sum(1 for hop_index in range(len(one_hop_labels)) if
                                      one_hop_labels[hop_index] == child_label)
        # 累加结果
        total_count_in_one_hop += count_zero_in_neighbors

    # 输出统计结果
    print(f"Total with label parent with label child: {total_count_in_one_hop}")

    # break
