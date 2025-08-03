"""
针对274个会话树中的每个节点提取特征

for 2
"""
import json
import torch
import numpy as np
from torch import nn

import warnings
import torch.nn.functional as F
from .utils_algo import generate_uniform_edges_candidate_labels
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning)


class phemes_train(Dataset):
    def __init__(self, data_list, graph_list, label_list, ground_list):
        self.datas = data_list
        self.graphs = graph_list
        self.labels = label_list
        self.ground_labels = ground_list

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        index = idx
        feature_w = torch.tensor(self.datas[index], dtype=torch.float32)
        feature_s = torch.tensor(mask_random_features(self.datas[index]), dtype=torch.float32)
        edge_index = torch.tensor(self.graphs[index], dtype=torch.long)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        ground_truth = torch.tensor(self.ground_labels[index], dtype=torch.long)
        idx = torch.tensor(idx, dtype=torch.long)

        return Data(w=feature_w, s=feature_s, edge_index=edge_index, y=y,
                    ground_truth=ground_truth, idx=idx)


class phemes_test(Dataset):
    def __init__(self, data_list, graph_list, label_list):
        self.datas = data_list
        self.graphs = graph_list
        self.labels = label_list

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        x = torch.tensor(self.datas[index], dtype=torch.float32)
        edge_index = torch.tensor(self.graphs[index], dtype=torch.long)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        idx = torch.tensor(index, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y, idx=idx)

#
# def custom_collate_fn_train(batch):
#     graphs = [item[2] for item in batch]
#     graphs = [torch.tensor(g) if not isinstance(g, torch.Tensor) else g for g in graphs]
#
#     # 调试：输出 graph 类型，确认它们是张量
#     for i, g in enumerate(graphs):
#         print(f"Graph {i} type: {type(g)}, shape: {g.shape if isinstance(g, torch.Tensor) else 'N/A'}")
#
#     padded_graphs = pad_sequence(graphs, batch_first=True, padding_value=0)
#
#     week_features = [item[0] for item in batch]
#     strong_features = [item[1] for item in batch]
#     labels = [item[3] for item in batch]
#     ground_labels = [item[4] for item in batch]
#     indices = [item[5] for item in batch]
#
#     week_features = torch.stack(week_features, dim=0)
#     strong_features = torch.stack(strong_features, dim=0)
#     labels = torch.stack(labels, dim=0)
#     ground_labels = torch.stack(ground_labels, dim=0)
#
#     return week_features, strong_features, padded_graphs, labels, ground_labels, indices
#
#
# def custom_collate_fn_test(batch):
#     features = [item[0] for item in batch]
#     graphs = [item[1] for item in batch]
#     labels = [item[2] for item in batch]
#
#     features = torch.stack(features, dim=1)
#     graphs = torch.stack(graphs, dim=0)
#     labels = torch.stack(labels, dim=1)
#
#     return features, graphs, labels


def add_random_noise(semantic_vector, noise_level=0.05):
    noise = np.random.normal(0, noise_level, semantic_vector.shape)
    return semantic_vector + noise


def mask_random_features(semantic_vector, mask_prob=0.15):
    n, m = semantic_vector.shape
    mask = torch.rand(n, m) < mask_prob
    semantic_vector[mask] = 0

    return semantic_vector


def get_tree(train_feature, whole_tree):
    train_feature = torch.tensor(train_feature)

    def count_nodes(tree):
        count = 1  # 当前节点本身计为一个节点

        for child in tree.values():
            if isinstance(child, dict):  # 如果子节点是一个字典，表示有子树
                count += count_nodes(child)  # 递归计算子树的节点数量
            else:
                count += 1
        return count
    whole_tree_nodes = count_nodes(whole_tree)
    output = torch.zeros(0, 768)
    for i in range(whole_tree_nodes):
        output = torch.cat((output, train_feature[i].unsqueeze(0)), dim=0)

    # epsilon = 1e-6
    # output = output - output.mean(dim=0) + epsilon

    return output.float()


def tree_to_edges_matrix(tree):
    edges = []

    def traverse(node, parent):
        for key, value in node.items():
            parent_int = int(parent) if parent is not None else None
            key_int = int(key)
            if parent_int is not None:
                edges.append([parent_int, key_int])
            if isinstance(value, dict):
                traverse(value, key)
    traverse(tree, None)

    edges_matrix = np.array(edges).T  # 转置为2行
    return torch.tensor(edges_matrix)


def load_phemes_edges(args, partial_rate, batch_size):
    test_ratio = 0.2
    five_events = ['ch', 'fg', 'gc', 'ow', 'ss']

    # 训练集预处理
    whole_tree = []
    label_set = []
    label_arr = []

    for eve_index in range(0, 5):
        tree_in = open('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/'+five_events[eve_index] + '_tree.json', 'r')
        label_in = open('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/'+five_events[eve_index] + '_label.txt', 'r')
        for line in tree_in:
            line = line.strip()
            whole_tree.append(json.loads(line))
        for line in label_in:
            label_set.append([int(x) for x in line.strip().split("\t")])
            label_arr += [int(x) for x in line.strip().split("\t")]
        tree_in.close()
        label_in.close()

        # len(whole_tree) 274
        # len(label_set) 274
        # len(label_arr) 4612

    data_feature = np.concatenate((np.load('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/' + five_events[0] + "deberta-v3.npy"),
                                    np.load('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/' + five_events[1] + "deberta-v3.npy"),
                                    np.load('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/' + five_events[2] + "deberta-v3.npy"),
                                    np.load('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/' + five_events[3] + "deberta-v3.npy"),
                                    np.load('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/' + five_events[4] + "deberta-v3.npy")))
    # shape(274, 103, 768)
    data_index_set = [i for i in range(0, data_feature.shape[0])]  #length 274 会话集
    # 计算训练集和测试集的划分大小
    test_size = int(len(data_index_set) * test_ratio)
    # 划分数据集  220 54
    np.random.shuffle(data_index_set)

    # 划分训练集和测试集
    train_data_index = sorted(data_index_set[test_size:])
    test_data_index = sorted(data_index_set[:test_size])

    output_train = []
    train_edges_matrix_list = []
    label_arr_index = 0
    train_label_arr_list = []
    arr_index_list = [(int(label_arr_index))]

    for index in train_data_index:
        edges_matrix = tree_to_edges_matrix(whole_tree[index])
        train_edges_matrix_list.append(edges_matrix)
        output = get_tree(data_feature[index], whole_tree[index]['0'])  #shape(n, 768)
        output_train.append(output)
        n = output.size(0)
        train_label_arr_list.append(torch.tensor(label_arr[label_arr_index:label_arr_index + n]))
        label_arr_index += n
        arr_index_list.append(int(label_arr_index))
        # print(label_arr_index, label_arr_list[index])
        # output_combined = torch.cat((output_combined, output), dim=0)  # 4612*768
    # print(output_train[0].shape, train_edges_matrix_list[0].shape, train_label_arr_list[0].shape)
    # torch.Size([25, 768]) torch.Size([2, 24]) torch.Size([25])
    print(train_label_arr_list[0], len(train_label_arr_list))
    partialY = []
    for i in range(len(train_data_index)):
        partialY.append(generate_uniform_edges_candidate_labels(args.num_class, torch.tensor(train_label_arr_list[i]), partial_rate))
        # train_initial = torch.zeros([train_label_arr_list[i].size(0), args.num_class])
        # for j in range(train_label_arr_list[i].size(0)):
        #     train_initial[j][train_label_arr_list[i][j]] = 1
        #
        # partialY.append(train_initial)

    print(partialY[0], len(partialY))# partialY = torch.zeros([train_labels.size(0), 5])
    # for i in range(train_labels.size(0)):
    #     partialY[i][train_labels[i]] = 1

    print("Finish Generating Candidate Label Sets!\n")

    partial_matrix_dataset = phemes_train(output_train, train_edges_matrix_list, partialY, train_label_arr_list)

    output_test = []
    test_edges_matrix_list = []
    label_arr_index = 0
    test_label_arr_list = []
    for index in test_data_index:
        edges_matrix = tree_to_edges_matrix(whole_tree[index])
        test_edges_matrix_list.append(edges_matrix)
        output = get_tree(data_feature[index], whole_tree[index]['0'])  # shape(n, 768)
        output_test.append(output)
        n = output.size(0)
        test_label_arr_list.append(label_arr[label_arr_index:label_arr_index + n])
        label_arr_index += n
        # print(label_arr_index, label_arr_list[index])
        # output_combined = torch.cat((output_combined, output), dim=0)  # 4612*768
    test_matrix_dataset = phemes_test(output_test, test_edges_matrix_list, test_label_arr_list)

    # train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    # train_labels = torch.tensor([train_dataset[i][2] for i in range(len(train_dataset))], dtype=torch.long)
    # train_data = train_data.detach()

    # test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    # test_labels = torch.tensor([test_dataset[i][2] for i in range(len(test_dataset))], dtype=torch.long)
    # test_data = test_data.detach().float()

    train_loader = DataLoader(dataset=partial_matrix_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              sampler=None,
                              drop_last=True)  # collate_fn=custom_collate_fn_train
    test_loader = DataLoader(dataset=test_matrix_dataset, batch_size=batch_size*4,
                             shuffle=False, num_workers=4, sampler=None)

    # index_all = torch.tensor(torch.zeros(0))
    # for batch_data in train_loader:
    #     print(batch_data, batch_data.w.shape, batch_data.edge_index.shape, batch_data.y.shape,
    #           batch_data.ground_truth.shape, batch_data.index, batch_data.batch, batch_data.ptr)
    #     index_all = torch.cat((index_all, batch_data.idx), dim=0)
    # print(index_all.shape, sorted(index_all))
    # for batch_data in test_loader:
    #     print(batch_data.x, batch_data.edge_index, batch_data.y, batch_data.batch, batch_data.index)
    #     index_all = torch.cat((index_all, batch_data.idx), dim=0)
    # print(index_all.shape, sorted(index_all))
    # print(len(arr_index_list), arr_index_list)
    # print(arr_index_list[107] - arr_index_list[106])
    # print(arr_index_list[77] - arr_index_list[76])
    # print(arr_index_list[6] - arr_index_list[5])
    # print(arr_index_list[211] - arr_index_list[210])

    return train_loader, partialY, test_loader, arr_index_list


# if __name__ == '__main__':
#     load_phemes(partial_rate=0.05, batch_size=4)

