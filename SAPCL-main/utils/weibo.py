"""
针对weibo数据集每个节点提取特征

"""
import json
import torch
import numpy as np
from torch import nn

import warnings
import random
from torch.nn import LSTM
from .utils_algo import generate_uniform_cv_candidate_labels

from torch.utils.data import DataLoader, Dataset, random_split

warnings.filterwarnings("ignore", category=UserWarning)


class weibosDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class weibos_test_combine(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        # user-defined label (partial labels)
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.labels[index]

        return data, label


class weibos_Augmentention(Dataset):
    def __init__(self, dataset, given_label_matrix, true_labels):
        self.dataset = dataset
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        squared_matrix = self.dataset[index] ** 2
        noise_matrix = self.dataset[index]
        each_feature_w = noise_matrix.float()
        each_feature_s = squared_matrix.float()
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]

        return each_feature_w, each_feature_s, each_label, each_true_label, index


def add_random_noise(semantic_vector, noise_level=0.05):
    noise = np.random.normal(0, noise_level, semantic_vector.shape)
    return semantic_vector + noise


def mask_random_features(semantic_vector, mask_prob=0.15):
    mask = np.random.rand(len(semantic_vector)) < mask_prob
    semantic_vector[mask] = 0  # 将随机选择的元素设置为0
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

    return output


def load_weibo(partial_rate, batch_size):
    test_ratio = 0.2

    # 训练集预处理
    label_set = []
    label_arr = []

    train_label_in = open('/SAPCL-main/data/nlpcc-stance/stance_labels.txt', 'r')
    for line in train_label_in:
        label_set.append([int(x) for x in line.strip().split("\t")])
        label_arr += [int(x) for x in line.strip().split("\t")]
    train_label_in.close()

    train_sampler = None

    train_feature = np.load('/SAPCL-main/data/nlpcc-stance/bert_vectors.npy')
    # shape(2986, 768)
    train_index_set = [i for i in range(0, train_feature.shape[0])]  #length 274 会话集

    output_combined = torch.zeros(0, 768)
    for index in train_index_set:
        output = torch.from_numpy(train_feature[index]).unsqueeze(0)  #shape(1, 768)
        output_combined = torch.cat((output_combined, output), dim=0)

    data = output_combined
    dataset = weibosDataset(data, label_arr)
    # 计算训练集和测试集的划分大小
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    # 划分数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))], dtype=torch.long)

    train_data = train_data.detach()

    test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))], dtype=torch.long)
    test_data = test_data.detach().float()
    # print(test_data.shape)  # 922, 768

    partialY = generate_uniform_cv_candidate_labels(train_labels, partial_rate)
    # partialY = torch.zeros([train_labels.size(0), 3])
    # for i in range(train_labels.size(0)):
    #     partialY[i][train_labels[i]] = 1

    print(partialY)

    # train_data = train_data.to(torch.double)
    # test_data = test_data.to(torch.double)
    #
    # def one_hot(labels, num_classes):
    #     return torch.eye(num_classes)[labels]
    #
    # train_labels_onehot = partialY
    # train_labels_onehot[train_labels_onehot == 0] = -1
    # test_labels_onehot = one_hot(test_labels, 3)
    # train_labels_onehot = train_labels_onehot.to(torch.double).T
    # test_labels_onehot = test_labels_onehot.to(torch.double).T
    #
    # import scipy.io
    #
    # train_data_np = train_data.numpy()
    # train_labels_onehot_np = train_labels_onehot.numpy()
    # test_data_np = test_data.numpy()
    # test_labels_onehot_np = test_labels_onehot.numpy()
    #
    # # 保存为 .mat 文件
    # scipy.io.savemat('weibo_pdataset.mat', {
    #     'train_data': train_data_np,
    #     'train_target': train_labels_onehot_np,
    #     'test_data': test_data_np,
    #     'test_target': test_labels_onehot_np
    # })

    partial_matrix_dataset = weibos_Augmentention(train_data, partialY.float(), train_labels.float())
    test_matrix_dataset = weibos_test_combine(test_data, test_labels.float())

    train_loader = DataLoader(dataset=partial_matrix_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)
    test_loader = DataLoader(dataset=test_matrix_dataset, batch_size=batch_size*16,
                             shuffle=False, num_workers=4, sampler=None)

    # for batch_idx, batch in enumerate(test_loader):
    #     print(batch_idx, len(batch))
    #     print(batch[0].shape, batch[1].shape)

    return train_loader, partialY, train_sampler, test_loader


# if __name__ == '__main__':
#     load_weibo(partial_rate=0.05, batch_size=64)

