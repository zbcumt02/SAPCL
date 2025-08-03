"""
针对274个会话树中的每个节点提取特征
"""
import json
import torch
import numpy as np
from torch import nn

import warnings
import random
from torch.nn import LSTM
from utils_algo import generate_uniform_cv_candidate_labels

from torch.utils.data import DataLoader, Dataset, random_split

warnings.filterwarnings("ignore", category=UserWarning)

class TreeLSTMCell(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, class_num=5):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.class_num = class_num

        # LSTM 门的线性变换
        self.W_i = torch.nn.Linear(self.input_size, self.hidden_size)
        self.U_i = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_f = torch.nn.Linear(self.input_size, self.hidden_size)
        self.U_f = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_o = torch.nn.Linear(self.input_size, self.hidden_size)
        self.U_o = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_u = torch.nn.Linear(self.input_size, self.hidden_size)
        self.U_u = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=self.hidden_size, kernel_size=(2, self.hidden_size))
        self.hidden_buffer = []
        self.classifier = torch.nn.Linear(self.hidden_size, self.class_num)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.sum = True
        self.node_shape = int(self.hidden_size ** 0.5)

    def forward(self, inputs, tree, current_child_id):
        inputs = torch.Tensor(inputs)
        batch_size = 1

        # 递归计算所有子节点的输出
        children_outputs = [self.forward(inputs, tree[child_id], child_id) for child_id in tree]

        if children_outputs:
            children_states = children_outputs
        else:
            children_states = [(torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))]

        # 计算当前节点的状态
        return self.node_forward(inputs[int(current_child_id), :], children_states)

    def node_forward(self, inputs, children_states):
        batch_size = 1
        K = len(children_states)

        # 如果self.sum为True，则对子节点的隐藏状态进行求和
        if (self.sum):
            average_h = torch.zeros(batch_size, self.hidden_size)
            for index in range(int(K)):
                average_h = average_h + children_states[index][0]
        else:
            # 如果self.sum为False，则对子节点的隐藏状态进行卷积处理
            child_tensor_list = []
            for index in range(int(K)):
                child_tensor_list.append(children_states[index][0])
            child_tensor = torch.stack(child_tensor_list).view(1, 1, K, self.hidden_size)
            child_tensor_conv = self.conv(child_tensor)
            average_h = torch.max(child_tensor_conv, dim=2)[0]
            average_h = average_h.view(1, -1)

        # LSTM的门计算
        i = torch.sigmoid(self.W_i(inputs) + self.U_i(average_h))
        o = torch.sigmoid(self.W_o(inputs) + self.U_o(average_h))
        u = torch.tanh(self.W_u(inputs) + self.U_u(average_h))

        # 计算遗忘门
        sum_f = torch.zeros(batch_size, self.hidden_size)
        for index in range(int(K)):
            f = torch.sigmoid(self.W_f(inputs) + self.U_f(children_states[index][0]))
            sum_f += f * children_states[index][1]

        # 计算细胞状态和隐藏状态
        c = sum_f + i * u
        h = o * torch.tanh(c)

        # 将隐藏状态和细胞状态调整为(M, N)维度
        h_reshaped = h.view(batch_size, self.node_shape, self.node_shape)
        hidden_node = h_reshaped.squeeze(0)

        # 存储输出状态
        self.hidden_buffer.append(hidden_node)  # 将输出添加到缓存中
        return h, c

    def get_hidden_buffer(self, inputs, tree, current_child_id):
        self.hidden_buffer = []
        self.forward(inputs, tree, current_child_id)
        output = torch.stack(self.hidden_buffer, dim=0)
        return output

    def get_node_shape(self):
        return self.node_shape


class tweetsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class tweets_Augmentention(Dataset):
    def __init__(self, dataset, given_label_matrix, true_labels):
        self.dataset = dataset
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        squared_matrix = self.dataset[index] ** 2
        log_matrix = torch.log(self.dataset[index] + 1)
        each_image_w = log_matrix
        each_image_s = squared_matrix
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]

        return each_image_w, each_image_s, each_label, each_true_label, index


class tweets_test_Augmentention(Dataset):
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


def load_tweets(partial_rate, batch_size):
    test_ratio = 0.2
    five_events = ['ch', 'fg', 'gc', 'ow', 'ss']

    cd = TreeLSTMCell()

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

    train_sampler = None

    train_feature = np.concatenate((np.load('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/' + five_events[0] + "vbert2.npy"),
                                    np.load('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/' + five_events[1] + "vbert2.npy"),
                                    np.load('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/' + five_events[2] + "vbert2.npy"),
                                    np.load('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/' + five_events[3] + "vbert2.npy"),
                                    np.load('D:/PycharmProjects/25-3-zb1/SAPCL-main/data/phemes/' + five_events[4] + "vbert2.npy")))
    # shape(274, 103, 768)
    train_index_set = [i for i in range(0, train_feature.shape[0])] #length 274 会话集
    # random.shuffle(train_index_set)

    node_shape = cd.get_node_shape()
    output_combined = torch.zeros(0, node_shape, node_shape)
    for index in train_index_set:
        output = cd.get_hidden_buffer(train_feature[index], whole_tree[index]['0'], 0) #shape(n, 8, 8)
        print(output.shape)
        output_combined = torch.cat((output_combined, output), dim=0)

    data = output_combined  # 4612*8*8
    dataset = tweetsDataset(data, label_arr)
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
    test_data = test_data.detach()
    # print(test_data.shape)  # 922, 16, 16

    partialY = generate_uniform_cv_candidate_labels(train_labels, partial_rate)

    partial_matrix_dataset = tweets_Augmentention(train_data, partialY.float(), train_labels.float())
    test_matrix_dataset = tweets_test_Augmentention(test_data, test_labels.float())

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


if __name__ == '__main__':
    load_tweets(partial_rate=0.05, batch_size=64)

    # 独热编码制造噪声环境
    # train_label_set_onehot = []
    # for label_arr_index in range(len(label_set)):
    #     label_arr = label_set[label_arr_index]
    #     label_one_hot = torch.zeros([len(label_arr), 5])
    #     set_theta = 1 - partial_rate
    #     for label_index in range(len(label_arr)):
    #         label_one_hot[label_index, (label_arr[label_index])] = 1
    #         random_number1 = random.random()
    #         if label_one_hot[label_index, 2] == 1 and random_number1 > set_theta:
    #             label_one_hot[label_index, 3] = 1
    #         random_number2 = random.random()
    #         if label_one_hot[label_index, 3] == 1 and random_number2 > set_theta:
    #             label_one_hot[label_index, 2] = 1
    #     train_label_set_onehot.append(label_one_hot)
