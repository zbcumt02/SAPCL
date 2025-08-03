import sys
import os
import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy
from torch.nn import BatchNorm1d
from collections import OrderedDict
import torch.nn as nn

sys.path.append(os.getcwd())


class TD_GCN(torch.nn.Module):
    def __init__(self, input_feats=768, hidden_feats=64, output_feats=64, dropout=0.5):
        super(TD_GCN, self).__init__()
        self.conv1 = GCNConv(input_feats, hidden_feats)
        self.conv2 = GCNConv(input_feats + hidden_feats, output_feats)
        self.device = torch.device('cuda:{}'.format(int(0)) if torch.cuda.is_available() else 'cpu')
        self.num_features_list = [hidden_feats * r for r in [1]]

        def creat_network(self, name):
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                layer_list[name + 'conv{}'.format(l)] = torch.nn.Conv1d(
                    in_channels=hidden_feats,
                    out_channels=hidden_feats,
                    kernel_size=1,
                    bias=False)
                layer_list[name + 'norm{}'.format(l)] = torch.nn.BatchNorm1d(num_features=hidden_feats)
                layer_list[name + 'relu{}'.format(l)] = torch.nn.LeakyReLU()
            layer_list[name + 'conv_out'] = torch.nn.Conv1d(in_channels=hidden_feats,
                                                            out_channels=1,
                                                            kernel_size=1)
            return layer_list

        self.sim_network = torch.nn.Sequential(creat_network(self, 'sim_val'))
        mod_self = self
        mod_self.num_features_list = [hidden_feats]
        self.W_mean = torch.nn.Sequential(creat_network(mod_self, 'W_mean'))
        self.W_bias = torch.nn.Sequential(creat_network(mod_self, 'W_bias'))
        self.B_mean = torch.nn.Sequential(creat_network(mod_self, 'B_mean'))
        self.B_bias = torch.nn.Sequential(creat_network(mod_self, 'B_bias'))
        self.fc1 = torch.nn.Linear(hidden_feats, 2, bias=False)
        self.fc2 = torch.nn.Linear(hidden_feats, 2, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.eval_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.bn1 = BatchNorm1d(hidden_feats + input_feats)

    def forward(self, data, edges, batch, ptr):
        x, edge_index = data, edges
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        edge_loss, edge_pred = self.edge_infer(x, edge_index)

        rootindex = ptr[:-1]
        # print(rootindex, data.shape, x2.shape, batch.shape)
        # tensor([ 0, 37, 48, 57], device='cuda:0') torch.Size([101, 768]) torch.Size([101, 64])
        # tensor([ 0, 13, 26, 39], device='cuda:0') torch.Size([75, 768]) torch.Size([75, 64])
        root_extend = torch.zeros(len(batch), x1.size(1)).to(self.device)
        # print(x.shape, root_extend.shape)
        batch_size = max(batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight=edge_pred)
        x = F.relu(x)
        root_extend = torch.zeros(len(batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        # x = scatter_mean(x, data.batch, dim=0)
        return x, edge_loss

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        x_i = x[row - 1].unsqueeze(2)
        x_j = x[col - 1].unsqueeze(1)
        x_ij = torch.abs(x_i - x_j)
        sim_val = self.sim_network(x_ij)
        edge_pred = self.fc1(sim_val)
        edge_pred = torch.sigmoid(edge_pred)
        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        logit_mean = w_mean * sim_val + b_mean
        logit_var = torch.log((sim_val ** 2) * torch.exp(w_bias) + torch.exp(b_bias))
        edge_y = torch.normal(logit_mean, logit_var)
        edge_y = torch.sigmoid(edge_y)
        edge_y = self.fc2(edge_y)
        logp_x = F.log_softmax(edge_pred, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)
        edge_loss = self.eval_loss(logp_x, p_y)
        return edge_loss, torch.mean(edge_pred, dim=-1).squeeze(1)


class BU_GCN(torch.nn.Module):
    def __init__(self, input_feats=768, hidden_feats=64, output_feats=64, dropout=0.5):
        super(BU_GCN, self).__init__()
        self.conv1 = GCNConv(input_feats, hidden_feats)
        self.conv2 = GCNConv(input_feats + hidden_feats, output_feats)
        self.device = torch.device('cuda:{}'.format(int(0)) if torch.cuda.is_available() else 'cpu')
        self.num_features_list = [hidden_feats * r for r in [1]]

        def creat_network(self, name):
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                layer_list[name + 'conv{}'.format(l)] = torch.nn.Conv1d(
                    in_channels=hidden_feats,
                    out_channels=hidden_feats,
                    kernel_size=1,
                    bias=False)
                layer_list[name + 'norm{}'.format(l)] = torch.nn.BatchNorm1d(num_features=hidden_feats)
                layer_list[name + 'relu{}'.format(l)] = torch.nn.LeakyReLU()
            layer_list[name + 'conv_out'] = torch.nn.Conv1d(in_channels=hidden_feats,
                                                            out_channels=1,
                                                            kernel_size=1)
            return layer_list

        self.sim_network = torch.nn.Sequential(creat_network(self, 'sim_val'))
        mod_self = self
        mod_self.num_features_list = [hidden_feats]  #
        self.W_mean = torch.nn.Sequential(creat_network(mod_self, 'W_mean'))
        self.W_bias = torch.nn.Sequential(creat_network(mod_self, 'W_bias'))
        self.B_mean = torch.nn.Sequential(creat_network(mod_self, 'B_mean'))
        self.B_bias = torch.nn.Sequential(creat_network(mod_self, 'B_bias'))
        self.fc1 = torch.nn.Linear(hidden_feats, 2, bias=False)
        self.fc2 = torch.nn.Linear(hidden_feats, 2, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.eval_loss = torch.nn.KLDivLoss(reduction='batchmean')  # mean
        self.Rbn1 = BatchNorm1d(hidden_feats + input_feats)

    def forward(self, data, edges, batch, ptr):
        x, edge_index = data, edges
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        edge_loss, edge_pred = self.edge_infer(x, edge_index)

        rootindex = ptr[:-1]

        root_extend = torch.zeros(len(batch), x1.size(1)).to(self.device)
        batch_size = max(batch) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_pred)
        x = F.relu(x)
        root_extend = torch.zeros(len(batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = torch.cat((x, root_extend), 1)

        # x = scatter_mean(x, data.batch, dim=0)
        return x, edge_loss

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        x_i = x[row - 1].unsqueeze(2)
        x_j = x[col - 1].unsqueeze(1)
        x_ij = torch.abs(x_i - x_j)
        sim_val = self.sim_network(x_ij)
        edge_pred = self.fc1(sim_val)
        edge_pred = torch.sigmoid(edge_pred)

        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        logit_mean = w_mean * sim_val + b_mean
        logit_var = torch.log((sim_val ** 2) * torch.exp(w_bias) + torch.exp(b_bias))

        edge_y = torch.normal(logit_mean, logit_var)
        edge_y = torch.sigmoid(edge_y)
        edge_y = self.fc2(edge_y)

        logp_x = F.log_softmax(edge_pred, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)
        edge_loss = self.eval_loss(logp_x, p_y)
        return edge_loss, torch.mean(edge_pred, dim=-1).squeeze(1)


class SAPCL_GCN(torch.nn.Module):
    def __init__(self, name='SAPCL_GCN', head='mlp', feat_dim=128, num_class=5, input_feats=768,
                 hidden_feats=64, dim_in=512, output_feats=64):
        super(SAPCL_GCN, self).__init__()
        self.TD_GCN = TD_GCN(input_feats=input_feats, hidden_feats=hidden_feats, output_feats=output_feats)
        self.BU_GCN = BU_GCN(input_feats=input_feats, hidden_feats=hidden_feats, output_feats=output_feats)
        self.fc = torch.nn.Linear((hidden_feats + output_feats) * 2, num_class)

        if head == 'linear':
            self.head = nn.Linear(hidden_feats, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear((hidden_feats + output_feats) * 2, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, data, td_edge, bu_edge, batch, ptr):
        TD_x, TD_edge_loss = self.TD_GCN(data, td_edge, batch, ptr)
        BU_x, BU_edge_loss = self.BU_GCN(data, bu_edge, batch, ptr)

        out = torch.cat((BU_x, TD_x), 1)  # [m, 256]
        logits = self.fc(out)
        logits = F.log_softmax(logits, dim=1)  # [m, num_class]
        feat_c = self.head(out)
        # print(logits.shape, F.normalize(feat_c, dim=1).shape, TD_edge_loss, BU_edge_loss)
        # torch.Size([101, 5]) torch.Size([101, 128])
        # tensor(0.05, device='cuda:0', grad_fn=<DivBackward0>) tensor(0.02, device='cuda:0', grad_fn=<DivBackward0>)
        return logits, F.normalize(feat_c, dim=1), TD_edge_loss, BU_edge_loss


