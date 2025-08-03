import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def f1_compute(pred, labels):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()

    # 计算 f1_macro 和 f1_micro
    f1_macro = f1_score(labels, pred, average='macro', zero_division=1)  # 宏平均 F1
    f1_weighted = f1_score(labels, pred, average='weighted', zero_division=1)
    precision = precision_score(labels, pred, average='weighted', zero_division=1)  # Precision (p) for macro average
    recall = recall_score(labels, pred, average='weighted', zero_division=1)  # Recall (r) for macro average

    f1_macro, f1_weighted, precision, recall = (torch.tensor(f1_macro).unsqueeze(0),
                                                torch.tensor(f1_weighted).unsqueeze(0),
                                                torch.tensor(precision).unsqueeze(0),
                                                torch.tensor(recall).unsqueeze(0))
    return f1_macro, f1_weighted, precision, recall


def hamming_loss(output, labels):
    pred = output.cpu().numpy().flatten()  # 转为一维数组
    labels = labels.cpu().numpy().flatten()  # 转为一维数组
    loss = np.sum(pred != labels) / sum(labels)
    return loss


def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples


def sigmoid_rampup(current, rampup_length, exp_coe=5.0):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-exp_coe * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    if torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = partial_rate
    # print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    # print("Finish Generating Candidate Label Sets!\n")
    return partialY.float()


def generate_uniform_edges_candidate_labels(K, train_labels, partial_rate=0.1):

    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = partial_rate
    # print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    # print("Finish Generating Candidate Label Sets!\n")
    return partialY.float()


def top_down_graph(edge_index, td_droprate = 0):
    if td_droprate > 0:
        row = list(edge_index[0])
        col = list(edge_index[1])
        length = len(row)
        poslist = random.sample(range(length), int(length * (1 - td_droprate)))
        poslist = sorted(poslist)
        row = list(np.array(row)[poslist])
        col = list(np.array(col)[poslist])
        td_edgeindex = [row, col]
    else:
        td_edgeindex = edge_index

    return torch.tensor(td_edgeindex, dtype=torch.long)


def bottom_up_graph(edge_index, bu_droprate = 0):
    burow = list(edge_index[1])
    bucol = list(edge_index[0])
    if bu_droprate > 0:
        length = len(burow)
        poslist = random.sample(range(length), int(length * (1 - bu_droprate)))
        poslist = sorted(poslist)
        row = list(np.array(burow)[poslist])
        col = list(np.array(bucol)[poslist])
        bu_edgeindex = [row, col]
    else:
        bu_edgeindex = [burow, bucol]

    return torch.tensor(bu_edgeindex, dtype=torch.long)



