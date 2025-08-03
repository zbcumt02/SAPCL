import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist


class SAPCL(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch)
        # momentum encoder
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))  # 缓冲区存储特征表示：8192*128
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue))  # 缓冲区存储伪目标：8192
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # 缓冲区初始化为0，队列的指针位置，用于跟踪队列中下一个插入位置
        self.register_buffer("prototypes", torch.zeros(args.num_class, args.low_dim))  # 缓冲区初始化为0，存储每类别原型特征：100*128
        self.queue = F.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)  # args.moco_m = 0.999

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys  # 将当前批次的特征表示keys从ptr开始存储
        self.queue_pseudo[ptr:ptr + batch_size] = labels  # 将当前批次的标签labels从ptr开始存储至伪标签队列
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer 更新队列指针

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        # 获取当前批次大小
        batch_size_this = x.shape[0]

        # 随机打乱索引
        idx_shuffle = torch.randperm(batch_size_this).cuda()

        # 根据打乱后的索引来排列数据
        idx_unshuffle = torch.argsort(idx_shuffle)

        # 获取当前 GPU 上的打乱索引
        idx_this = idx_shuffle

        return x[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    def forward(self, img_w, im_s=None, partial_Y=None, td_edge=None, bu_edge=None, batch=None, ptr=None,
                args=None, eval_only=False):
        # print(img_w.shape, td_edge.shape, bu_edge.shape, batch.shape, ptr.shape)
        output, q, TD_edge_loss, BU_edge_loss = self.encoder_q(img_w, td_edge, bu_edge, batch, ptr)
        if eval_only:
            return output
        # for testing

        return output, TD_edge_loss, BU_edge_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    device_count = torch.cuda.device_count()
    device_list = [torch.device(f'cuda:{i}') for i in range(device_count)]  # Default to just a single device (cuda:0)

    # Gather the tensors from all devices by moving the original tensor to each device
    tensors_gather = [tensor.to(device) for device in device_list]

    # Concatenate all tensors along dimension 0
    output = torch.cat(tensors_gather, dim=0)
    return output
