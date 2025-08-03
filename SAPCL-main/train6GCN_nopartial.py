"""

for noise without partial -- EBGCN -- pheme_trees6 -- Accuracy is 59.46% (84.40%)(0.18)(0.52)(0.17)(0.20)

roberta-Accuracy is
deberta-Accuracy is

"""
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
# import torch.utils.data
import torch.utils.data.distributed
import tensorboard_logger as tb_logger
import numpy as np
from model6 import SAPCL
from utils.utils_algo import *
from utils.utils_loss import partial_loss, SupConLoss
from partialHGNN import BERT_PT_HCL
from partialEGCN import EBGCN
from utils.pheme_trees6 import load_phemes_edges
import logging

logging.basicConfig(level=logging.ERROR)

# warnings.filterwarnings("ignore")

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation')
parser.add_argument('--dataset', default='phemes', type=str,
                    choices=['phemes'])
parser.add_argument('--exp-dir', default='experiment/phemes', type=str,
                    help='experiment directory for saving checkpoints and logs, '
                         'phemes')
parser.add_argument('-a', '--arch', metavar='ARCH', default='gnnforbert', choices=['supconGNN2'],
                    help='nnetwork architecture (only supconGNN2 used in SAPCL)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='50,75,100',
                    help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=False,
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=123, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=False,
                    # action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--num-class', default=5, type=int,
                    help='number of class')
parser.add_argument('--low-dim', default=128, type=int,
                    help='embedding dimension')
parser.add_argument('--moco_queue', default=8192, type=int,
                    help='queue size; number of negative samples')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='momentum for updating momentum encoder')
parser.add_argument('--proto_m', default=0.99, type=float,
                    help='momentum for computing the momving average of prototypes')
parser.add_argument('--loss_weight', default=0.5, type=float,
                    help='contrastive loss weight')
parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str,
                    help='pseudo target updating coefficient (phi)')
parser.add_argument('--prot_start', default=1, type=int,
                    help = 'Start Prototype Updating')
parser.add_argument('--partial_rate', default=0.05, type=float,
                    help='ambiguity level (q)')
parser.add_argument('--hierarchical', action='store_true',
                    help='for CIFAR-100 fine-grained training')
parser.add_argument('--TD_droprate', type=float, default=0.2, metavar='TD_droprate',
                    help='drop rate for edges in the top-down propagation graph')
parser.add_argument('--BU_droprate', type=float, default=0.2, metavar='BU_droprate',
                    help='drop rate for edges in the bottom-up dispersion graph')
parser.add_argument('--edge_loss_td', type=float, default=0.2, metavar='edge_loss_td',
                    help='a hyperparameter gamma to weight the unsupervised relation learning loss '
                         'in the top-down propagation graph')
parser.add_argument('--edge_loss_bu', type=float, default=0.2, metavar='edge_loss_bu',
                    help='a hyperparameter gamma to weight the unsupervised relation learning loss '
                         'in the bottom-up dispersion graph')


def main():
    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = False

    model_path = 'ds_{ds}_pr_{pr}_lr_{lr}_ep_{ep}_ps_{ps}_lw_{lw}_pm_{pm}_arch_{arch}_heir_{heir}_sd_{seed}'.format(
                                            ds=args.dataset,
                                            pr=args.partial_rate,
                                            lr=args.lr,
                                            ep=args.epochs,
                                            ps=args.prot_start,
                                            lw=args.loss_weight,
                                            pm=args.proto_m,
                                            arch=args.arch,
                                            seed=args.seed,
                                            heir=args.hierarchical)
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = True
    args.gpu = gpu
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = SAPCL(args, EBGCN)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.dataset == 'phemes':
        train_loader, train_givenY_list, test_loader, arr_index_list = load_phemes_edges(args,
                                                                                         partial_rate=args.partial_rate,
                                                                                         batch_size=args.batch_size)
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")
    # this train loader is the partial label training loader

    print('Calculating uniform targets...')
    train_givenY = torch.cat(train_givenY_list, dim=0)
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float()/tempY
    confidence = confidence.cuda()
    # calculate confidence
    loss_fn = partial_loss(confidence)
    loss_cont_fn = SupConLoss()
    # set loss functions (with pseudo-targets maintained)

    if args.gpu == 0:
        logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir,'tensorboard'), flush_secs=2)
    else:
        logger = None

    print('\nStart Training\n')

    best_acc = 0
    mmc = 0  #mean max confidence
    for epoch in range(args.start_epoch, args.epochs):
        is_best = False
        start_upd_prot = epoch >= args.prot_start

        adjust_learning_rate(args, optimizer, epoch)
        train(train_loader, model, arr_index_list, loss_fn, loss_cont_fn, optimizer, epoch, args, logger, start_upd_prot)
        loss_fn.set_conf_ema_m(epoch, args)
        # reset phi

        acc_test = test(model, test_loader, args, epoch, logger)
        mmc = loss_fn.confidence.max(dim=1)[0].mean()

        with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
            f.write('Epoch {}: Acc {}, Best Acc {}. (lr {}, MMC {})\n'.format(epoch
                , acc_test, best_acc, optimizer.param_groups[0]['lr'], mmc))
        if acc_test > best_acc:
            best_acc = acc_test
            is_best = True

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
            best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))


def train(train_loader, model, arr_index_list, loss_fn, loss_cont_fn,
          optimizer, epoch, args, tb_logger, start_upd_prot=False):
    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_td_log = AverageMeter('Loss@TD', ':2.2f')
    loss_bu_log = AverageMeter('Loss@BU', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, loss_cls_log, loss_td_log, loss_bu_log],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    i = 0
    # for i, (datas_w, datas_s, labels, true_labels, index) in enumerate(train_loader):
    for batch_data in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)
        X_w, X_s, Y, edge, index_graph, batch, ptr = (batch_data.w.cuda(), batch_data.s.cuda(), batch_data.y.cuda(),
                                                      batch_data.edge_index, batch_data.idx.cuda(),
                                                      batch_data.batch.cuda(), batch_data.ptr.cuda())
        Y_true = batch_data.ground_truth.long().detach().cuda()
        # for showing training accuracy and will not be used when training
        td_edge = top_down_graph(edge, args.TD_droprate)
        bu_edge = bottom_up_graph(edge, args.BU_droprate)
        td_edge, bu_edge = td_edge.cuda(), bu_edge.cuda()
        index = torch.zeros(0, dtype=torch.long)
        for i in index_graph:
            start_idx = arr_index_list[i]
            end_idx = arr_index_list[i + 1]
            selected_numbers = torch.arange(start_idx, end_idx)

            index = torch.cat((index, selected_numbers))
        # print(X_w.shape, X_s.shape, Y.shape, td_edge.shape, bu_edge.shape, index.shape, index, Y_true.shape)
        # torch.Size([75, 768]) torch.Size([75, 768]) torch.Size([75, 5]) torch.Size([2, 56]) torch.Size([2, 56])
        # torch.Size([4]) torch.Size([75])

        (cls_out, TD_edge_loss, BU_edge_loss) = model(X_w, X_s, Y, td_edge, bu_edge, batch, ptr, args)
        batch_size = cls_out.shape[0]
        # print("batch_size: ", batch_size)

        # classification loss
        loss_cls = loss_fn(cls_out, index)
        # graph loss
        # TD_edge_loss, BU_edge_loss
        loss = (loss_cls + args.edge_loss_td * TD_edge_loss + args.edge_loss_bu * BU_edge_loss)
        loss_cls_log.update(loss_cls.item())
        loss_td_log.update(TD_edge_loss.item())
        loss_bu_log.update(BU_edge_loss.item())

        # log accuracy
        acc = accuracy(cls_out, Y_true)[0]
        acc_cls.update(acc[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()
        i += 1
        if i % args.print_freq == 0:
            progress.display(i)

    if args.gpu == 0:
        tb_logger.log_value('Train Acc', acc_cls.avg, epoch)
        tb_logger.log_value('Classification Loss', loss_cls_log.avg, epoch)
        tb_logger.log_value('Top2Down Loss', loss_td_log.avg, epoch)
        tb_logger.log_value('Bottom2Up Loss', loss_bu_log.avg, epoch)


def test(model, test_loader, args, epoch, tb_logger):
    with torch.no_grad():
        print('==> Evaluation...')
        model.eval()
        top1_acc = AverageMeter("Top1")
        top2_acc = AverageMeter("Top2")
        f1_macro = AverageMeter("F1-macro-score")
        f1_weight = AverageMeter("F1-weight-score")
        P_score = AverageMeter("Precision-score")
        R_score = AverageMeter("Recall-score")
        # print(test_loader.batch_size)
        # for batch_idx, batch in enumerate(test_loader):
        #     print(batch_idx)
        #     print(batch[0].shape, batch[1].shape)
        # return 1
        for batch_data in test_loader:
            data, edge, labels, batch, ptr = (batch_data.x.cuda(), batch_data.edge_index, batch_data.y.cuda(),
                                              batch_data.batch.cuda(), batch_data.ptr.cuda())
            td_edge = top_down_graph(edge)
            bu_edge = bottom_up_graph(edge)
            td_edge, bu_edge = td_edge.cuda(), bu_edge.cuda()
            # print(data.shape, edge.shape, labels.shape, td_edge.shape, bu_edge.shape, batch.shape, ptr.shape)
            # torch.Size([226, 768]) torch.Size([2, 210]) torch.Size([226])
            # torch.Size([2, 210]) torch.Size([2, 210]) torch.Size([226]) torch.Size([17])
            outputs = model(img_w=data, td_edge=td_edge, bu_edge=bu_edge, batch=batch, ptr=ptr, args=args, eval_only=True)
            acc1, acc2 = accuracy(outputs, labels, topk=(1, 2))
            f1macro, f1weight, precision, recall = f1_compute(outputs, labels)
            top1_acc.update(acc1[0])
            top2_acc.update(acc2[0])
            f1_macro.update(f1macro[0])
            f1_weight.update(f1weight[0])
            P_score.update(precision[0])
            R_score.update(recall[0])

            # 在单GPU上，直接获取最终的平均值
            print('Accuracy is %.2f%% (%.2f%%)(%.2f)(%.2f)(%.2f)(%.2f)' %
                  (top1_acc.avg, top2_acc.avg, f1_macro.avg, f1_weight.avg, P_score.avg, R_score.avg))

            # 如果是主GPU（通常是0号GPU），记录到TensorBoard
            if args.gpu == 0:
                tb_logger.log_value('Top1 Acc', top1_acc.avg, epoch)
                tb_logger.log_value('Top2 Acc', top2_acc.avg, epoch)
                tb_logger.log_value('F1 macro', f1_macro.avg, epoch)
                tb_logger.log_value('F1 micro', f1_weight.avg, epoch)
                tb_logger.log_value('F1 micro', P_score.avg, epoch)
                tb_logger.log_value('F1 micro', R_score.avg, epoch)

        return top1_acc.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


if __name__ == '__main__':
    main()
