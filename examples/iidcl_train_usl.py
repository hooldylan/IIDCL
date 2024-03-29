# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import os
os.chdir(sys.path[0])  ##### solve the problem that the python path is not true
sys.path.append("../")
sys.path.append("../../")
import collections
import time
from datetime import timedelta
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.cluster import DBSCAN

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.trainers_support import ClusterContrastTrainerSupport

from clustercontrast.losses.cam_loss import InterCamProxy

start_epoch = best_mAP = 0


def get_data(args, name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):

    if args.self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(args, dataset, height, width, batch_size, workers, testset=None):
    if args.self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args, start_epoch=0):
    if 'resnet' in args.arch:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                num_classes=0, pooling_type=args.pooling_type,pretrained_path=args.pretrained_path)
    else:
        model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate
                , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem)
        # checkpoint = load_checkpoint(args.resume)
        # copy_state_dict(checkpoint['state_dict'], model, strip='module.')
        if start_epoch == -1:
            start_epoch = 0
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model, start_epoch

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    try:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(log_dir=osp.join(args.logs_dir, 'tfboard'))
    except:
        try:
            from tensorboardX import SummaryWriter
            logger = SummaryWriter(log_dir=osp.join(args.logs_dir, 'tfboard'))
        except:
            logger = None

    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args, args.dataset, args.data_dir)
    test_loader = get_test_loader(args, dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model, start_epoch = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    print('optimizer: %s'%(args.optimizer))
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    if args.use_support:
        trainer = ClusterContrastTrainerSupport(args, model)
    else:
        trainer = ClusterContrastTrainer(args, model)

    for epoch in range(start_epoch, args.epochs):
        if args.resume != '' and epoch == start_epoch:
            cmc_scores, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

        

        if epoch % (args.cluster_step) == 0 or epoch == start_epoch:
            with torch.no_grad():
                print('==> Create pseudo labels for unlabeled data')
                cluster_loader = get_test_loader(args, dataset, args.height, args.width,
                                                args.batch_size, args.workers, testset=sorted(dataset.train))

                features, _ = extract_features(model, cluster_loader, print_freq=50)
                features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
                rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

                if epoch == start_epoch:
                    eps = args.eps
                    print('Clustering criterion: eps: {:.3f}'.format(eps))
                    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

                pseudo_labels = cluster.fit_predict(rerank_dist)
                num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
                #pseudo_labels, num_cluster = compute_pseudo_labels(features, cluster, args.k1)

            # generate new dataset with pseudo-labels

            idxs, cids, pids = [], [], []
            
            pseudo_labeled_dataset = []
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
                if label != -1:
                    pseudo_labeled_dataset.append((fname, label.item(), cid))
                    idxs.append(i)
                    cids.append(cid)
                    pids.append(label.item())
            # generate new dataset and calculate cluster centers
            @torch.no_grad()
            def generate_cluster_features(labels, features):
                centers = collections.defaultdict(list)
                for i, label in enumerate(labels):
                    if label == -1:
                        continue
                    centers[labels[i]].append(features[i])

                centers = [
                    torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                ]

                centers = torch.stack(centers, dim=0)
                return centers

            cluster_features = generate_cluster_features(pseudo_labels, features)
            del cluster_loader
            sample_type = args.sample_type

            memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                                    momentum=args.momentum, use_hard=args.use_hard, sample_type=sample_type).cuda()
            print("Epoch {} uses sample_type: {}".format(epoch, sample_type))
            memory.features = F.normalize(cluster_features, dim=1).cuda()

            trainer.memory = memory

            # reindex
            idxs, cids, pids = np.asarray(idxs), np.asarray(cids), np.asarray(pids)
            features = features[idxs, :]

            print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

            train_loader = get_train_loader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset)

            centroids_g = []
            cam_proxy, cam_proxy_pids, cam_proxy_cids = [], [], []
            for pid in sorted(np.unique(pids)):  # loop all pids
                idxs_p = np.where(pids == pid)[0]
                # 全局中心
                # .mean(0) 每一列的平均值
                centroids_g.append(features[idxs_p].mean(0))

                for cid in sorted(np.unique(cids[idxs_p])):  # loop all cids for pid
                    idxs_c = np.where(cids == cid)[0]
                    # numpy.intersect1d()函数查找两个数组的交集，并返回两个输入数组中都有序的，唯一的值。
                    idxs_cp = np.intersect1d(idxs_p, idxs_c)
                    # 全局相机代理，添加
                    cam_proxy.append(features[idxs_cp].mean(0))
                    # 代理的pid
                    cam_proxy_pids.append(pid)
                    # 代理的cid
                    cam_proxy_cids.append(cid)

            del features
            # 将某一个维度除以那个维度对应的范数(默认是2范数)
            centroids_g = F.normalize(torch.stack(centroids_g), p=2, dim=1)
            # 得到的是一个Tensor的张量（向量）
            #model.module.classifier.weight.data[:num_cluster].copy_(centroids_g)
            # 计算相机间的损失
            memory_cam = InterCamProxy(centroids_g.size(1), len(cam_proxy_pids)).cuda()
             # 相机代理的平均值
            memory_cam.proxy = F.normalize(torch.stack(cam_proxy), p=2, dim=1).cuda()
            memory_cam.pids = torch.Tensor(cam_proxy_pids).long().cuda()
            memory_cam.cids = torch.Tensor(cam_proxy_cids).long().cuda()
            trainer.memory_cam = memory_cam
        train_loader.new_epoch()
        
        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader), logger=logger)

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            cmc_scores, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            if epoch % args.save_step == 0 or (epoch == args.epochs - 1):
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_{}.pth.tar'.format(epoch)))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))
    if logger is not None:
        logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ISE for unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--cluster-step', type=int, default=1,
                        help="cluster for several epochs")   

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--resume', type=str, default='', metavar='PATH')#../model/vit_small_ics_cfs_lup.pth
    parser.add_argument('--pretrained_path', type=str, default='')

    #vit
    parser.add_argument('--drop-path-rate', type=float, default=0.3)
    parser.add_argument('--hw-ratio', type=int, default=1)
    parser.add_argument('--self-norm', action="store_true")
    parser.add_argument('--conv-stem', action="store_true")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--save-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--loss_weight', type=float, default=1.0,
                        help="loss weight for cluster contrast")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem', choices=['avg', 'gem'])
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--sample-type', type=str, default='hard', choices=['hard', 'ori'])

    # support sample parameters
    parser.add_argument('--use_support', action="store_true")
    parser.set_defaults(use_support=True)
    parser.add_argument('--support_base_lambda', type=float, default=1.0)
    parser.add_argument('--topk', type=int, default=1, help='find top-k nearest center')

    # label_preserving (LP) loss parameters
    parser.add_argument('--temp_lp_loss', type=float, default=0.6,
                        help="temperature for LP loss")
    parser.add_argument('--lp_loss_weight', type=float, default=0.1,
                        help="loss weight for LP loss")

    #cam
    parser.add_argument('--lam-cam', type=float, default=1.0,
                        help="weighting parameter of inter-camera contrastive loss")
    main()
