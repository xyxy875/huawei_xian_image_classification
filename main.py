# -*- coding: utf-8 -*-
"""
基于 PyTorch resnet50 实现的图片分类代码
原代码地址：https://github.com/pytorch/examples/blob/master/imagenet/main.py
可以与原代码进行比较，查看需修改哪些代码才可以将其改造成可以在 ModelArts 上运行的代码
在ModelArts Notebook中的代码运行方法：
（0）准备数据
大赛发布的公开数据集是所有图片和标签txt都在一个目录中的格式
如果需要使用 torch.utils.data.DataLoader 来加载数据，则需要将数据的存储格式做如下改变：
1）划分训练集和验证集，分别存放为 train 和 val 目录；
2）train 和 val 目录下有按类别存放的子目录，子目录中都是同一个类的图片
prepare_data.py中的 split_train_val 函数就是实现如上功能，建议先在自己的机器上运行该函数，然后将处理好的数据上传到OBS
执行该函数的方法如下：
cd {prepare_data.py所在目录}
python prepare_data.py --input_dir '../datasets/train_data' --output_train_dir '../datasets/train_val/train' --output_val_dir '../datasets/train_val/val'

（1）从零训练
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --deploy_script_path './deploy_scripts' --arch 'resnet50' --num_classes 54 --workers 4 --epochs 6 --pretrained True --seed 0

（2）加载已有模型继续训练
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --deploy_script_path './deploy_scripts' --arch 'resnet50' --num_classes 54 --workers 4 --epochs 6 --seed 0 --resume '../model_snapshots/epoch_0_2.4.pth'

（3）评价单个pth文件
cd {main.py所在目录}
python main.py --data_url '../datasets/train_val' --train_url '../model_snapshots' --arch 'resnet50' --num_classes 54 --seed 0 --eval_pth '../model_snapshots/epoch_5_8.4.pth'
"""
import argparse
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
# os.system('pip install torch==1.1.0')
# os.system('pip install torchvision==0.3.0')

try:
    import moxing as mox
except:
    print('not use moxing')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np
import pretrained_models
import pretrained_models.utils
model_names = sorted(name for name in pretrained_models.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrained_models.__dict__[name]))

from prepare_data import prepare_data_on_modelarts

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', required=True,
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='evaluate model on validation set')
parser.add_argument('--eval_pth', default='', type=str,
                    help='the *.pth model path need to be evaluated on validation set')
parser.add_argument('--pretrained', default='imagenet', type=str,
                    help='use pre-trained model or not')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# These arguments are added for adapting ModelArts
parser.add_argument('--num_classes', required=True, type=int, help='the num of classes which your task should classify')
parser.add_argument('--local_data_root', default='/cache/', type=str,
                    help='a directory used for transfer data between local path and OBS path')
parser.add_argument('--data_url', required=True, type=str, help='the training and validation data path')
parser.add_argument('--test_data_url', default='', type=str, help='the test data path')
parser.add_argument('--data_local', default='', type=str, help='the training and validation data path on local')
parser.add_argument('--test_data_local', default='', type=str, help='the test data path on local')
parser.add_argument('--train_url', required=True, type=str, help='the path to save training outputs')
parser.add_argument('--train_local', default='', type=str, help='the training output results on local')
parser.add_argument('--tmp', default='', type=str, help='a temporary path on local')
parser.add_argument('--deploy_script_path', default='', type=str,
                    help='a path which contain config.json and customize_service.py, '
                         'if it is set, these two scripts will be copied to {train_url}/model directory')
parser.add_argument('--own', default=False, type = bool, help = 'whether use own model')
parser.add_argument('--lr_interv', default={30:0.3, 48:0.1, 60:0.1}, type=dict, help='lr_adjust_interval')
parser.add_argument('--no_freeze', default=8, type=int, help='epoch that unfreeze backbone')
parser.add_argument('--lr_ratio', default=0.1, type=float, help='lr_ratio-backbone/fc')

best_acc1 = 0
MIXUP_EPOCH = 1000

def main():
    args, unknown = parser.parse_known_args()
    args = prepare_data_on_modelarts(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, idx_to_class
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained and not args.own:
        print("=> using pre-trained model '{}'".format(args.arch))
        os.environ['TORCH_MODEL_ZOO'] = '../pre-trained_model/pytorch'
        if not mox.file.exists('../pre-trained_model/pytorch/resnet50-19c8e357.pth'):
            mox.file.copy('s3://ma-competitions-bj4/model_zoo/pytorch/resnet50-19c8e357.pth',
                          '../pre-trained_model/pytorch/resnet50-19c8e357.pth')
            print('copy pre-trained model from OBS to: %s success' %
                  (os.path.abspath('../pre-trained_model/pytorch/resnet50-19c8e357.pth')))
        else:
            print('use exist pre-trained model at: %s' %
                  (os.path.abspath('../pre-trained_model/pytorch/resnet50-19c8e357.pth')))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.pretrained and args.own:
        # print("=> using pre-trained model '{}'".format(args.arch))
        # os.environ['TORCH_MODEL_ZOO'] = '../pre-trained_model/pytorch'
        # os.environ['TORCH_MODEL_ZOO'] = 's3://left-dragon-xian/train_val/'
        # mox.file.copy('s3://left-dragon-xian/train_val/models/se_resnext101_32x4d-3b2fe3d8.pth',
        #               '../pre-trained_model/pytorch/se_resnext101_32x4d-3b2fe3d8.pth')
        # print('copy pre-trained model from OBS to: %s success' %
        #       (os.path.abspath('../pre-trained_model/pytorch/se_resnext101_32x4d-3b2fe3d8.pth')))
        model = pretrained_models.__dict__[args.arch](num_classes=1000,
                                                     pretrained=args.pretrained)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.own:
        # model.last_linear = nn.Linear(model.last_linear.in_features, args.num_classes)
        model.last_linear = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_linear.in_features, args.num_classes)
        )
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
    #                              betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam([{'params': model.module.layer1.parameters(), 'lr': 0},
                                  {'params': model.module.layer2.parameters(), 'lr': 0},
                                  {'params': model.module.ca.parameters(), 'lr': 0},
                                  {'params': model.module.sa.parameters(), 'lr': 0},
                                  {'params': model.module.layer3.parameters(), 'lr': 0},
                                  {'params': model.module.layer4.parameters(), 'lr': 0},
                                  {'params': model.module.last_linear[1].parameters(), 'lr': args.lr}, ],
                                  betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=8, verbose=True,
    #                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # optionally resume from a checkpoint
    if args.resume:
        # if os.path.isfile(args.resume):
        if mox.file.exists(args.resume) and (not mox.file.is_directory(args.resume)):
            if args.resume.startswith('s3://'):
                restore_model_name = args.resume.rsplit('/', 1)[1]
                mox.file.copy(args.resume, '/cache/tmp/' + restore_model_name)
                args.resume = '/cache/tmp/' + restore_model_name
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if args.resume.startswith('/cache/tmp/'):
                os.remove(args.resume)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data_local, 'train1224')
    valdir = os.path.join(args.data_local, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.58071129, 0.52168848, 0.46118198],
                                     std=[0.29525993, 0.3020832, 0.32327064])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08),
            # # transforms.RandomAffine(degrees=45, shear=30, fillcolor=0, resample=PIL.Image.BILINEAR),
            # transforms.RandomRotation(degrees=30),
            # transforms.RandomPerspective(distortion_scale=0.5, p=1, interpolation=3),
            transforms.RandomResizedCrop(224, scale=(0.3, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    # ImageFolder类会将traindir目录下的每个子目录名映射为一个label id，然后将该id作为模型训练时的标签
    # ImageFolder类会将traindir目录下的每个子目录名映射为一个label id，然后将该id作为模型训练时的标签
    # 比如，traindir目录下的子目录名分别是0~53，ImageFolder类将这些目录名当做class_name，再做一次class_to_idx的映射
    # 最终得到这样的class_to_idx：{"0": 0, "1":1, "10":2, "11":3, ..., "19": 11, "2": 12, ...}
    # 其中key是class_name，value是idx，idx就是模型训练时的标签
    # 因此我们在保存训练模型时，需要保存这种idx与class_name的映射关系，以便在做模型推理时，能根据推理结果idx得到正确的class_name
    idx_to_class = OrderedDict()
    for key, value in train_dataset.class_to_idx.items():
        idx_to_class[value] = key

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.eval_pth != '':
        if mox.file.exists(args.eval_pth) and (not mox.file.is_directory(args.eval_pth)):
            if args.eval_pth.startswith('s3://'):
                model_name = args.eval_pth.rsplit('/', 1)[1]
                mox.file.copy(args.eval_pth, '/cache/tmp/' + model_name)
                args.eval_pth = '/cache/tmp/' + model_name
            print("=> loading checkpoint '{}'".format(args.eval_pth))
            if args.gpu is None:
                checkpoint = torch.load(args.eval_pth)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.eval_pth, map_location=loc)
            if args.eval_pth.startswith('/cache/tmp/'):
                os.remove(args.eval_pth)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.eval_pth, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.eval_pth))

        validate(val_loader, model, criterion, args, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if (epoch + 1) % args.print_freq == 0:
            acc1, val_loss, prob_results = validate(val_loader, model, criterion, args, epoch)

            class_prob = prob_results[1].diagonal()
            class_acc = prob_results[0].diagonal()
            weight_cur = 1 + (1 - class_acc) + (1 - class_prob)
            weight_cur = torch.from_numpy(weight_cur).float()
            print('weight_cur (max/min) : {}/{}'.format(weight_cur.max(), weight_cur.min()))
            # weight = weight / weight.max()
            try:
                weight = 0 * weight + 1 * weight_cur
            except:
                weight = weight_cur
            print('weight (max/min) : {}/{}'.format(weight.max(), weight.min()))
            # criterion = nn.CrossEntropyLoss(weight=weight).cuda(args.gpu)

            # remember best acc@1 and save checkpoint
            is_best = False
            best_acc1 = max(acc1.item(), best_acc1)
            pth_file_name = os.path.join(args.train_local, 'epoch_%s_%s.pth'
                                         % (str(epoch + 1), str(round(acc1.item(), 3))))
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'idx_to_class': idx_to_class
                }, is_best, pth_file_name, args)
            # scheduler.step(val_loss)

    if args.epochs >= args.print_freq:
        save_best_checkpoint(best_acc1, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    print('-----------Learning rate------------')
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    if epoch > MIXUP_EPOCH:
        mix_up_flag = True
    else:
        mix_up_flag = False

    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if mix_up_flag:
            inputs, targets_a, targets_b, lam = mixup_data(images.cuda(), target.cuda(), alpha=1.0)  # related to mix_up
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)  # related to mix_up
        else:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # confusion_matrix added by YJ on 20191209
    cm = np.zeros((54, 54))
    prob_matrix = ProbMatrix(num_classes=54)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # compute_confusion _ added by YJ on 20191209
            cm += compute_confusion(output, target) # 需要检查
            prob_matrix.update_one_batch(output=output, target=target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        for i in range(54):
            cm[i] = cm[i] / cm[i].sum(axis=0)

        print('————Weak classes————')
        for i in range(54):
            cm[i] = cm[i] / cm[i].sum(axis=0)
        diag = cm.diagonal()
        diag_sort_idx = np.argsort(diag)
        diag_sorted = diag[diag_sort_idx]
        for i in range(54):
            print('Class:{} Acc:{:.2f}'.format(diag_sort_idx[i], diag_sorted[i]), end='\t')

        print('\n————Similar pairs————')
        confusion_pairs = {}
        for i in range(54):
            for j in range(54):
                if cm[i][j] > 0 and i != j:
                    confusion_pairs['{}--{}: '.format(i, j)] = cm[i][j]
        sorted_confusion_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        for confusion_pair in sorted_confusion_pairs:
            print(confusion_pair[0] + '   {:.3f}'.format(confusion_pair[1]))

        confusion_file_name = os.path.join(args.train_local, 'epoch_{}.npy'.format(epoch))
        np.save(confusion_file_name, cm)
        if args.train_url.startswith('s3'):
            mox.file.copy(confusion_file_name,
                          args.train_url + '/' + os.path.basename(confusion_file_name))
            os.remove(confusion_file_name)

        prob_file_name = os.path.join(args.train_local, 'epoch_prob_{}.npy'.format(epoch))
        prob_results = prob_matrix.compute_results()
        np.save(prob_file_name, prob_results)
        if args.train_url.startswith('s3'):
            print('moving')
            mox.file.copy(prob_file_name,
                          args.train_url + '/' + os.path.basename(prob_file_name))
            os.remove(prob_file_name)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, prob_results


def save_checkpoint(state, is_best, filename, args):
    if not is_best:
        torch.save(state, filename)
        if args.train_url.startswith('s3'):
            mox.file.copy(filename,
                          args.train_url + '/' + os.path.basename(filename))
            os.remove(filename)


def save_best_checkpoint(best_acc1, args):
    best_acc1_suffix = '%s.pth' % str(round(best_acc1, 3))
    pth_files = mox.file.list_directory(args.train_url)
    for pth_name in pth_files:
        if pth_name.endswith(best_acc1_suffix):
            break

    # mox.file可兼容处理本地路径和OBS路径
    if not mox.file.exists(os.path.join(args.train_url, 'model')):
        mox.file.mk_dir(os.path.join(args.train_url, 'model'))

    mox.file.copy(os.path.join(args.train_url, pth_name), os.path.join(args.train_url, 'model/model_best.pth'))
    mox.file.copy(os.path.join(args.deploy_script_path, 'config.json'),
                  os.path.join(args.train_url, 'model/config.json'))
    mox.file.copy(os.path.join(args.deploy_script_path, 'customize_service.py'),
                  os.path.join(args.train_url, 'model/customize_service.py'))
    if mox.file.exists(os.path.join(args.train_url, 'model/config.json')) and \
            mox.file.exists(os.path.join(args.train_url, 'model/customize_service.py')):
        print('copy config.json and customize_service.py success')
    else:
        print('copy config.json and customize_service.py failed')


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

class ProbMatrix():
    def __init__(self, num_classes):
        self.mat = np.zeros((num_classes, num_classes))
        self.mat_num = np.zeros((num_classes, num_classes))
        self.mat_max = np.zeros((num_classes, num_classes))
        self.mat_min = np.ones((num_classes, num_classes))
        self.num_classes = num_classes

    def update_one_batch(self, output, target):
        output_prob = torch.nn.functional.softmax(output, dim=1)  # batch*52
        _, pred = output.topk(1, 1, True, True)
        prob, pred = output_prob.topk(1, 1, True, True)
        pred = pred.squeeze()
        prob = prob.squeeze()
        # pred_class = torch.Tensor([int(idx_to_class[x]) for x in pred.cpu().numpy().tolist()]).int()
        # target_class = torch.Tensor([int(idx_to_class[x]) for x in target.cpu().numpy().tolist()]).int()
        # for i in range(len(pred)):
        #     self.mat[target_class[i], pred_class[i]] += prob[i]
        #     self.mat_num[target_class[i], pred_class[i]] += 1
        #     self.mat_max[target_class[i], pred_class[i]] = max(prob[i], self.mat_max[target_class[i], pred_class[i]])
        #     self.mat_min[target_class[i], pred_class[i]] = min(prob[i], self.mat_min[target_class[i], pred_class[i]])
        for i in range(len(pred)):
            self.mat[target[i], pred[i]] += prob[i]
            self.mat_num[target[i], pred[i]] += 1
            self.mat_max[target[i], pred[i]] = max(prob[i], self.mat_max[target[i], pred[i]])
            self.mat_min[target[i], pred[i]] = min(prob[i], self.mat_min[target[i], pred[i]])

    def compute_results(self):
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if self.mat_num[i, j] != 0:
                    self.mat[i, j] = self.mat[i, j] / self.mat_num[i, j]
        for i in range(self.num_classes):
            self.mat_num[i] = self.mat_num[i] / self.mat_num[i].sum()
        self.mat_min[self.mat_min == 1] = 0
        result_mat = np.stack((self.mat_num, self.mat, self.mat_max, self.mat_min))
        return result_mat

# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.3 ** (epoch // 25))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, args):
    # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.3 ** (epoch // 10))
    if epoch == args.no_freeze:
        for i, param_group in enumerate(optimizer.param_groups):
            if i <= 1:
                param_group['lr'] = 0.3 * args.lr
            elif i <= 5:
                param_group['lr'] = 1 * args.lr
        # for param_group in optimizer.param_groups:
        #     if param_group['lr'] == 0:
        #         param_group['lr'] = args.lr_ratio * args.lr
        # base_lr = optimizer.last_linear
    if epoch in args.lr_interv:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr_interv[epoch] * param_group['lr']


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_confusion(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze()
    pred_class = torch.Tensor([int(idx_to_class[x]) for x in pred.cpu().numpy().tolist()]).int()
    target_class = torch.Tensor([int(idx_to_class[x]) for x in target.cpu().numpy().tolist()]).int()
    cm = confusion_matrix(target_class, pred_class, labels=range(54))
    return cm

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# related to mix_up
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == '__main__':
    main()
