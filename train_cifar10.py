# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47
modified by @moorekevin

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from adan_pytorch import Adan
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import platform

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer
from models.mobilevit import mobilevit_xxs

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')  # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false',
                    help='disable use randomaug')
parser.add_argument('--noamp', action='store_true',
                    help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true',
                    help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int,
                    help="parameter for convmixer")

parser.add_argument('--weight_decay', default=0.02,
                    type=float, help='weight decay for Adan')
parser.add_argument('--adan_beta1', default=0.02,
                    type=float, help='beta1 for Adan')
parser.add_argument('--adan_beta2', default=0.08,
                    type=float, help='beta2 for Adan')
parser.add_argument('--adan_beta3', default=0.01,
                    type=float, help='beta3 for Adan')

args = parser.parse_args()

if __name__ == "__main__":

    # take in args
    usewandb = ~args.nowandb
    if usewandb:
        import wandb
        if args.opt == "adan":
            watermark = (f"{args.net}_lr{args.lr:.0e}_"
                         f"b1={args.adan_beta1:.3f}_"
                         f"b2={args.adan_beta2:.3f}_"
                         f"b3={args.adan_beta3:.3f}_"
                         f"wd{args.weight_decay:.3f}")
        else:
            watermark = (f"{args.net}_lr{args.lr:.0e}_"
                         f"opt={args.opt}")
        wandb.init(project="cifar10-challange",
                   name=watermark, config=args)
        # wandb.config.update(args)
        # For overwriting arguments in a wandb sweep
        config = wandb.config
        args.net = config.net
        args.lr = config.lr
        args.opt = config.opt
        args.adan_beta1 = config.adan_beta1
        args.adan_beta2 = config.adan_beta2
        args.adan_beta3 = config.adan_beta3
        args.weight_decay = config.weight_decay
        args.n_epochs = wandb.config.n_epochs

    bs = int(args.bs)
    imsize = int(args.size)

    device = (
        'mps' if torch.backends.mps.is_available() and platform.system() == 'Darwin'
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

    if device == 'cuda':
        from torch.cuda.amp import autocast, GradScaler
        def autocast_context(): return autocast(enabled=use_amp)
    else:
        from torch.amp import autocast, GradScaler
        def autocast_context(): return autocast(device_type=device, enabled=use_amp)

    use_amp = not args.noamp and device == 'cuda'

    aug = args.noaug

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    if args.net == "vit_timm":
        size = 384
    else:
        size = imsize

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:
        N = 2
        M = 14
        transform_train.transforms.insert(0, RandAugment(N, M))

    # Prepare dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model factory..
    print('==> Building model..')
    # net = VGG('VGG19')
    if args.net == 'res18':
        net = ResNet18()
    elif args.net == 'vgg':
        net = VGG('VGG19')
    elif args.net == 'res34':
        net = ResNet34()
    elif args.net == 'res50':
        net = ResNet50()
    elif args.net == 'res101':
        net = ResNet101()
    elif args.net == "convmixer":
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(256, 16, kernel_size=args.convkernel,
                        patch_size=1, n_classes=10)
    elif args.net == "mlpmixer":
        from models.mlpmixer import MLPMixer
        net = MLPMixer(
            image_size=32,
            channels=3,
            patch_size=args.patch,
            dim=512,
            depth=6,
            num_classes=10
        )
    elif args.net == "vit_small":
        from models.vit_small import ViT
        net = ViT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.net == "vit_tiny":
        from models.vit_small import ViT
        net = ViT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=4,
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.net == "simplevit":
        from models.simplevit import SimpleViT
        net = SimpleViT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=6,
            heads=8,
            mlp_dim=512
        )
    elif args.net == "vit":
        # ViT for cifar10
        net = ViT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.net == "vit_timm":
        import timm
        net = timm.create_model("vit_base_patch16_384", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.net == "cait":
        from models.cait import CaiT
        net = CaiT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=6,   # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05
        )
    elif args.net == "cait_small":
        from models.cait import CaiT
        net = CaiT(
            image_size=size,
            patch_size=args.patch,
            num_classes=10,
            dim=int(args.dimhead),
            depth=6,   # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05
        )
    elif args.net == "swin":
        from models.swin import swin_t
        net = swin_t(window_size=args.patch,
                     num_classes=10,
                     downscaling_factors=(2, 2, 2, 1))
    elif args.net == "mobilevit":
        net = mobilevit_xxs(size, 10)
    else:
        raise ValueError(f"'{args.net}' is not a valid model")

    # For Multi-GPU
    if 'cuda' in device:
        print(device)
        if args.dp:
            print("using data parallel")
            net = torch.nn.DataParallel(net)  # make parallel
            cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            'checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # Loss is CE
    criterion = nn.CrossEntropyLoss()

    if args.opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(
            net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    elif args.opt == "adan":
        optimizer = Adan(
            net.parameters(), lr=args.lr,
            betas=(args.adan_beta1, args.adan_beta2, args.adan_beta3),
            weight_decay=args.weight_decay)

    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.n_epochs)

    # Training
    scaler = GradScaler(enabled=use_amp)

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Train with amp
            with autocast_context():
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return train_loss/(batch_idx+1)

    # Validation

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {"model": net.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict()}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/'+args.net +
                       '-{}-ckpt.t7'.format(args.patch))
            best_acc = acc

        os.makedirs("log", exist_ok=True)
        content = time.ctime() + ' ' + \
            f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
        print(content)
        with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
            appender.write(content + "\n")
        return test_loss, acc

    list_loss = []
    list_acc = []

    if usewandb:
        wandb.watch(net)

    net.to(device)
    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        trainloss = train(epoch)
        val_loss, acc = test(epoch)

        scheduler.step()

        list_loss.append(val_loss)
        list_acc.append(acc)

        # Log training..
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
                       "epoch_time": time.time()-start})

        # Write out csv..
        with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss)
            writer.writerow(list_acc)
        print(list_loss)

    # writeout wandb
    if usewandb:
        wandb.save("wandb_{}.h5".format(args.net))
