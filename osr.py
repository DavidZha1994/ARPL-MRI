import os
import sys
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from models import gan
from models.models import classifier32, classifier32ABN
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR, ixi_slice_OSR
from utils import Logger, save_networks, load_networks, save_GAN, mkdir_if_missing
from core import train, train_cs, test

import datetime

import torchvision
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='mnist',
                    help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet | ixi_slice")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')

# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1,
                    help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002,
                    help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1,
                    help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1,
                    help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--usecpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true',
                    help="Confusing Sample", default=False)

args = parser.parse_args()
options = vars(args)



def main_worker(options, current_time):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['usecpu']:
        use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'],
                         batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'],
                           batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'],
                        batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'],
                           batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader
        out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=options['dataroot'],
                                batch_size=options['batch_size'], img_size=options['img_size'])
        outloader = out_Data.test_loader
    elif 'tiny_imagenet' in options['dataset']:
        Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'],
                                 batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    else:
        Data = ixi_slice_OSR(known=options['known'], dataroot=options['dataroot'],
                             batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
        
    options['num_classes'] = Data.num_classes
    options['legendname'] = Data.legendname
    feat_dim = 2
    
    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu':  use_gpu
        }
    )

    # Model
    #print("Creating model: {}".format(options['model']))
    if options['cs']:
        net = classifier32ABN(
            num_classes=options['num_classes'], use_gpu=options['use_gpu'])
    else:
        net = classifier32(
            num_classes=options['num_classes'], use_gpu=options['use_gpu'])

    if options['cs']:
        print("Creating GAN")
        nz, ns = options['nz'], 1
        if 'tiny_imagenet' or 'ixi_slice' in options['dataset']:
            netG = gan.Generator(1, nz, 64, 3)
            netD = gan.Discriminator(1, 3, 64)
        else:
            netG = gan.Generator32(1, nz, 64, 3)
            netD = gan.Discriminator32(1, 3, 64)
        fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
        criterionD = nn.BCELoss()

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()
        if options['cs']:
            netG = nn.DataParallel(netG, device_ids=[i for i in range(
                len(options['gpu'].split(',')))]).cuda()
            netD = nn.DataParallel(netD, device_ids=[i for i in range(
                len(options['gpu'].split(',')))]).cuda()
            fixed_noise.cuda()

    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if options['dataset'] == 'cifar100':
        model_path += '_50'
        file_name = '{}_{}_{}_{}_{}'.format(
            options['model'], options['loss'], 50, options['item'], options['cs'])
    else:
        file_name = '{}_{}_{}_{}'.format(
            options['model'], options['loss'], options['item'], options['cs'])

    if options['eval']:
        net, criterion = load_networks(
            net, model_path, file_name, criterion=criterion)
        results = test(current_time, net, criterion, testloader,
                       outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(
            results['ACC'], results['AUROC'], results['OSCR']))

        return results

    params_list = [{'params': net.parameters()},
                   {'params': criterion.parameters()}]

    if options['dataset'] == 'tiny_imagenet' or options['dataset'] == 'ixi_slice':
        optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    else:
        optimizer = torch.optim.SGD(
            params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)
    if options['cs']:
        optimizerD = torch.optim.Adam(
            netD.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(
            netG.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 90, 120])

    start_time = time.time()

    log_dir=f'./logs/{current_time}'
    with SummaryWriter(log_dir = log_dir, comment='ixi_slice') as writer:
        for epoch in range(options['max_epoch']):
            print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

            if options['cs']:
                train_cs(net, netD, netG, criterion, criterionD,
                        optimizer, optimizerD, optimizerG,
                        trainloader, epoch=epoch, **options)

            loss_all = train(current_time, net, criterion, optimizer,
                            trainloader, epoch=epoch, **options)

            if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
                print("==> Test", options['loss'])
                results = test(current_time, net, criterion, testloader,
                            outloader, epoch=epoch, **options)
                print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(
                    results['ACC'], results['AUROC'], results['OSCR']))

                save_networks(net, model_path, file_name, criterion=criterion)
                if options['cs']: 
                    save_GAN(netG, netD, model_path, file_name)
                    fake = netG(fixed_noise)
                    GAN_path = os.path.join(model_path, current_time, 'samples')
                    mkdir_if_missing(GAN_path)
                    vutils.save_image(fake.data, '%s/gan_samples_epoch_%03d.png'%(GAN_path, epoch), normalize=True)

            if options['stepsize'] > 0:
                scheduler.step()
            
                data_batch, label_batch = next(iter(trainloader))
                writer.add_scalar('Train/loss', float(loss_all), epoch)
                writer.add_scalar('Test/ACC', float(results['ACC']), epoch)
                writer.add_scalar('Test/AUROC', float(results['AUROC']), epoch)
                writer.add_scalar('Test/OSCR', float(results['OSCR']), epoch)
                grid = torchvision.utils.make_grid(data_batch)
                writer.add_image('input_image', grid, epoch)
                writer.add_graph(net, data_batch)
        
                sys.stdout = Logger(osp.join(options['outf'],current_time, 'logs.txt'))
        

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

        return results


if __name__ == '__main__':
    
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 32
    results = dict()

    from split import splits_2020 as splits

    for i in range(len(splits[options['dataset']])):
        known = splits[options['dataset']][len(splits[options['dataset']])-i-1]
        if options['dataset'] == 'cifar100':
            unknown = splits[options['dataset']+'-' +
                             str(options['out_num'])][len(splits[options['dataset']])-i-1]
        elif options['dataset'] == 'tiny_imagenet':
            img_size = 64
            options['lr'] = 0.001
            unknown = list(set(list(range(0, 200))) - set(known))
        elif options['dataset'] == 'ixi_slice':
            img_size = 64
            options['lr'] = 0.001
            unknown = list(set(list(range(0, 17))) - set(known))
        else:
            unknown = list(set(list(range(0, 10))) - set(known))

        options.update(
            {
                'item':     i,
                'known':    known,
                'unknown':  unknown,
                'img_size': img_size
            }
        )

        dir_name = '{}_{}'.format(options['model'], options['loss'])
        current_time = time.strftime("%Y-%m-%d_%H_%M_%s",time.localtime())
        dir_path = os.path.join(options['outf'], 'results', dir_name, current_time)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if options['dataset'] == 'cifar100':
            file_name = '{}_{}.csv'.format(
                options['dataset'], options['out_num'])
        else:
            file_name = options['dataset'] + '.csv'

        res = main_worker(options, current_time)
        res['unknown'] = unknown
        res['known'] = known
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))
        
