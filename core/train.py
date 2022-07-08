from utils import AverageMeter
import os
import os.path as osp
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('Agg')


def train(current_time, net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()

    loss_all = 0
    all_features, all_labels = [], []
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y = net(data, True)
            logits, loss = criterion(x, y, labels)

            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))

        if options['use_gpu']:
            all_features.append(x.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
        else:
            all_features.append(x.data.numpy())
            all_labels.append(labels.data.numpy())

        #if (batch_idx+1) % options['print_freq'] == 0:
        print("Batch {}/{}\t Loss {:.6f} ({:.6f})"
                .format(batch_idx+1, len(trainloader), losses.val, losses.avg))

        loss_all += losses.avg

    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)
    plot_features(current_time, all_features, all_labels,
                  options['num_classes'], options['legendname'], epoch, prefix='train')

    return loss_all


def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG,
             trainloader, epoch=None, **options):
    print('train with confusing samples')
    losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    netD.train()
    netG.train()

    torch.cuda.empty_cache()

    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        gan_target = torch.FloatTensor(labels.size()).fill_(0)
        if options['use_gpu']:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            gan_target = gan_target.cuda()

        data, labels = Variable(data), Variable(labels)

        noise = torch.FloatTensor(
            data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1)
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterionD(output, targetv)
        errD_real.backward()

        # train with fake
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterionD(output, targetv)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterionD(output, targetv)

        # minimize the true distribution
        if options['use_gpu']:
            x, y = net(fake, True, 1 *
                    torch.ones(data.shape[0], dtype=torch.long).cuda())
        else:
            x, y = net(fake, True, 1 *
                    torch.ones(data.shape[0], dtype=torch.long))
        errG_F = criterion.fake_loss(x).mean()
        generator_loss = errG + options['beta'] * errG_F
        generator_loss.backward()
        optimizerG.step()

        lossesG.update(generator_loss.item(), labels.size(0))
        lossesD.update(errD.item(), labels.size(0))

        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        if options['use_gpu']:
            x, y = net(data, True, 0 *
                    torch.ones(data.shape[0], dtype=torch.long).cuda())
        else:
            x, y = net(data, True, 0 *
                    torch.ones(data.shape[0], dtype=torch.long))
        _, loss = criterion(x, y, labels)

        # KL divergence
        noise = torch.FloatTensor(
            data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1)
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        if options['use_gpu']:
            x, y = net(fake, True, 1 *
                    torch.ones(data.shape[0], dtype=torch.long).cuda())
        else:
            x, y = net(fake, True, 1 *
                    torch.ones(data.shape[0], dtype=torch.long))

        F_loss_fake = criterion.fake_loss(x).mean()
        total_loss = loss + options['beta'] * F_loss_fake
        total_loss.backward()
        optimizer.step()

        losses.update(total_loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})"
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg, lossesG.val, lossesG.avg, lossesD.val, lossesD.avg))

        loss_all += losses.avg

    return loss_all

from torchvision.datasets import ImageFolder

def plot_features(current_time, features, labels, num_classes, legendname, epoch, prefix):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',
              'C6', 'C7', 'C8', 'C9', 'C10', 'C11']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels == label_idx, 0],
            features[labels == label_idx, 1],
            c=colors[label_idx],
            s=1,
        )

    plt.legend(legendname, loc='upper right')
    dirname = osp.join('log', current_time, prefix)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
