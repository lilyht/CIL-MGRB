import os
import sys
import math
import util
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

from model import *
from dataset import *
import cal_MG
import cal_RB
import cal_HXE


def get_network(args):
    netname = args.net
    if netname == 'resnet':
        if args.ds[0:5] == 'cifar':  # cifar100 or cifar10
            return resnet34(len(classes)), nn.CrossEntropyLoss()
        else:
            return resnet18(len(classes)), nn.CrossEntropyLoss()
    elif netname[0:3] == 'Net':
        model_file = __import__('Model_FaulDiag')
        model_cls_name = netname
        Net = getattr(model_file, model_cls_name)
        return Net(66), nn.CrossEntropyLoss()
    else:
        print("No matching network!")


def convertlabelindex(args, labels):
    if args.ds == 'FARON':
        labels = torch.squeeze(labels).long().cuda()
    if args.ds[0:5] == 'cifar' or args.ds == 'FARON':
        for lb_idx in range(len(labels)):
            labels[lb_idx] = label_map[labels[lb_idx].item()]
    return labels


def feature_save(args, clsnet, inc_idx, dataloader):
    net_t.eval()
    clsnet.eval()

    with torch.no_grad():
        flag = True
        for data in dataloader:
            images, labels = data[0].cuda(), data[1].cuda()

            if args.ds == 'FARON':
                images = images.unsqueeze(1)
                images = images.type(torch.FloatTensor).cuda()
            labels = convertlabelindex(args, labels)

            features = net_t(images)  # the vector before the classifier

            if flag:
                features_all = features
                labels_all = labels
                flag = False
            else:
                features_all = torch.cat((features_all, features), 0)
                labels_all = torch.cat((labels_all, labels), 0)

        f_array = features_all.cpu().numpy()
        l_array = labels_all.cpu().numpy()
        np.save("./save/" + args.savename + "f_array_" + str(inc_idx) + ".npy", f_array)
        np.save("./save/" + args.savename + "l_array_" + str(inc_idx) + ".npy", l_array)


def test_step(inc_idx, clsnet, dataloader, flagtag=False, compute_cofmatrix=False):
    old_cls_num = test_num_lst[inc_idx - 1]
    new_cls_num = test_num_lst[inc_idx] - test_num_lst[inc_idx - 1]

    printtestloss = False
    testlosslst = []
    p_lst = []
    l_lst = []

    correct = 0
    correct5 = 0
    total = 0
    running_loss = 0.0
    total_hier_correct = None

    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

    net_t.eval()
    clsnet.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].cuda(), data[1].cuda()

            if args.ds == 'FARON':
                images = images.unsqueeze(1)
                images = images.type(torch.FloatTensor).cuda()

            labels = convertlabelindex(args, labels)

            outputs = net_t(images)
            outputs = clsnet(outputs)

            # top-1 acc
            _, predicted = torch.max(outputs.data, 1)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            total += labels.size(0)

            if compute_cofmatrix:  # True
                predicted_lst = predicted.tolist()
                labels_lst = labels.tolist()
                p_lst = p_lst + predicted_lst
                l_lst = l_lst + labels_lst

            correct += (predicted == labels).sum().item()

            if args.ds == 'cifar10':
                res = predicted == labels
                for lb_idx in range(len(labels)):
                    label_single = labels[lb_idx]
                    class_correct[label_single] += res[lb_idx].item()
                    class_total[label_single] += 1

            elif args.ds == 'cifar100' or args.ds == 'FARON':
                if flagtag:
                    res = predicted == labels
                    for lb_idx in range(len(labels)):
                        label_single = labels[lb_idx]
                        class_correct[label_single] += res[lb_idx].item()
                        class_total[label_single] += 1

            # calculate the top-5 accuracy except CIFAR10
            if args.ds == 'cifar10':
                topk = (1, 1)
            else:
                topk = (1, 5)
            maxk = max(topk)
            y_resize = labels.view(-1, 1)
            _, pred = outputs.topk(maxk, 1, True, True)
            correct5 += torch.eq(pred, y_resize).sum().float().item()
    print("test loss: ", running_loss)

    if compute_cofmatrix:
        with open('./draw/matrix' + args.savename + '.txt', mode='a') as filename:
            filename.write(str(l_lst))
            filename.write('\n')
            filename.write(str(p_lst))
        matrix = confusion_matrix(l_lst, p_lst)
        # Normalize by row
        matrix = matrix.astype(np.float)
        linesum = matrix.sum(1)
        linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
        matrix /= linesum  # matrix: <class 'numpy.ndarray'>
        # plot
        plt.matshow(matrix, cmap=plt.cm.Blues)
        plt.colorbar()
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        plt.savefig('./draw/matrix_' + args.savename + '.jpg', dpi=300)

    print("Accuracy on the test: top-1:{}, top-5: {}".format(100.0 * correct / total, 100.0 * correct5 / total))
    if args.ds == 'cifar10':
        acc_str = ''
        for acc_idx in range(test_num_lst[inc_idx]):
            try:
                acc = class_correct[acc_idx] / class_total[acc_idx]
            except:
                acc = 0
            finally:
                acc_str += 'classID: %d\tacc:%.2f \n' % (acc_idx, acc * 100)
        print(acc_str)
    elif args.ds == 'cifar100':
        if flagtag:
            acc_str = ''
            for acc_idx in range(test_num_lst[inc_idx]):
                try:
                    acc = class_correct[acc_idx] / class_total[acc_idx]
                except:
                    break
                finally:
                    acc_str += 'classID: %d\tacc:%.2f \n' % (acc_idx, acc * 100)
            print(acc_str)
            print("class_correct:", class_correct)
            old_cls_avgacc = sum(class_correct[: old_cls_num]) * 1.0 / old_cls_num
            new_cls_avgacc = sum(class_correct[old_cls_num: old_cls_num + new_cls_num]) * 1.0 / new_cls_num
            print("avg_old_accuracy: {}, avg_new_accuracy: {}".format(old_cls_avgacc, new_cls_avgacc))

            flagtag = False
    elif args.ds == 'FARON':
        if flagtag:
            acc_str = ''
            for acc_idx in range(test_num_lst[inc_idx]):
                try:
                    acc = class_correct[acc_idx] / class_total[acc_idx]
                except:
                    acc = 0
                finally:
                    acc_str += 'classID: %d\tacc:%.2f \n' % (acc_idx, acc * 100)
            print(acc_str)
            flagtag = False


def hier_loss(args, outputs_s, labels, name, inc_idx, datainfo, nodepathinfo, coarse_label, beta=10.0, max_depth=18):
    """
    Calculate the multi-granularity regularization term
    Args:
        args
        outputs_s (tensor): outputs of the student model. The size is (args.bs, n+m)
        labels (tensor): target. The size is torch.Size([args.bs])
        name (str): name of the criterion, CE or KL or MSE
        inc_idx (int): incremental phase
        beta (float): the parameter to control the distribution of the soft label
        max_depth(int): the depth of the current structure
    Return:
        the multi-granularity regularization term
    """

    hierloss = cal_MG.MGloss(args, outputs_s, labels, name, inc_idx, datainfo,
                             nodepathinfo, test_num_lst, coarse_label, beta, max_depth)
    return hierloss


def cal_distill_old(args, clsnet, previous_model, previous_fc, inc_idx, inputs, labels, T=2.0):
    """
    Calculate the distillation loss for the old samples in a batch
    Args:
        clsnet: classifier
        previous_model, previous_fc: the model of the last phase
        inc_idx (int): phase of incremental learning
        inputs (torch.Tensor): torch.Size([128, 1, 20, 121])
        labels (torch.Tensor): torch.Size([128])
    Returns:
        distill_old_loss
    """

    old_labels = []
    old_inputs = []
    distill_old_loss = 0.0

    for i in range(labels.size(0)):
        if labels[i] < test_num_lst[inc_idx - 1]:
            old_labels.append(labels[i].tolist())
            old_inputs.append(inputs[i].tolist())

    old_labels = torch.tensor(old_labels)
    old_inputs = torch.tensor(old_inputs)

    if old_labels.size(0) > 1:
        old_outputs_s = net_t(old_inputs)
        old_outputs_s = clsnet(old_outputs_s)

        with torch.no_grad():
            pre_p = previous_model(old_inputs)
            pre_p = previous_fc(pre_p)
            pre_p = F.softmax(pre_p / 2.0, dim=1)

        logp = F.log_softmax(old_outputs_s[:, :test_num_lst[inc_idx] - args.m] / T, dim=1)
        distill_old_loss = -torch.mean(torch.sum(pre_p * logp, dim=1)) * T * T
    else:
        distill_old_loss = torch.tensor(distill_old_loss).cuda().requires_grad_()

    return distill_old_loss


def train_main():
    loss_log_frequency = 500
    T = 2.0  # the temperature of distillation loss
    print("label_map: {}".format(label_map))
    print("order_lst： {}".format(datainfo.item()["order"]))
    print("train_num_lst: {}".format(train_num_lst))
    print("test_num_lst: {}".format(test_num_lst))
    previous_model = None
    previous_fc = None

    totalclasses, lamdalst, milestones, epoch_1, epoch_2, group = util.get_trainparameter(args)
    if args.no_dcp:  # No decoupling training phase
        epoch_2 = 0

    start_epoch_1 = -1
    start_epoch_2 = -1
    start_inc_idx = 0
    ############################################
    # Load from .pth file
    if args.loadfrompth != "":
        path_checkpoint = './checkpoint/ckpt_model_' + args.loadfrompth + '.pth'  # path of breakpoints
        checkpoint = torch.load(path_checkpoint)  # load checkpoint
        # Load model parameters
        net_t.load_state_dict(checkpoint['net_t'])
        stage = checkpoint['stage']
        start_inc_idx = checkpoint['inc_idx']
        lr = checkpoint['lr']
        if stage == "stage1":
            start_epoch_1 = checkpoint['epoch']  # set start epoch
            start_epoch_2 = -1
        elif stage == "stage2":
            start_epoch_1 = epoch_1 - 1
            start_epoch_2 = checkpoint['epoch']  # set start epoch
        else:
            print("stage information error!")

        print("###############\nInterrupted at: inc_idx: {}, epoch_1: {}, epoch_2: {}\n###############".format(
            start_inc_idx, start_epoch_1, start_epoch_2))
    ############################################

    for inc_idx in range(start_inc_idx, group):
        print("======Incremental learning phase {} stage1 begins======".format(inc_idx))
        clsnet = Classifier(test_num_lst[inc_idx])
        clsnet = nn.DataParallel(clsnet).cuda()
        optimizer, clsoptimizer, train_scheduler, cls_train_scheduler = util.get_trainoptim(args, milestones, net_t,
                                                                                            clsnet)
        ############################################
        # Load from .pth file
        if args.loadfrompth != "":
            path_checkpoint = './checkpoint/ckpt_model_' + args.loadfrompth + '.pth'
            checkpoint = torch.load(path_checkpoint)
            if start_epoch_1 != epoch_1 - 1:
                clsnet.load_state_dict(checkpoint['clsnet'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            clsoptimizer.load_state_dict(checkpoint['clsoptimizer'])
            stage = checkpoint['stage']
            start_inc_idx = checkpoint['inc_idx']
            lr = checkpoint['lr']
            if start_inc_idx > 0:
                previous_model = deepcopy(net_t)
                if start_epoch_1 == epoch_1 - 1 and start_epoch_2 == epoch_2 - 1:
                    previous_fc = Classifier(test_num_lst[start_inc_idx])
                elif start_epoch_1 != epoch_1 - 1 or start_epoch_2 != epoch_2 - 1:
                    previous_fc = Classifier(test_num_lst[start_inc_idx - 1])
                previous_fc = nn.DataParallel(previous_fc).cuda()
                previous_model.load_state_dict(checkpoint['previous_model'])
                previous_fc.load_state_dict(checkpoint['previous_fc'])
            train_scheduler.load_state_dict(checkpoint['train_scheduler'])
            cls_train_scheduler.load_state_dict(checkpoint['cls_train_scheduler'])
        ############################################

        # load data
        trainloader, validloader, testloader = util.get_dataloader(args, inc_idx)
        iter_per_epoch = len(trainloader)
        print("iter_per_epoch: {}".format(iter_per_epoch))
        warmup_scheduler = util.WarmUpLR(optimizer, iter_per_epoch * 1)

        # define loss list
        loss1lst, loss2lst, loss3lst = [], [], []

        cal_coarse_flag = False

        # start stage1
        for epoch in range(start_epoch_1 + 1, epoch_1):

            if inc_idx == 0:
                # print("epoch {}".format(epoch), flush=True)
                running_loss = 0.0
                net_t.train()
                clsnet.train()

                for i, data in enumerate(trainloader, 0):
                    if epoch < 1:
                        warmup_scheduler.step()
                    inputs, labels = data[0].cuda(), data[1].cuda()
                    # standard: torch.Size([128, 3, 32, 32]), torch.Size([128])

                    if args.ds == 'FARON':
                        inputs = inputs.unsqueeze(1)
                        inputs = inputs.type(torch.FloatTensor).cuda()
                        # FARON: torch.Size([128, 1, 20, 121]) torch.Size([128])
                    labels = convertlabelindex(args, labels)

                    optimizer.zero_grad()
                    clsoptimizer.zero_grad()

                    outputs_t = net_t(inputs)
                    outputs_t = clsnet(outputs_t)

                    _, predicted = torch.max(outputs_t.data, 1)

                    loss = criterion(outputs_t, labels)
                    loss.backward()
                    optimizer.step()
                    clsoptimizer.step()

                    running_loss += loss.item()
                    if i % loss_log_frequency == 0:
                        print('[%d, %5d] loss: %.6f  learnrate:%.5f' % (
                            epoch + 1, i + 1, running_loss / loss_log_frequency, optimizer.param_groups[0]['lr']),
                              flush=True)
                        running_loss = 0.0

                loss1lst.append(running_loss)

                if epoch >= 1:
                    train_scheduler.step(epoch)
                    cls_train_scheduler.step(epoch)

                if epoch % 5 == 0:
                    test_step(inc_idx, clsnet, dataloader=trainloader, flagtag=False)
                else:
                    test_step(inc_idx, clsnet, dataloader=testloader, flagtag=False)

                if epoch == epoch_1 - 1:
                    feature_save(args, clsnet, inc_idx, dataloader=trainloader)

                ##################
                # save model
                checkpoint = {
                    "net_t": net_t.state_dict(),
                    "clsnet": clsnet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'clsoptimizer': clsoptimizer.state_dict(),
                    "epoch": epoch,
                    "inc_idx": inc_idx,
                    "stage": "stage1",
                    "lr": optimizer.param_groups[0]['lr'],
                    'train_scheduler': train_scheduler.state_dict(),
                    'cls_train_scheduler': cls_train_scheduler.state_dict()
                }
                ##################
                # save .pth file
                if not os.path.isdir("./checkpoint"):
                    os.mkdir("./checkpoint")
                torch.save(checkpoint, './checkpoint/ckpt_model_' + args.savename + '.pth')
                args.loadfrompth = ""

            else:
                if cal_coarse_flag == False:
                    if "hierloss" in args.setloss and args.vis_hier:
                        # clustering using visual features
                        # coarse_label = util.vis_cluster(args, net_t, trainloader, label_map, test_num_lst[inc_idx], 5)
                        if args.ds == "cifar100":
                            n_clusters = args.cluster  # n_clusters = 10
                        elif args.ds == "cifar10":
                            n_clusters = 2
                        else:
                            n_clusters = args.cluster
                        cur_clsnum = test_num_lst[inc_idx]
                        clsid_vec = {}
                        feature_dict = {}
                        net_t.eval()
                        cnt = 0

                        for i in range(cur_clsnum):
                            feature_dict[i] = torch.zeros(1, 512).cuda()

                        for i, data in enumerate(validloader, 0):
                            # validloader  trainloader
                            inputs, labels = data[0].cuda(), data[1].cuda()

                            if args.ds == 'FARON':
                                inputs = inputs.unsqueeze(1)
                                inputs = inputs.type(torch.FloatTensor).cuda()
                            labels = convertlabelindex(args, labels)

                            outputs_t = net_t(inputs)  # get the feature
                            for lb_idx in range(len(labels)):
                                feature_dict[labels[lb_idx].item()] = torch.cat((feature_dict[labels[lb_idx].item()],
                                                                                 outputs_t[lb_idx].unsqueeze(0)), 0)
                            cnt = cnt + 1
                            # print(cnt)

                        for i in range(cur_clsnum):
                            id_allvectors = feature_dict[i]
                            id_mean = torch.mean(id_allvectors, dim=0)
                            id_mean = id_mean.view(-1)
                            clsid_vec[i] = id_mean.tolist()

                        mean_vector = []
                        for i in range(cur_clsnum):
                            # join in order
                            mean_vector.append(clsid_vec[i])

                        X = np.array(mean_vector)
                        print(X.shape)  # (66, 2420)
                        if len(X) < n_clusters:
                            n_clusters = 5
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
                        coarse_label = kmeans.labels_.tolist()
                        max_depth = 2
                    elif "hierloss" in args.setloss and args.ds == "cifar10":
                        coarse_label = util.semantic_hier(args, order_lst, test_num_lst[inc_idx], 2)
                        max_depth = 2
                    elif "hierloss" in args.setloss and args.ds == "FARON":
                        coarse_label = util.FARONFeature_hier(args, order_lst, test_num_lst[inc_idx], 15)
                        max_depth = 2
                    elif "hierloss" in args.setloss:
                        coarse_label = util.semantic_hier(args, order_lst, test_num_lst[inc_idx], 3)
                        max_depth = 2
                    elif "hierloss" in args.setloss and (args.ds == "miniimagenet" or args.ds == "imagenet100"):
                        coarse_label = [0]
                        max_depth = util.cal_height(args, test_num_lst[inc_idx])
                    else:
                        coarse_label = [0]
                        max_depth = 2
                    print("coarse_label:", coarse_label)
                    cal_coarse_flag = True

                w_lst = cal_RB.ClassBalancedWeight(args, inc_idx, totalclasses, test_num_lst)

                celoss, dis_loss, hierloss, cb_loss, focal_loss, dis_old_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                running_loss = 0.0
                net_t.train()  # train
                clsnet.train()  # train

                for i, data in enumerate(trainloader, 0):
                    if epoch < 1:
                        warmup_scheduler.step()
                    inputs, labels = data[0].cuda(), data[1].cuda()

                    if args.ds == 'FARON':
                        inputs = inputs.unsqueeze(1)
                        inputs = inputs.type(torch.FloatTensor).cuda()
                    labels = convertlabelindex(args, labels)

                    optimizer.zero_grad()
                    clsoptimizer.zero_grad()

                    outputs_s = net_t(inputs)
                    outputs_s = clsnet(outputs_s)  # torch.Size([128, n+m])
                    _, predicted = torch.max(outputs_s.data, 1)

                    alpha = lamdalst[inc_idx]

                    with torch.no_grad():
                        pre_p = previous_model(inputs)
                        pre_p = previous_fc(pre_p)
                        pre_p = F.softmax(pre_p / T, dim=1)  # pre_p size: torch.Size([128, n])
                    logp = F.log_softmax(outputs_s[:, :test_num_lst[inc_idx] - args.m] / T, dim=1)
                    loss1 = -torch.mean(torch.sum(pre_p * logp, dim=1)) * T * T * alpha * args.dscale
                    dis_loss += loss1.item()

                    if args.cbloss:  # class balanced loss
                        weights = []
                        for lb_idx in range(len(labels)):
                            weights.append(w_lst[labels[lb_idx]])
                        weights = torch.Tensor(weights)
                        weights = torch.unsqueeze(weights, 1).cuda()
                        labels_ = F.one_hot(labels, num_classes=test_num_lst[inc_idx])
                        # weights
                        log_outputs_softmax = F.log_softmax(outputs_s, dim=1)
                        loss2 = -torch.mean(torch.sum(labels_ * log_outputs_softmax * weights))
                        celoss += loss2.item()
                    elif args.HXEloss:
                        loss2 = cal_HXE.HXELoss(args, outputs_s, labels, test_num_lst, inc_idx,
                                                20, datainfo, coarse_label, 0.1)
                        celoss += loss2.item()
                    else:  # cross entropy loss
                        loss2 = criterion(outputs_s, labels)
                        celoss += loss2.item()

                    # multi-granularity regularization
                    if "hierloss" in args.setloss:
                        loss3 = hier_loss(args, outputs_s, labels, "KL", inc_idx, datainfo, nodepathinfo,
                                          coarse_label, beta=args.beta, max_depth=max_depth) * args.scale
                        hierloss += loss3.item()

                    # ce/ce+distill/ce+hierloss/ce+distill+hierloss/distill+hierloss
                    if args.setloss == 'ce':
                        total_loss = loss2
                    elif args.setloss == 'ce+distill':
                        total_loss = loss1 + (1 - alpha) * loss2
                    elif args.setloss == 'ce+hierloss':
                        total_loss = loss2 + loss3
                    elif args.setloss == 'distill+hierloss':
                        total_loss = loss1 + loss3
                    elif args.setloss == 'ce+distill+hierloss':
                        total_loss = loss1 + (1 - alpha) * loss2 + loss3
                    else:
                        total_loss = loss1 + (1 - alpha) * loss2

                    total_loss.backward()
                    optimizer.step()
                    clsoptimizer.step()

                    running_loss += total_loss.item()

                    if i % loss_log_frequency == 0:
                        print('[%d, %5d] loss: %.6f  learnrate:%.5f' % (
                            epoch + 1, i + 1, running_loss / loss_log_frequency, optimizer.param_groups[0]['lr']),
                              flush=True)
                        running_loss = 0.0

                if epoch >= 1:
                    train_scheduler.step(epoch)
                    cls_train_scheduler.step(epoch)

                if epoch % 5 == 0:
                    test_step(inc_idx, clsnet, trainloader, flagtag=False)
                else:
                    test_step(inc_idx, clsnet, testloader)
                if epoch == epoch_1 - 1:
                    feature_save(args, clsnet, inc_idx, dataloader=trainloader)

                ##################
                # save model
                checkpoint = {
                    "net_t": net_t.state_dict(),
                    "clsnet": clsnet.state_dict(),
                    "previous_model": previous_model.state_dict(),
                    "previous_fc": previous_fc.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'clsoptimizer': clsoptimizer.state_dict(),
                    "epoch": epoch,
                    "stage": "stage1",
                    "inc_idx": inc_idx,
                    "lr": optimizer.param_groups[0]['lr'],
                    'train_scheduler': train_scheduler.state_dict(),
                    'cls_train_scheduler': cls_train_scheduler.state_dict()
                }
                # save .pth file
                if not os.path.isdir("./checkpoint"):
                    os.mkdir("./checkpoint")
                torch.save(checkpoint, './checkpoint/ckpt_model_' + args.savename + '.pth')
                ##################

                loss1lst.append(celoss)
                loss2lst.append(dis_loss)
                if "hierloss" in args.setloss:
                    loss3lst.append(hierloss)

        print("----Evaluate on test set...", flush=True)
        test_step(inc_idx, clsnet, dataloader=testloader, flagtag=True)

        x1 = range(start_epoch_1 + 1, epoch_1)
        y1 = loss1lst
        y2 = loss2lst
        if "hierloss" in args.setloss:
            y3 = loss3lst
        print("x1", x1)
        print("y1", y1)

        plt.figure(num=2, figsize=(20, 12))

        if inc_idx == 0:
            ax4 = plt.subplot(224)
            ax4.plot(x1, y1, color="b", linestyle="--", label='old_model—ce')
            plt.savefig('./save/img/' + args.savename + '.jpg')
        else:
            ax1 = plt.subplot(221)
            ax1.plot(x1, y1, color="b", linestyle="--", label='stage1—ce')
            if args.setloss == 'ce+distill':
                ax1.plot(x1, y2, color="r", linestyle="-.", label='stage1—distill')
            elif args.setloss == 'ce+hierloss':
                ax1.plot(x1, y3, color="y", linestyle="--", label='stage1—hierloss')
                ax3 = plt.subplot(223)
                ax3.plot(x1, y3, color="y", linestyle="--", label='stage1—hierloss')
            elif args.setloss == 'ce+distill+hierloss' or args.setloss == 'distill+hierloss':
                print("y3", y3)
                ax1.plot(x1, y2, color="r", linestyle="-.", label='stage1—distill')
                ax1.plot(x1, y3, color="y", linestyle="--", label='stage1—hierloss')
                ax3 = plt.subplot(223)
                ax3.plot(x1, y3, color="y", linestyle="--", label='stage1—hierloss')

        plt.savefig('./save/img/' + args.savename + '.jpg')
        start_epoch_1 = -1

        '''
        stage2: decoupling train stage: retrain the fully connected layer with sampled data (balanced)
        '''

        if inc_idx > 0:
            print("======Incremental learning phase {} stage2 begins======".format(inc_idx))
            # clsoptimizer, clstrain_scheduler = util.get_clsnetoptim(args, milestones, clsnet)
            clsoptimizer, clstrain_scheduler = util.get_clsnetoptim(args, [15], clsnet)
            ######################
            # Load from .pth file
            if args.loadfrompth != "":
                path_checkpoint = './checkpoint/ckpt_model_' + args.loadfrompth + '.pth'
                checkpoint = torch.load(path_checkpoint)
                net_t.load_state_dict(checkpoint['net_t'])
                clsnet.load_state_dict(checkpoint['clsnet'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                clsoptimizer.load_state_dict(checkpoint['clsoptimizer'])
                lr = checkpoint['lr']
                if start_inc_idx > 0:
                    previous_model.load_state_dict(checkpoint['previous_model'])
                    previous_fc.load_state_dict(checkpoint['previous_fc'])
                # train_scheduler.load_state_dict(checkpoint['train_scheduler'])
                # cls_train_scheduler.load_state_dict(checkpoint['cls_train_scheduler'])
                args.loadfrompth = ""
            ######################

            loss1lst, loss2lst, loss3lst = [], [], []

            for epoch in range(start_epoch_2 + 1, epoch_2):

                celoss, dis_loss, hierloss, cb_loss = 0.0, 0.0, 0.0, 0.0
                running_loss = 0.0
                net_t.eval()
                clsnet.train()  # decoupling training

                if epoch_2 == 0:
                    validloader = trainloader  # when epoch_2=0, it means the validloader is not required

                for i, data in enumerate(validloader, 0):
                    if epoch < 1:
                        warmup_scheduler.step()
                    inputs, labels = data[0].cuda(), data[1].cuda()  # label: type labels: <class 'torch.Tensor'>
                    if args.ds == 'FARON':
                        inputs = inputs.unsqueeze(1)
                        inputs = inputs.type(torch.FloatTensor).cuda()
                    labels = convertlabelindex(args, labels)

                    clsoptimizer.zero_grad()
                    outputs_s = net_t(inputs)
                    outputs_s = clsnet(outputs_s)

                    _, predicted = torch.max(outputs_s.data, 1)

                    alpha = lamdalst[inc_idx]

                    # knowledge distillation loss
                    with torch.no_grad():
                        pre_p = net_t(inputs)
                        pre_p = clsnet(pre_p)
                        pre_p = F.softmax(pre_p / T, dim=1)  # pre_p size: torch.Size([128, n])
                    logp = F.log_softmax(outputs_s[:, :test_num_lst[inc_idx]] / T, dim=1)  # torch.Size([128, n])
                    loss1 = -torch.mean(torch.sum(pre_p * logp, dim=1)) * T * T * alpha * args.dscale
                    dis_loss += loss1.item()

                    # cross entropy loss
                    loss2 = criterion(outputs_s, labels)
                    celoss += loss2.item()

                    if "hierloss" in args.setloss:
                        loss3 = hier_loss(args, outputs_s, labels, "KL", inc_idx, datainfo,
                                          nodepathinfo, coarse_label, beta=args.beta) * args.scale
                        hierloss += loss3.item()

                    if args.setloss == 'ce':
                        total_loss = loss2
                    elif args.setloss == 'ce+distill':
                        total_loss = loss1 + (1 - alpha) * loss2
                    elif args.setloss == 'ce+distill+hierloss':
                        total_loss = loss1 + (1 - alpha) * loss2 + loss3
                    elif args.setloss == 'distill+hierloss':
                        total_loss = loss1 + loss3
                    elif args.setloss == 'ce+hierloss':
                        total_loss = loss2 + loss3
                    else:
                        total_loss = loss2
                    total_loss.backward()
                    clsoptimizer.step()

                    running_loss += total_loss.item()

                    if i % loss_log_frequency == 0:
                        print('[%d, %5d] loss: %.3f  learnrate:%.5f' % (epoch + 1, i + 1,
                                                                        running_loss / loss_log_frequency,
                                                                        clsoptimizer.param_groups[0]['lr']), flush=True)
                        running_loss = 0.0

                if epoch >= 1:
                    clstrain_scheduler.step(epoch)
                if epoch % 5 == 0:
                    test_step(inc_idx, clsnet, validloader, flagtag=False)
                elif inc_idx == group - 1 and epoch == epoch_2 - 1:
                    test_step(inc_idx, clsnet, testloader, flagtag=False, compute_cofmatrix=True)
                else:
                    test_step(inc_idx, clsnet, testloader)

                loss1lst.append(celoss)
                loss2lst.append(dis_loss)
                if "hierloss" in args.setloss:
                    loss3lst.append(hierloss)

            print("----Evaluate on test set...", flush=True)
            test_step(inc_idx, clsnet, dataloader=testloader, flagtag=True)
            print("======Incremental learning phase {} stage2 ends======".format(inc_idx))

            x1 = range(start_epoch_2 + 1, epoch_2)
            y1 = loss1lst
            y2 = loss2lst
            if "hierloss" in args.setloss:
                y3 = loss3lst
            if epoch_2 != 0:
                plt.figure(num=2, figsize=(20, 12))
                ax2 = plt.subplot(222)
                ax2.plot(x1, y1, color="b", linestyle="--", label='stage2—ce')
                if args.setloss == 'ce+distill':
                    ax2.plot(x1, y2, color="r", linestyle="-.", label='stage2—distill')
                elif args.setloss == 'ce+hierloss':
                    ax2.plot(x1, y3, color="y", linestyle="--", label='stage2—hierloss')
                elif args.setloss == 'ce+distill+hierloss' or args.setloss == 'distill+hierloss':
                    ax2.plot(x1, y2, color="r", linestyle="-.", label='stage2—distill')
                    ax2.plot(x1, y3, color="y", linestyle="--", label='stage2—hierloss')

                plt.savefig('./save/img/' + args.savename + '.jpg')

        previous_model = deepcopy(net_t)
        previous_fc = deepcopy(clsnet)

        ##################
        # save model
        print("save model, inc_idx: {}".format(inc_idx))
        checkpoint = {
            "net_t": net_t.state_dict(),
            "clsnet": clsnet.state_dict(),
            "previous_model": previous_model.state_dict(),
            "previous_fc": previous_fc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'clsoptimizer': clsoptimizer.state_dict(),
            "epoch": epoch_2 - 1,
            "stage": "stage2",
            "inc_idx": inc_idx,
            "lr": clsoptimizer.param_groups[0]['lr'],
            'train_scheduler': train_scheduler.state_dict(),
            'cls_train_scheduler': cls_train_scheduler.state_dict()
        }
        # save .pth file
        if not os.path.isdir("./checkpoint"):
            os.mkdir("./checkpoint")
        torch.save(checkpoint, './checkpoint/ckpt_model_' + args.savename + '.pth')
        start_epoch_2 = -1
        ##################

        # output number of parameter
        if inc_idx == group - 1:
            test_step(inc_idx, clsnet, testloader, flagtag=False, compute_cofmatrix=True)
            print('# our model net_t parameters:', sum(param.numel() for param in net_t.parameters()))
            print('# our model clsnet parameters:', sum(param.numel() for param in clsnet.parameters()))


if __name__ == '__main__':
    print("run etrain.py")
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, required=False, default="cpu",
                        help='GPU number')
    parser.add_argument('-complexity', type=int, required=False, default=2,
                        help='Compexity for googlenet')
    parser.add_argument('-net', type=str, required=False, default="resnet",
                        help='Networks:resnet,googlenet,hiernet')
    parser.add_argument('-ds', type=str, required=False, default="cifar100",
                        help='Dataset:cifar100,cifar10,miniimagenet,imagenet100,FARON')
    parser.add_argument('-n', type=int, required=True, default=20,
                        help='number of the init classes, please input like this: -n 20')
    parser.add_argument('-m', type=int, required=True, default=20,
                        help='number of the classes increamentally transfered to model, please input like this: -m 20')
    parser.add_argument('-beta', type=float, required=False, default=10.0,
                        help='param for hier loss')
    parser.add_argument('-lr', type=float, required=False, default=0.1,
                        help='learning rate')
    parser.add_argument('-bs', type=int, required=False, default=128,
                        help='batch size')
    parser.add_argument('-lrdecay', type=float, required=False, default=0.2,
                        help='learning rate decay')
    parser.add_argument('-savename', type=str, required=True, default="model_cf100",
                        help='model parameters are saved in this file')
    parser.add_argument('-cluster', type=int, required=False, default=-1,
                        help='If you use this parameter, the network will use word embedding to generate the '
                             'hierarchical information')
    parser.add_argument('-scale', type=float, required=False, default=1.0,
                        help='Control the scale of hierloss')
    parser.add_argument('-dscale', type=float, required=False, default=1.0,
                        help='Control the scale of distillation loss')
    parser.add_argument('-setloss', type=str, required=False, default="ce+hierloss",
                        help='set loss: ce/ce+distill/ce+hierloss/ce+distill+hierloss/distill+hierloss')
    parser.add_argument('--cbloss', action='store_true',
                        help='use class balanced loss instead of cross entropy loss')
    parser.add_argument('--no_dcp', action='store_true',
                        help='no decoupling stage(used when abalation study)')
    parser.add_argument('--vis_hier', action='store_true',
                        help='use visual feature to build hierarchy instead of semantic information')
    parser.add_argument('--HXEloss', action='store_true',
                        help='use HXE loss instead of cross entropy loss')
    parser.add_argument('--ckp_prefix', type=str, default=os.path.basename(sys.argv[0])[:-3],
                        help='Checkpoint prefix')
    parser.add_argument('-loadfrompth', type=str, required=False, default="",
                        help='load saved from .pth, previous model parameters are saved in this file')
    parser.add_argument('--disable_gpu_occupancy', action='store_false', help='disable GPU occupancy')

    args = parser.parse_args()
    print("--------------------------------------------------")
    for k in args.__dict__:
        print('{0:25}{1:15}'.format(k, str(args.__dict__[k])))
    print("--------------------------------------------------")

    util.setup_gpu(args, os)
    args.ckp_prefix = '{}_{}_{}_{}_'.format(args.ckp_prefix, args.ds, args.n, args.m)

    classes, datainfopath, nodepathinfo = util.get_structure(args)
    datainfo = np.load(datainfopath, allow_pickle=True)
    label_map = datainfo.item()["label_map"]
    order_lst = datainfo.item()["order"]
    test_num_lst = datainfo.item()["test_num"]
    train_num_lst = datainfo.item()["train_num"]

    net_t, criterion = get_network(args)
    if args.gpu != "cpu":
        net_t = nn.DataParallel(net_t).cuda()
    train_main()
