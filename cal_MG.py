import sys
import math
import utils
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def MGloss(args, outputs_s, labels, name, inc_idx, datainfo, nodepathinfo, 
           test_num_lst, coarse_label, beta=10.0, max_depth=18):
    '''
    Calculate the multi-granularity regularization term
    Args:
        args
        outputs_s (tensor): outputs of the student model. The size is (args['b'], n+m)
        labels (tensor): target. The size is torch.Size([args['b']])
        name (str): name of the criterion, CE or KL or MSE
        inc_idx (int): incremental phase
        beta (float): the parameter to control the distribution of soft label
        max_depth(int): the depth of the current structure
    Returns:
        hier loss
    '''
    _, predicted = torch.max(outputs_s.data, 1)
        
    if args["vis_hier"] == True or args["dataset"] == "cifar10":
        # order_lst = datainfo.item()["order"]
        # coarse_label = util.semantic_hier(args, order_lst, test_num_lst[inc_idx], args['cluster'])
        
        # 生成标准的coarse标签
        coarse = []
        for lb_idx in range(len(labels)):
            coarse.append(coarse_label[labels[lb_idx].item()])
        coarsetensor = torch.tensor(coarse).cuda()
        hl_std_all = []
        l1 = beta * (-0.5)
        l2 = beta * (-1)
        l3 = 0

        for lb_idx in range(len(labels)):
            hl_std = []  # 标准分布
            std_coarse = coarse[lb_idx]
            std_fine = labels[lb_idx]
            for l in range(test_num_lst[inc_idx]):  # n+m个类
                if coarse_label[l] == std_coarse and l == std_fine:
                    hl_std.append(l3)
                elif coarse_label[l] == std_coarse:
                    hl_std.append(l1)
                else:
                    hl_std.append(l2)
            hl_std2 = [x for x in hl_std]
            hl_std_all.append(hl_std2)
        
        hl_std_tensor = torch.tensor(hl_std_all).cuda().requires_grad_()

        if name == "KL":
            hier_criterion = nn.KLDivLoss(reduction="batchmean")  # KL散度
            outputs_s = F.log_softmax(outputs_s, dim=1)
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            return hier_criterion(outputs_s, hl_std_tensor)
        elif name == "MSE":
            hier_criterion = nn.MSELoss()  # MSE
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            outputs_s = F.softmax(outputs_s, dim=1)
            return hier_criterion(outputs_s, hl_std_tensor)
        elif name == "CE":
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            outputs_s = F.softmax(outputs_s, dim=1)
            outputs_s = torch.clamp(outputs_s, min=1e-10, max=1.0)
            hloss = torch.sum(-hl_std_tensor * torch.log(outputs_s), dim=1)
            return torch.mean(hloss)
    
    elif args['cluster'] == -1:
        # 使用datainfo中定义的coarse_label，本身有粗粒度标签的数据集（如cifar100）适用
        if args['dataset'] == "cifar100":
            coarse_label = datainfo.item()["coarse_label"]
            # 生成一个batch中数据对应的coarse标签
            coarse = []
            for lb_idx in range(len(labels)):
                coarse.append(coarse_label[labels[lb_idx].item()])
            coarsetensor = torch.tensor(coarse).cuda()

            hl_std_all = []
            l1 = beta * (-0.5)
            l2 = beta * (-1)
            l3 = 0

            for lb_idx in range(len(labels)):
                hl_std = []  # 标准分布
                std_coarse = coarse[lb_idx]
                std_fine = labels[lb_idx]
                for l in range(test_num_lst[inc_idx]):  # n+m个类
                    if coarse_label[l] == std_coarse and l == std_fine:
                        hl_std.append(l3)
                    elif coarse_label[l] == std_coarse:
                        hl_std.append(l1)
                    else:
                        hl_std.append(l2)

                hl_std2 = [x for x in hl_std]
                hl_std_all.append(hl_std2)

            hl_std_tensor = torch.tensor(hl_std_all).cuda().requires_grad_()

            if name == "KL":
                hier_criterion = nn.KLDivLoss(reduction="batchmean")  # KL散度
                outputs_s = F.log_softmax(outputs_s, dim=1)
                hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
                return hier_criterion(outputs_s, hl_std_tensor)
            elif name == "MSE":
                hier_criterion = nn.MSELoss()  # MSE
                hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
                outputs_s = F.softmax(outputs_s, dim=1)
                return hier_criterion(outputs_s, hl_std_tensor)
            elif name == "CE":
                hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
                outputs_s = F.softmax(outputs_s, dim=1)
                outputs_s = torch.clamp(outputs_s, min=1e-10, max=1.0)
                hloss = torch.sum(-hl_std_tensor * torch.log(outputs_s), dim=1)
                return torch.mean(hloss)
        
        
    elif args['cluster'] == -2:
        # mini_imagenet和imagenet的树结构
        
        clsname_path = nodepathinfo.item()["clsname_path"]
        clsid_path = nodepathinfo.item()["clsid_path"]
                
        hl_std_all = []
        
        for lb_idx in range(len(labels)):
            hl_std = []  # 标准分布
            lb = labels[lb_idx].item()
            for j in range(test_num_lst[inc_idx]):  # n+m个类
                if j == lb:
                    hl_std.append(0)
                else:
                    # print("l: {}, j: {}".format(lb, j))
                    j_pathlst = clsid_path[j]  # 第j类的路径
                    trg_pathlst = clsid_path[lb]  # ground truth的路径
                    sub_depth = max(len(clsid_path[j]), len(clsid_path[lb]))  # 两个子节点最深的深度
                    min_depth = min(len(clsid_path[j]), len(clsid_path[lb]))
                    common_depth = 0
                    
                    for k in range(sub_depth):
                        if k < min_depth:
                            if j_pathlst[k] == trg_pathlst[k]:
                                common_depth = common_depth + 1
                            else:
                                break
                    sub_depth = sub_depth - common_depth
                    ll = -1.0 * beta * (1.0 * sub_depth / (max_depth - 1))
                    hl_std.append(ll)
            hl_std_all.append(hl_std)
            
        hl_std_tensor = torch.tensor(hl_std_all).cuda().requires_grad_()

        if name == "KL":
            hier_criterion = nn.KLDivLoss(reduction="batchmean")  # KL散度
            outputs_s = F.log_softmax(outputs_s, dim=1)
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            # print("hl_std_tensor[0]: ", hl_std_tensor[0])
            return hier_criterion(outputs_s, hl_std_tensor)
        elif name == "MSE":
            hier_criterion = nn.MSELoss()  # MSE
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            outputs_s = F.softmax(outputs_s, dim=1)
            return hier_criterion(outputs_s, hl_std_tensor)
        elif name == "CE":
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            outputs_s = F.softmax(outputs_s, dim=1)
            outputs_s = torch.clamp(outputs_s, min=1e-10, max=1.0)
            hloss = torch.sum(-hl_std_tensor * torch.log(outputs_s), dim=1)
            return torch.mean(hloss)
    
    else:
        # 语义聚类算法生成层次结构
        if args['dataset'] == 'cifar10':
            if args['cluster'] == 2:
                coarse_label = [0, 0, 0, 0, 1, 0, 0, 1, 1, 1]
                coarse_label = [0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
        
        # 生成标准的coarse标签
        coarse = []
        for lb_idx in range(len(labels)):
            coarse.append(coarse_label[labels[lb_idx].item()])
        coarsetensor = torch.tensor(coarse).cuda()
        hl_std_all = []
        l1 = beta * (-0.5)
        l2 = beta * (-1)
        l3 = 0

        for lb_idx in range(len(labels)):
            hl_std = []  # 标准分布
            std_coarse = coarse[lb_idx]
            std_fine = labels[lb_idx]
            for l in range(test_num_lst[inc_idx]):  # n+m个类
                if coarse_label[l] == std_coarse and l == std_fine:
                    hl_std.append(l3)
                elif coarse_label[l] == std_coarse:
                    hl_std.append(l1)
                else:
                    hl_std.append(l2)
            hl_std2 = [x for x in hl_std]
            hl_std_all.append(hl_std2)
        
        hl_std_tensor = torch.tensor(hl_std_all).cuda().requires_grad_()

        if name == "KL":
            hier_criterion = nn.KLDivLoss(reduction="batchmean")  # KL散度
            outputs_s = F.log_softmax(outputs_s, dim=1)
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            return hier_criterion(outputs_s, hl_std_tensor)
        elif name == "MSE":
            hier_criterion = nn.MSELoss()  # MSE
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            outputs_s = F.softmax(outputs_s, dim=1)
            return hier_criterion(outputs_s, hl_std_tensor)
        elif name == "CE":
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            outputs_s = F.softmax(outputs_s, dim=1)
            outputs_s = torch.clamp(outputs_s, min=1e-10, max=1.0)
            hloss = torch.sum(-hl_std_tensor * torch.log(outputs_s), dim=1)
            return torch.mean(hloss)