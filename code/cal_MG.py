import sys
import math
import util
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def MGloss(args, outputs_s, labels, name, inc_idx, datainfo, nodepathinfo, 
           test_num_lst, coarse_label, beta=10.0, max_depth=18):
    """
    Calculate the multi-granularity regularization term
    Args:
        args
        outputs_s (tensor): outputs of the student model. The size is (args.bs, n+m)
        labels (tensor): target. The size is torch.Size([args.bs])
        name (str): name of the criterion, CE or KL or MSE
        inc_idx (int): incremental phase
        beta (float): the parameter to control the distribution of soft label
        max_depth(int): the depth of the current structure
    Returns:
        multi-granularity loss
    """

    _, predicted = torch.max(outputs_s.data, 1)

    if args.vis_hier == True or args.ds == "cifar10":
        # generate standard coarse labels
        coarse = []
        for lb_idx in range(len(labels)):
            coarse.append(coarse_label[labels[lb_idx].item()])
        coarsetensor = torch.tensor(coarse).cuda()
        hl_std_all = []
        l1 = beta * (-0.5)
        l2 = beta * (-1)
        l3 = 0

        for lb_idx in range(len(labels)):
            hl_std = []  # standard label
            std_coarse = coarse[lb_idx]
            std_fine = labels[lb_idx]
            for l in range(test_num_lst[inc_idx]):  # n+m classes
                if coarse_label[l] == std_coarse and l == std_fine:
                    hl_std.append(l3)
                elif coarse_label[l] == std_coarse:
                    hl_std.append(l1)
                else:
                    hl_std.append(l2)
            hl_std2 = [x for x in hl_std]
            hl_std_all.append(hl_std2)
        
        hl_std_tensor = torch.tensor(hl_std_all).cuda().requires_grad_()

        hier_criterion = nn.KLDivLoss(reduction="batchmean")
        outputs_s = F.log_softmax(outputs_s, dim=1)
        hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
        return hier_criterion(outputs_s, hl_std_tensor)
        
    elif args.ds == "FARON":
        # get the structure by clustering the mean values of classes
        coarse = []
        for lb_idx in range(len(labels)):
            coarse.append(coarse_label[labels[lb_idx].item()])
        coarsetensor = torch.tensor(coarse).cuda()
        hl_std_all = []
        l1 = beta * (-0.5)
        l2 = beta * (-1)
        l3 = 0
        
        for lb_idx in range(len(labels)):
            hl_std = []  # standard label
            std_coarse = coarse[lb_idx]
            std_fine = labels[lb_idx]
            for l in range(test_num_lst[inc_idx]):
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
            hier_criterion = nn.KLDivLoss(reduction="batchmean")
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
    
    elif args.cluster == -1:
        # suitable for datasets with multi-granularity structure (such as cifar100)
        # use "coarse_label" defined in datainfo
        if args.ds == "cifar100":
            coarse_label = datainfo.item()["coarse_label"]
            # generate coarse labels corresponding to the data in a batch
            coarse = []
            for lb_idx in range(len(labels)):
                coarse.append(coarse_label[labels[lb_idx].item()])
            coarsetensor = torch.tensor(coarse).cuda()

            hl_std_all = []
            l1 = beta * (-0.5)
            l2 = beta * (-1)
            l3 = 0

            for lb_idx in range(len(labels)):
                hl_std = []  # standard label
                std_coarse = coarse[lb_idx]
                std_fine = labels[lb_idx]
                for l in range(test_num_lst[inc_idx]):
                    if coarse_label[l] == std_coarse and l == std_fine:
                        hl_std.append(l3)
                    elif coarse_label[l] == std_coarse:
                        hl_std.append(l1)
                    else:
                        hl_std.append(l2)

                hl_std2 = [x for x in hl_std]
                hl_std_all.append(hl_std2)

            hl_std_tensor = torch.tensor(hl_std_all).cuda().requires_grad_()

            hier_criterion = nn.KLDivLoss(reduction="batchmean")
            outputs_s = F.log_softmax(outputs_s, dim=1)
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            return hier_criterion(outputs_s, hl_std_tensor)

        
        elif args.ds == "FARON":
            # use the structure provided in tree_15.xlsx
            coarse_label = datainfo.item()["coarse_label"]
            # generate coarse labels corresponding to the data in a batch
            coarse = []
            for lb_idx in range(len(labels)):
                coarse.append(coarse_label[labels[lb_idx].item()])
            coarsetensor = torch.tensor(coarse).cuda()
            hl_std_all = []
            l1 = beta * (-0.5)
            l2 = beta * (-1)
            l3 = 0

            for lb_idx in range(len(labels)):
                hl_std = []
                std_coarse = coarse[lb_idx]
                std_fine = labels[lb_idx]
                for l in range(test_num_lst[inc_idx]):
                    if coarse_label[l] == std_coarse and l == std_fine:
                        hl_std.append(l3)
                    elif coarse_label[l] == std_coarse:
                        hl_std.append(l1)
                    else:
                        hl_std.append(l2)

                hl_std2 = [x for x in hl_std]
                hl_std_all.append(hl_std2)

            hl_std_tensor = torch.tensor(hl_std_all).cuda().requires_grad_()
            hier_criterion = nn.KLDivLoss(reduction="batchmean")
            outputs_s = F.log_softmax(outputs_s, dim=1)
            hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
            return hier_criterion(outputs_s, hl_std_tensor)

    elif args.cluster == -2:
        # suitable for miniimagenet and imagenet
        clsid_path = nodepathinfo.item()["clsid_path"]
                
        hl_std_all = []
        
        for lb_idx in range(len(labels)):
            hl_std = []
            lb = labels[lb_idx].item()
            for j in range(test_num_lst[inc_idx]):
                if j == lb:
                    hl_std.append(0)
                else:
                    j_pathlst = clsid_path[j]  # the path of class j
                    trg_pathlst = clsid_path[lb]  # the path of the ground truth
                    sub_depth = max(len(clsid_path[j]), len(clsid_path[lb]))  # deeper depth of two children nodes
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
        hier_criterion = nn.KLDivLoss(reduction="batchmean")
        outputs_s = F.log_softmax(outputs_s, dim=1)
        hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
        return hier_criterion(outputs_s, hl_std_tensor)
    
    else:
        # generate hierarchies using semantic clustering
        if args.ds == 'cifar100':
            if args.cluster == 20:
                coarse_label = [15, 8, 8, 15, 9, 4, 15, 18, 18, 12, 4, 10, 0, 7, 18, 12, 6, 18, 0, 8,
                                10, 11, 2, 8, 16, 10, 12, 5, 4, 12, 15, 14, 6, 7, 18, 1, 7, 12, 8, 8,
                                6, 10, 12, 12, 2, 1, 4, 4, 13, 8, 12, 17, 8, 12, 1, 3, 13, 3, 18, 18,
                                13, 12, 3, 12, 2, 18, 12, 13, 2, 7, 13, 8, 2, 12, 3, 15, 1, 4, 17, 3,
                                7, 19, 18, 4, 18, 8, 18, 4, 6, 18, 7, 10, 17, 8, 5, 12, 8, 15, 7, 13]

        elif args.ds == 'cifar10':
            if args.cluster == 2:
                # coarse_label = [0, 0, 0, 0, 1, 0, 0, 1, 1, 1]
                coarse_label = [0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
        
        elif args.ds == "FARON":
            # kmeans
            if args.cluster == 5:
                coarse_label = [3, 3, 2, 0, 2, 2, 1, 3, 1, 2, 0, 2, 3, 2, 3, 0, 2, 0, 0, 0, 0, 0,
                                2, 0, 0, 0, 2, 1, 4, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 2, 1, 0, 
                                2, 0, 0, 2, 0, 3, 3, 2, 2, 0, 0, 2, 2, 2, 1, 2, 3, 3, 2, 2, 0, 0]
            elif args.cluster == 10:
                coarse_label = [9, 9, 3, 2, 3, 5, 4, 6, 4, 5, 2, 3, 7, 5, 6, 3, 5, 0, 2, 2, 2, 2, 
                                5, 2, 0, 2, 5, 4, 8, 3, 2, 2, 0, 0, 5, 2, 2, 2, 3, 5, 4, 3, 4, 2,
                                5, 1, 2, 1, 2, 6, 9, 5, 1, 2, 2, 3, 5, 5, 4, 3, 9, 9, 5, 1, 2, 2]
            elif args.cluster == 15:
                coarse_label = [3, 5, 0, 2, 0, 6, 12, 4, 12, 6, 2, 7, 10, 6, 4, 14, 6, 14, 8, 2, 13, 8,
                                6, 8, 14, 2, 6, 1, 9, 0, 2, 8, 14, 14, 6, 2, 2, 8, 0, 6, 12, 0, 1, 8, 6,
                                11, 13, 11, 8, 4, 5, 6, 0, 13, 8, 7, 6, 6, 1, 0, 0, 5, 6, 11, 2, 2]

        elif args.ds == 'miniimagenet':
            if args.cluster == 10:
                coarse_label = [5, 3, 2, 8, 1, 0, 5, 5, 6, 0, 0, 8, 4, 3, 5, 5, 4, 0, 0, 8,
                                5, 5, 2, 4, 4, 9, 8, 0, 5, 7, 4, 4, 4, 5, 0, 1, 0, 6, 6, 5,
                                5, 5, 2, 5, 8, 5, 5, 4, 6, 2, 2, 0, 7, 2, 4, 4, 1, 4, 8, 4,
                                6, 5, 2, 5, 3, 4, 4, 8, 9, 0, 1, 2, 8, 8, 1, 4, 4, 2, 4, 1,
                                3, 0, 5, 5, 5, 5, 0, 8, 1, 4, 0, 0, 5, 1, 3, 0, 5, 6, 2, 1]

        coarse = []
        for lb_idx in range(len(labels)):
            coarse.append(coarse_label[labels[lb_idx].item()])
        coarsetensor = torch.tensor(coarse).cuda()
        hl_std_all = []
        l1 = beta * (-0.5)
        l2 = beta * (-1)
        l3 = 0

        for lb_idx in range(len(labels)):
            hl_std = []
            std_coarse = coarse[lb_idx]
            std_fine = labels[lb_idx]
            for l in range(test_num_lst[inc_idx]):
                if coarse_label[l] == std_coarse and l == std_fine:
                    hl_std.append(l3)
                elif coarse_label[l] == std_coarse:
                    hl_std.append(l1)
                else:
                    hl_std.append(l2)
            hl_std2 = [x for x in hl_std]
            hl_std_all.append(hl_std2)
        
        hl_std_tensor = torch.tensor(hl_std_all).cuda().requires_grad_()
        hier_criterion = nn.KLDivLoss(reduction="batchmean")
        outputs_s = F.log_softmax(outputs_s, dim=1)
        hl_std_tensor = F.softmax(hl_std_tensor, dim=1)
        return hier_criterion(outputs_s, hl_std_tensor)
