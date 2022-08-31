import sys
import math
import util
import argparse


def ClassBalancedWeight(args, inc_idx, totalclasses, test_num_lst):
    """
    Calculate the class balanced weight
    Args:
        args
        inc_idx (int)
        test_num_lst (list)
    Return:
        CB loss
    """

    tmp = (totalclasses - args.n) // args.m  # total number of incremental learning phases
    if args.ds == "cifar100" or args.ds == "miniimagenet":
        train_sample_lst = [500]
        for i in range(tmp):
            train_sample_lst.append(18)

    elif args.ds == "cifar10":
        train_sample_lst = [5000]
        for i in range(tmp):
            train_sample_lst.append(180)

    elif args.ds == "imagenet100":
        train_sample_lst = [1300]
        for i in range(tmp):
            train_sample_lst.append(18)

    elif args.ds == "FARON":
        if args.n == 11 and args.m == 11:
            # train_sample_lst = [100, 27, 14, 9, 6, 5, 4]  # k=330
            train_sample_lst = [100, 54, 27, 18, 14, 11, 9]  # k=66
            exemplar_train_num = [-1, 655, 327, 218, 163, 131, 109]
            exemplar_val_num = [-1, 72, 36, 24, 18, 14, 12]
        elif args.n == 22 and args.m == 22:
            train_sample_lst = [100, 41, 20, 14]  # k=990
            exemplar_train_num = [-1, 327, 163, 109]
            exemplar_val_num = [-1, 36, 18, 12]
        elif args.n == 33 and args.m == 33:
            train_sample_lst = [100, 18, 9]  # k=66
            exemplar_train_num = [-1, 218, 109]
            exemplar_val_num = [-1, 24, 12]
        train_count = {0: 2681, 1: 140, 2: 161, 3: 134, 4: 164, 5: 138, 6: 137, 7: 140, 8: 288, 9: 145, 10: 136, 11: 87,
                       12: 652, 13: 137, 14: 134, 15: 83, 16: 135, 17: 174, 18: 174, 19: 135, 20: 86, 21: 176, 22: 134,
                       23: 160, 24: 217, 25: 108, 26: 140, 27: 138, 28: 146, 29: 93, 30: 102, 31: 162, 32: 145, 33: 91,
                       34: 143, 35: 126, 36: 126, 37: 165, 38: 164, 39: 138, 40: 135, 41: 172, 42: 145, 43: 200,
                       44: 135, 45: 145, 46: 96, 47: 160, 48: 153, 49: 154, 50: 149, 51: 165, 52: 158, 53: 85, 54: 151,
                       55: 86, 56: 136, 57: 141, 58: 140, 59: 162, 60: 338, 61: 139, 62: 135, 63: 208, 64: 134,
                       65: 86}  # seed = 36

    w_lst = []
    if args.cbloss == True:
        if args.ds == "FARON":
            w_lst = []
            eta = (totalclasses - 1) / totalclasses
            for i in range(test_num_lst[inc_idx] - args.m):
                if train_count[i] < exemplar_train_num[inc_idx]:
                    # make sure val is a balanced sample set, and the rest are used for training
                    zhishu = train_count[i] - exemplar_val_num[inc_idx]
                else:
                    zhishu = exemplar_train_num[inc_idx]
                w_lst.append((1 - eta) / (1 - math.pow(eta, zhishu)))

            for i in range(test_num_lst[inc_idx] - args.m, test_num_lst[inc_idx]):
                if train_count[i] < exemplar_val_num[inc_idx]:
                    zhishu = train_count[i] - exemplar_val_num[inc_idx] // 2
                else:
                    zhishu = train_count[i] - exemplar_val_num[inc_idx]
                w_lst.append((1 - eta) / (1 - math.pow(eta, zhishu)))
        else:
            w_lst = []
            eta = (totalclasses - 1) / totalclasses
            for i in range(test_num_lst[inc_idx] - args.m):
                w_lst.append((1 - eta) / (1 - math.pow(eta, train_sample_lst[inc_idx])))
            for i in range(test_num_lst[inc_idx] - args.m, test_num_lst[inc_idx]):
                w_lst.append((1 - eta) / (1 - math.pow(eta, train_sample_lst[0])))

    return w_lst
