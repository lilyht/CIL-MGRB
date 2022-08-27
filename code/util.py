import os
import sys
import time
import torch
import importlib
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
from torch.optim.lr_scheduler import _LRScheduler

from dataset import *
import LoadFARON
import LoadImagenet100
import LoadMiniImagenet

base_path = "/HOME/scz1839/run/data"
FARON_path = base_path + '/FARON/FARON_3d_121.mat'
miniimagenet_path = base_path + '/mini-imagenet/processed'
imagenet100_path = base_path + '/data/seed_1993_subset_100_imagenet/data'
wv_from_bin = KeyedVectors.load_word2vec_format(base_path + "/google.bin", binary=True)  # C bin format


class WarmUpLR(_LRScheduler):
    """
    warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: total_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_structure(args):
    if args.ds == 'cifar100':
        classes = list(range(0, 100))
        datainfopath = './save/cifar100_divide_info_{}_{}.npy'.format(args.n, args.m)
        nodepathinfo = np.load(miniimagenet_path + '/nodepathinfo.npy', allow_pickle=True)
    elif args.ds == 'cifar10':
        classes = list(range(0, 10))
        datainfopath = './save/cifar10_divide_info_{}_{}.npy'.format(args.n, args.m)
        nodepathinfo = np.load(miniimagenet_path + '/nodepathinfo.npy', allow_pickle=True)
    elif args.ds == 'miniimagenet':
        classes = list(range(0, 100))
        datainfopath = './save/miniimagenet_divide_info_{}_{}.npy'.format(args.n, args.m)
        nodepathinfo = np.load(miniimagenet_path + '/re-nodepathinfo.npy', allow_pickle=True)
    elif args.ds == 'FARON':
        classes = list(range(0, 66))
        datainfopath = './save/FARON_divide_info_{}_{}.npy'.format(args.n, args.m)
        nodepathinfo = np.load(miniimagenet_path + '/re-nodepathinfo.npy', allow_pickle=True)
    elif args.ds == 'imagenet100':
        classes = list(range(0, 100))
        datainfopath = './save/imagenet100_divide_info_{}_{}.npy'.format(args.n, args.m)
        nodepathinfo = np.load(imagenet100_path + '/nodepathinfo.npy', allow_pickle=True)

    if args.cluster == -2:  # for miniimagenet and imagenet100
        print("using WordNet tree structure")
    elif args.cluster == -1:
        print("using original hierarchy (such as cifar100)")
    else:
        print("using semantic or visual information to generate tree structure({} clusters)".format(args.cluster))
    return classes, datainfopath, nodepathinfo


def vis_cluster(args, net, trainloader, label_map, cur_clsnum=20, n_clusters=5):
    """
    First calculate the class feature through the feature extractor
    Then cluster

    Args:
        args
        net: the current network
        trainloader: data
        cur_clsnum(int): number of appearing classes currently
        n_clusters(int): number of clusters
    Returns:
        coarse_label(list): the coarse label of classes from original class 0 to class cur_clsnum
    """
    clsid_vec = {}
    feature_dict = {}
    for i in range(cur_clsnum):
        feature_dict[i] = torch.zeros(1, 512)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].cuda(), data[1].cuda()
        # standard: torch.Size([128, 3, 32, 32]), torch.Size([128])
        # print("origin labels:", labels)
        if args.ds == 'FARON':
            inputs = inputs.unsqueeze(1)
            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = torch.squeeze(labels).long().cuda()
            # FARON: torch.Size([128, 1, 20, 121]) torch.Size([128])
        if args.ds == 'cifar100' or args.ds == 'cifar10' or args.ds == 'FARON':
            # convert label
            for lb_idx in range(len(labels)):
                labels[lb_idx] = label_map[labels[lb_idx].item()]
        outputs_t = net(inputs)  # get feature
        for lb_idx in range(len(labels)):
            feature_dict[labels[lb_idx].item()] = torch.cat((feature_dict[labels[lb_idx].item()],
                                                             outputs_t[lb_idx].unsqueeze(0).cpu()), 0)

    # calculate the mean feature vector for each class
    for i in range(cur_clsnum):
        id_allvectors = feature_dict[i]
        id_mean = torch.mean(id_allvectors, dim=0)
        id_mean = id_mean.view(-1)
        clsid_vec[i] = id_mean.tolist()

    for i in range(cur_clsnum):
        # join in order
        clsid = labellst[i]
        mean_vector.append(clsid_vec[clsid])

    X = np.array(mean_vector)
    if len(X) < n_clusters:
        n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    coarse_label = kmeans.labels_.tolist()
    return coarse_label


def getCluster(ds, labellst, cur_clsnum, n_clusters):
    '''
    KMeans
    Args: 
        ds(str): dataset
        labellst(list): the order of the dataset, defined in dividecifar100.py
        cur_clsnum(int): number of appearing classes currently
        n_clusters(int): number of clusters
    Returns:
        coarse_label(list): the coarse label of classes from original class 0 
        to class cur_clsnum
    '''
    label_names = []
    if ds == 'cifar10':
        label_names = cifar10_label_names
    elif ds == 'cifar100':
        label_names = cifar100_label_names
    elif ds == 'miniimagenet' or ds == 'imagenet100':
        label_names = mini_label_names

    # print("Current clustering: {} classes in total".format(cur_clsnum))
    veclst = []
    coarse_label = []
    tmp = []
    for i in range(cur_clsnum):
        tmp.append(label_names[labellst[i]])
        # print(label_names[labellst[i]], end=" ")
        vec = wv_from_bin[label_names[labellst[i]]]
        veclst.append(vec)
    X = np.array(veclst)
    if cur_clsnum < n_clusters:  # when the value of cur_clsnum is too small
        n_clusters = 1
    kmeans = KMeans(n_clusters, random_state=0).fit(X)
    # print(kmeans.labels_)  # <class 'numpy.ndarray'>
    # for idx in range(n_clusters):
    #     print("cluster {}: ".format(idx), end=" ")
    #     for j in range(cur_clsnum):
    #         if kmeans.labels_[j] == idx:
    #             print(label_names[labellst[j]], end=" ")
    #     print("")
    for idx in range(cur_clsnum):
        coarse_label.append(kmeans.labels_[idx])
    return coarse_label


def semantic_hier(args, labellst, curclsnum, cluster_num=2):
    '''
    coarse_label
    Args:
        args
        labellst(list): the randlst of the dataset, defined in dividecifar100.py
        cur_clsnum(int): number of appearing claases currently
        cluster_num(int): number of clusters
    Returns:
        coarse_label(list)
    '''
    coarse_label = getCluster(args.ds, labellst, curclsnum, cluster_num)
    return coarse_label


def FARONFeature_hier(args, labellst, cur_clsnum, n_clusters=15):
    # calculate the class mean
    accum = 0
    mean_vector = []
    id_num = {-1: 0, 0: 85, 1: 96, 2: 83, 3: 91, 4: 86, 5: 86, 6: 86, 7: 108, 8: 87, 9: 140,
              10: 138, 11: 288, 12: 652, 13: 135, 14: 145, 15: 137, 16: 137, 17: 174, 18: 160, 19: 172, 20: 176,
              21: 164, 22: 153, 23: 134, 24: 145, 25: 143, 26: 338, 27: 158, 28: 136, 29: 135, 30: 134, 31: 165,
              32: 135, 33: 135, 34: 134, 35: 145, 36: 151, 37: 141, 38: 102, 39: 93, 40: 208, 41: 145, 42: 140,
              43: 174, 44: 162, 45: 217, 46: 138, 47: 126, 48: 135, 49: 134, 50: 154, 51: 160, 52: 140, 53: 146,
              54: 200, 55: 136, 56: 149, 57: 140, 58: 139, 59: 126, 60: 161, 61: 162, 62: 164, 63: 165, 64: 138,
              65: 2681}
    m = loadmat(FARON_path)
    labellst = [65, 57, 60, 34, 62, 46, 15, 52, 11, 35, 28, 8, 12, 16, 49, 2, 29, 43, 17, 32, 4, 20,
                23, 18, 45, 7, 42, 10, 53, 39, 38, 61, 24, 3, 25, 47, 59, 63, 21, 64, 13, 19, 14, 54,
                48, 41, 1, 51, 22, 50, 56, 31, 27, 0, 36, 6, 55, 37, 9, 44, 26, 58, 33, 40, 30, 5]
    clsid_vec = {}
    train_set_T = torch.from_numpy(m['x_train']).permute(2, 0, 1)

    for i in range(66):
        start_vid = accum
        accum = accum + id_num[i]
        end_vid = accum
        id_allvectors = train_set_T[start_vid:end_vid]
        # print(id_allvectors.shape)
        id_mean = torch.mean(id_allvectors, dim=0)
        id_mean = id_mean.view(-1)
        clsid_vec[i] = id_mean.tolist()
        # mean_vector.append(id_mean.tolist())

    for i in range(cur_clsnum):
        # join in order
        clsid = labellst[i]
        mean_vector.append(clsid_vec[clsid])

    X = np.array(mean_vector)
    if len(X) < n_clusters:
        n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    coarse_label = kmeans.labels_.tolist()
    return coarse_label


def cal_height(args, cur_clsnum):
    """
    Calculate the depth of the structure composed of all current classes
    Args:
        args:
        cur_clsnum(int): number of appearing claases currently
    Returns:
        height(int): the height of the LCS Tree
    """
    classes, datainfopath, nodepathinfo = get_structure(args)
    clsid_path = nodepathinfo.item()["clsid_path"]

    path_collect = []
    minlen = 18
    maxlen = 0
    for i in range(cur_clsnum):
        path_i = clsid_path[i]
        path_collect.append(path_i)
        if minlen > len(path_i):
            minlen = len(path_i)
        if maxlen < len(path_i):
            maxlen = len(path_i)
    # print("minlen:", minlen, "maxlen:", maxlen)
    common = 1
    for i in range(0, minlen):
        flag = True
        cur_node = path_collect[0][i]
        for j in range(cur_clsnum):
            if path_collect[j][i] != cur_node:
                flag = False
                break
        if not flag:
            break
        else:
            common = common + 1
    height = maxlen - common
    print("height:", height)
    return height


def get_trainparameter(args):
    """
    Args:
        args
    Returns:
        totalclasses(int)
        lamdalst(list)
        milestones(list)
        epoch_1(int): the number of epochs in the first training stage
        epoch_2(int): the number of epochs in the second(balanced) training stage
    """
    if args.ds == 'cifar100':
        totalclasses = 100
        milestones = [70, 140, 210]
        epoch_1 = 250
        epoch_2 = 40
        epoch_1 = 2
        epoch_2 = 2

    elif args.ds == 'cifar10':
        totalclasses = 10
        milestones = [70, 140, 210]
        # milestones = [80, 140, 180]

        epoch_1 = 200
        epoch_2 = 40
        epoch_1 = 2
        epoch_2 = 2

    elif args.ds == 'FARON':
        totalclasses = 66
        milestones = [20, 120, 180]
        milestones = [100, 200]
        # milestones = [5, 10, 15, 20]
        # 60 10
        epoch_1 = 40
        epoch_2 = 15

    elif args.ds == 'miniimagenet':
        totalclasses = 100
        # milestones = [30, 60, 90, 120] 
        milestones = [100, 120]
        epoch_1 = 160
        epoch_2 = 30
        epoch_1 = 2
        epoch_2 = 2

    elif args.ds == 'imagenet100':
        totalclasses = 100
        milestones = [30, 60, 90]
        epoch_1 = 120
        epoch_2 = 30

    group = (totalclasses - args.n) // args.m + 1
    lamdalst = [0]
    inc_phase_num = (totalclasses - args.n) // args.m
    for it in range(inc_phase_num):
        lamdalst.append((args.n + args.m * it) / (args.n + args.m * (it + 1)))

    print("--------------------------------------------------")
    print("{0:15}{1:15}".format("epoch_1", str(epoch_1)))
    print("{0:15}{1:15}".format("epoch_2", str(epoch_2)))
    print("{0:15}{1:15}".format("milestones", str(milestones)))
    print("{0:15}{1:15}".format("group", str(group)))
    print("--------------------------------------------------")
    return totalclasses, lamdalst, milestones, epoch_1, epoch_2, group


def get_trainoptim(args, milestones, net_t, clsnet):
    """
    Args:
        args
        milestones (list)
        net_t: feature extractor
        clsnet: classifier
    Returns:
        optimizer, clsoptimizer
    """
    if args.ds == 'cifar100':
        # optimizer = optim.SGD(net_t.parameters(), lr=args.lr, momentum=0.9,weight_decay=2e-4)
        # clsoptimizer = optim.SGD(clsnet.parameters(), lr=args.lr, momentum=0.9,weight_decay=2e-4)
        optimizer = optim.AdamW(net_t.parameters(), lr=args.lr, weight_decay=0.05)
        clsoptimizer = optim.AdamW(clsnet.parameters(), lr=args.lr, weight_decay=0.05)
    elif args.ds == 'cifar10':
        # optimizer = optim.SGD(net_t.parameters(), lr=args.lr, momentum=0.9,weight_decay=5e-4)
        # clsoptimizer = optim.SGD(net_t.parameters(), lr=args.lr, momentum=0.9,weight_decay=5e-4)
        optimizer = optim.AdamW(net_t.parameters(), lr=args.lr, weight_decay=0.05)
        clsoptimizer = optim.AdamW(clsnet.parameters(), lr=args.lr, weight_decay=0.05)
    elif args.ds == 'FARON':
        optimizer = optim.Adam(net_t.parameters(), lr=args.lr)
        clsoptimizer = optim.Adam(clsnet.parameters(), lr=args.lr)
    else:
        # optimizer = optim.SGD(net_t.parameters(), lr=args.lr, momentum=0.9,weight_decay=5e-4)
        # clsoptimizer = optim.SGD(clsnet.parameters(), lr=args.lr, momentum=0.9,weight_decay=5e-4)
        optimizer = optim.AdamW(net_t.parameters(), lr=args.lr, weight_decay=0.05)
        clsoptimizer = optim.AdamW(clsnet.parameters(), lr=args.lr, weight_decay=0.05)

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lrdecay)
    cls_train_scheduler = optim.lr_scheduler.MultiStepLR(clsoptimizer, milestones=milestones, gamma=args.lrdecay)
    print("setting train_scheduler...")
    return optimizer, clsoptimizer, train_scheduler, cls_train_scheduler


def get_clsnetoptim(args, milestones, clsnet):
    """
    Returns:
        clsoptimizer
    """
    if args.ds == 'cifar100':
        # clsoptimizer = optim.SGD(clsnet.parameters(), lr=args.lr, momentum=0.9,weight_decay=2e-4)
        clsoptimizer = optim.AdamW(clsnet.parameters(), lr=args.lr, weight_decay=0.05)
    elif args.ds == 'cifar10':
        # clsoptimizer = optim.SGD(clsnet.parameters(), lr=args.lr, momentum=0.9,weight_decay=5e-4)
        clsoptimizer = optim.AdamW(clsnet.parameters(), lr=args.lr, weight_decay=0.05)
    elif args.ds == 'FARON':
        clsoptimizer = optim.Adam(clsnet.parameters(), lr=args.lr)
    else:
        # clsoptimizer = optim.SGD(clsnet.parameters(), lr=args.lr, momentum=0.9,weight_decay=5e-4)
        clsoptimizer = optim.AdamW(clsnet.parameters(), lr=args.lr, weight_decay=0.05)
    clstrain_scheduler = optim.lr_scheduler.MultiStepLR(clsoptimizer, milestones=milestones, gamma=args.lrdecay)
    return clsoptimizer, clstrain_scheduler


def get_valid_dataset(args, inc_idx):
    """
    For cifar100 or cifar10
    Load a balanced validation set
    """
    if args.ds == 'cifar100':
        return cifar100_valid_dataset(args, inc_idx)
    elif args.ds == 'cifar10':
        return cifar10_valid_dataset(args, inc_idx)
    return True


def get_dataset(args, inc_idx, modeltype='old'):
    print("dataset: {}".format(args.ds))
    if args.ds == 'cifar10':
        return split_cifar10_dataset_load_batch(args, inc_idx)
    elif args.ds == 'cifar100':
        return split_cifar100_dataset_load_batch(args, inc_idx)
    elif args.ds == 'miniimagenet':
        if modeltype == 'old':
            return LoadMiniImagenet.get_mini_imagenet_dataset(args, 'old', inc_idx)
        elif modeltype == 'new':
            return LoadMiniImagenet.get_mini_imagenet_dataset(args, 'new', inc_idx)
    elif args.ds == 'FARON':
        if modeltype == 'old':
            return LoadFARON.get_FARON_dataset(args, 'old', inc_idx)
        elif modeltype == 'new':
            return LoadFARON.get_FARON_dataset(args, 'new', inc_idx)
    elif args.ds == 'imagenet100':
        if modeltype == 'old':
            return LoadImagenet100.get_imagenet100_dataset(args, 'old', inc_idx)
        elif modeltype == 'new':
            return LoadImagenet100.get_imagenet100_dataset(args, 'new', inc_idx)
    else:
        print("No matching dataset!")


def get_dataloader(args, inc_idx):
    if inc_idx == 0:
        if args.ds == 'cifar100':
            trainloader, testloader = get_dataset(args, inc_idx)
        elif args.ds == 'cifar10':
            trainloader, testloader = get_dataset(args, inc_idx)
        elif args.ds == 'FARON':
            trainloader, testloader = LoadFARON.get_FARON_dataset(args, 'old', inc_idx)
        elif args.ds == 'miniimagenet':
            trainloader, testloader = LoadMiniImagenet.get_mini_imagenet_dataset(args, 'old', inc_idx)
        elif args.ds == 'imagenet100':
            trainloader, testloader = LoadImagenet100.get_imagenet100_dataset(args, 'old', inc_idx)
        validloader = None
    else:
        if args.ds == 'cifar100':
            trainloader, testloader = get_dataset(args, inc_idx)
            validloader = get_valid_dataset(args, inc_idx - 1)
        elif args.ds == 'cifar10':
            trainloader, testloader = get_dataset(args, inc_idx)
            validloader = get_valid_dataset(args, inc_idx - 1)
        elif args.ds == 'FARON':
            trainloader, validloader, testloader = LoadFARON.get_FARON_dataset(args, 'new', inc_idx - 1)
        elif args.ds == 'miniimagenet':
            trainloader, validloader, testloader = LoadMiniImagenet.get_mini_imagenet_dataset(args, 'new', inc_idx - 1)
        elif args.ds == 'imagenet100':
            trainloader, validloader, testloader = LoadImagenet100.get_imagenet100_dataset(args, 'new', inc_idx - 1)
    return trainloader, validloader, testloader


def get_jointdataloader(args, inc_idx, modeltype='old'):
    ds = args.ds
    print("dataset: {}".format(ds))
    if ds == 'cifar10':
        return cifar10_dataset_joint(args, inc_idx)
    elif ds == 'cifar100':
        return cifar100_dataset_joint(args, inc_idx)
    elif ds == 'FARON':
        return LoadFARON.get_FARON_dataset(args, 'old', inc_idx)
    elif ds == 'miniimagenet':
        if modeltype == 'old':
            return LoadMiniImagenet.get_mini_imagenet_dataset(args, 'old', inc_idx)
        elif modeltype == 'new':
            return LoadMiniImagenet.get_mini_imagenet_dataset(args, 'new', inc_idx)
    else:
        print("No matching dataset!")


def setup_gpu(args, os):
    if args.gpu != "cpu":
        args.gpu = ",".join([c for c in args.gpu])
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Using device : ", args.gpu)


cifar10_label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cifar100_label_names = [
    'apple', 'fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

cifar100_label_names_sort = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor scenes', 'large_omnivores_and_herbivores',
    'medium-sized_mammals', 'non-insect_invertebrates', 'people',
    'reptiles', 'small_mammals', 'trees', 'vehicles_1', 'vehicles_2',
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'fish', 'flatfish', 'ray', 'shark', 'trout',
    'orchid', 'poppy', 'rose', 'sunflower', 'tulip',
    'bottle', 'bowl', 'can', 'cup', 'plate',
    'apple', 'mushroom', 'orange', 'pear', 'pepper',
    'clock', 'keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
    'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',
    'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'
]

mini_label_names = [
    'fish', 'goose', 'missile', 'cabinet', 'bars', 'dog', 'clog', 'carousel', 'poncho', 'curtain',
    'ladybug', 'snorkel', 'ferret', 'holster', 'dustcart', 'frying_pan', 'reel', 'Tibetan_mastiff', 'consomme',
    'breastplate',
    'container', 'spider_web', 'chime', 'dog', 'dog', 'crate', 'miniskirt', 'trifle', 'barrel', 'stage',
    'dog', 'oboe', 'bookshop', 'dalmatian', 'dugong', 'bus', 'orange', 'slot', 'miniature_poodle', 'lock',
    'bolete', 'jellyfish', 'bar', 'vase', 'ashcan', 'golden_retriever', 'dome', 'iPod', 'tank', 'carton',
    'dog', 'daddy_longlegs', 'ant', 'fox', 'dog', 'dog', 'finch', 'scoreboard', 'spike', 'wok',
    'photocopier', 'robin', 'accessories', 'rhinoceros_beetle', 'furnace', 'boxer', 'catamaran', 'bottle', 'dog',
    'dishrag',
    'triceratops', 'unicycle', 'lipstick', 'fence', 'electric_guitar', 'yawl', 'sloth', 'nematode', 'pipe_organ',
    'malamute',
    'tobacconist_shop', 'hourglass', 'king_crab', 'toucan', 'meerkat', 'cliff', 'sign', 'roof', 'rug', 'green_mamba',
    'wolf', 'lion', 'upright_piano', 'coral_reef', 'hotdog', 'aircraft_carrier', 'fireguard', 'cocktail_shaker',
    'cannon', 'mixing_bowl']
