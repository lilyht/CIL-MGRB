import pickle
import random
import argparse
import numpy as np

random_seed = 1993
np.random.seed(random_seed)

mini_imagenet_path = '/HOME/scz1839/run/data/mini-imagenet/processed'
test_filepath = mini_imagenet_path + '/test'
train_filepath = mini_imagenet_path + '/train'
wnid_path = '/re-wnids.txt'

info = {"train_num": [], "test_num": []}


def getdividedMiniImagenet(classnum_init, classnum_per_batch):
    """
    Args:
        classnum_init (int): the number of firstly trained classes(n)
        classnum_per_batch (int): the number of classes delivered to a batch(m)
    Usage:
        generate a file called 'miniimagenet_divide_info_n_m.npy'
    """

    TOTAL = (100 - classnum_init) // classnum_per_batch

    randlst = np.arange(100)
    np.random.shuffle(randlst)
    randlst = randlst.tolist()
    print("randlst: {}".format(randlst))

    info["order"] = randlst
    info["label_map"] = {}
    for idx in range(100):
        info["label_map"][randlst[idx]] = idx  # map
    print("label_map: {}".format(info["label_map"]))

    info["id2clsname"] = {}  # 0,1,2... -->n__, n__
    info["clsname2id"] = {}  # n__, n__ --> 0,1,2...

    with open(mini_imagenet_path + wnid_path) as wnid:
        i = 0
        for line in wnid:
            info["id2clsname"][i] = line.strip('\n')
            info["clsname2id"][line.strip('\n')] = i
            i = i + 1
    print("id2clsname: {}".format(info["id2clsname"]))
    print("clsname2id: {}".format(info["clsname2id"]))

    info["train_num"].append(classnum_init)
    info["test_num"].append(classnum_init)
    for i in range(TOTAL):
        info["train_num"].append(classnum_per_batch)
        info["test_num"].append(info["test_num"][-1] + classnum_per_batch)
    print("train_num: {}".format(info["train_num"]))
    print("test_num: {}".format(info["test_num"]))

    info["coarse_label"] = []
    for i in range(100):
        info["coarse_label"].append(1)
    print("coarse labels: {}".format(info["coarse_label"]))

    np.save('./save/miniimagenet_divide_info_{}_{}.npy'.format(classnum_init, classnum_per_batch), info)


class GenerageInfo():
    """
    Args:
        storage (int): total number of pictures can be stored
    Usage:
        called before loading the dataset
        will generate a file called 'miniimagenet_num_info_n_m.npy'
    """

    def __init__(self, storage, n, m):
        self.storage = storage
        self.val_proportion = 0.1
        self.n = n
        self.m = m
        self.group = (100 - n) // m + 1

    def get_num_info(self):
        info = {"exemplar_num": [], "train_num": [], "val_num": []}

        for inc_idx in range(self.group):
            exemplar_num = 20
            val_num = int(np.floor(self.val_proportion * exemplar_num))
            train_num = int(exemplar_num - val_num)
            info["exemplar_num"].append(exemplar_num)
            info["train_num"].append(train_num)
            info["val_num"].append(val_num)
            print("val_num: {}, train_num: {}, exemplar_num: {}".format(val_num, train_num, exemplar_num))

        np.save('./save/miniimagenet_num_info_{}_{}.npy'.format(self.n, self.m), info)


def main(k, n, m):
    getdividedMiniImagenet(n, m)

    cbs = GenerageInfo(k, n, m)
    cbs.get_num_info()


if __name__ == '__main__':
    print("divideminiimagenet.py", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, required=False, default=2000,
                        help='memory for miniimagenet')
    parser.add_argument('-n', type=int, required=True, default=20,
                        help='number of the init classes')
    parser.add_argument('-m', type=int, required=True, default=20,
                        help='number of the classes incrementally transferred to model')
    args = parser.parse_args()

    main(args.k, args.n, args.m)
