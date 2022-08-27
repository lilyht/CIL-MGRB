import util
import xlrd
import pickle
import random
import numpy as np
import argparse

structure_path = "/HOME/scz1839/run/data/FARON/tree_15.xlsx"
random_seed = 36  # the normal class is learned in the initial phase
# random_seed = 1993  # the normal class is learned in the final phase
np.random.seed(random_seed)

info = {"train_num": [], "test_num": []}


def getdividedFARON(classnum_init, classnum_per_batch):
    '''
    Args:
        classnum_init (int): the number of firstly trained classes(n)
        classnum_per_batch (int): the number of classes delivered to a batch(m)
    '''
    TOTAL = (66 - classnum_init) // classnum_per_batch

    randlst = np.arange(66)
    np.random.shuffle(randlst)
    randlst = randlst.tolist()
    print("randlst: {}".format(randlst))

    info["order"] = randlst
    info["label_map"] = {}
    for idx in range(66):
        info["label_map"][randlst[idx]] = idx  # mapping
    print("label_map: {}".format(info["label_map"]))

    info["id2clsname"] = {}  # 0,1,2... -->n__, n__
    info["clsname2id"] = {}  # n__, n__ --> 0,1,2... 

    info["train_num"].append(classnum_init)
    info["test_num"].append(classnum_init)
    for i in range(TOTAL):
        info["train_num"].append(classnum_per_batch)
        info["test_num"].append(info["test_num"][-1] + classnum_per_batch)
    print("train_num: {}".format(info["train_num"]))
    print("test_num: {}".format(info["test_num"]))

    # load the node hierarchy in the xlsx file, which should use the mapped coarse-grained node
    wb = xlrd.open_workbook(structure_path)
    sh = wb.sheet_by_name('Sheet1')
    info["coarse_label"] = []
    for i in randlst:
        info["coarse_label"].append(int(sh.cell(i, 0).value - 1))
    print("coarse labels: {}".format(info["coarse_label"]))
    np.save('./save/FARON_divide_info_{}_{}.npy'.format(classnum_init, classnum_per_batch), info)


class ClsBalancedSample():
    '''
    Args:
        storage (int): total number of pictures can be stored
    Usage:
        called before loading the dataset
        will generate a file called 'FARON_num_info_n_m.npy'
    '''

    def __init__(self, storage, n, m):
        self.storage = storage
        self.val_proportion = 0.1
        self.n = n
        self.m = m
        self.group = (66 - n) // m + 1

    def combine(self):
        info = {"exemplar_num": [], "train_num": [], "val_num": []}

        for inc_idx in range(self.group):
            exemplar_num = int(np.floor(self.storage / (self.n + inc_idx * self.m)))
            val_num = int(np.floor(self.val_proportion * exemplar_num))
            if val_num == 0:
                val_num = 1
            train_num = int(exemplar_num - val_num)
            info["exemplar_num"].append(exemplar_num)
            info["train_num"].append(train_num)
            info["val_num"].append(val_num)
            print("val_num: {}, train_num: {}, exemplar_num: {}".format(val_num, train_num, exemplar_num))

        np.save('./save/FARON_num_info_{}_{}.npy'.format(self.n, self.m), info)
        return True


def main(k, n, m):
    getdividedFARON(n, m)

    cbs = ClsBalancedSample(k, n, m)
    cbs.combine()


if __name__ == '__main__':
    print("divide_FARON.py", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, required=False, default=990,
                        help='memory for FARON')
    parser.add_argument('-n', type=int, required=True, default=11,
                        help='number of the init classes')
    parser.add_argument('-m', type=int, required=True, default=11,
                        help='number of the classes incrementally transferred to model ')
    args = parser.parse_args()

    main(args.k, args.n, args.m)
