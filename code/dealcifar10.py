from setup import *
import pickle
import random
import numpy as np
import argparse

'''
generate cifar10_divide_info_n_m.npy, test_x and train_sample and val_sample
'''

ori_train_path = "/HOME/scz1839/run/data/cifar10/cifar-10-batches-py/data_batch_"
test_filepath = '/HOME/scz1839/run/data/cifar10/cifar-10-batches-py/test_batch'
modeify_DATASET_ROOT = "./data/cifar10/cifar-10-batches-py/"
random_seed = 1993
np.random.seed(random_seed)
exemplar_train_lst = []
exemplar_val_lst = []
info = {"train_num": [], "test_num": []}  # create a dict


def init(k, n, m):
    global exemplar_train_lst
    global exemplar_val_lst
    group = (10 - n) // m + 1
    exemplar_train_lst.append(-1)
    exemplar_val_lst.append(-1)

    for inc_idx in range(group):
        exemplar_num = 200
        val_proportion = 0.1
        val_num = int(np.floor(val_proportion * exemplar_num))
        train_num = int(exemplar_num - val_num)
        exemplar_train_lst.append(train_num)
        exemplar_val_lst.append(val_num)
    print(exemplar_train_lst)
    print(exemplar_val_lst)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def reserveclass(randlst, startidx, endidx, tag, savetestpath):
    """
    Args:
        randlst (list): class order
        startidx (int): begin
        endidx (int): end
        tag (int): the tag_th batch
    """

    # divide testdata
    classlst = randlst[endidx:]
    total = len(classlst)
    testobj = unpickle(test_filepath)
    trdata_lst = testobj['data'].tolist()  # 50000*3072 (3072 = 32*32*3)

    for idx in range(10000):
        for i in range(total):
            if testobj['labels'][idx] == classlst[i]:
                testobj['labels'][idx] = -100  # mark as a special value
                testobj['filenames'][idx] = '#'
                trdata_lst[idx] = [-100]
                break

    while -100 in testobj['labels']:
        testobj['labels'].remove(-100)
    while [-100] in trdata_lst:
        trdata_lst.remove([-100])
    while '#' in testobj['filenames']:
        testobj['filenames'].remove('#')

    testobj['data'] = np.array(trdata_lst)
    print("The dataset for test consists of {} 32x32 colour images in {} classes".format(len(testobj['labels']),
                                                                                         10 - total))
    print("test classes in {} batch: {}\n".format(tag, randlst[:endidx]))
    info["test_batch_{}".format(tag)] = randlst[:endidx]
    with open(savetestpath + "test_" + str(tag), 'wb') as p:
        pickle.dump(testobj, p)


def getdividedCifar(classnum_init, classnum_per_batch, savetestpath, isrand=False):
    """
    Args:
        classnum_init (int): the number of firstly trained classes, n
        classnum_per_batch (int): the number of classes delivered to a batch, m
        isrand (bool): randomly divide or not

    """
    TOTAL = (10 - classnum_init) // classnum_per_batch

    randlst = np.arange(10)
    np.random.shuffle(randlst)
    randlst = randlst.tolist()  # [4, 2, 7, 6, 0, 3, 5, 8, 9, 1]
    # randlst = [4, 2, 7, 6, 0, 8, 5, 3, 9, 1]
    print(randlst)

    info["order"] = randlst
    info["label_map"] = {}
    for idx in range(10):
        info["label_map"][randlst[idx]] = idx  # mapping

    reserveclass(randlst, 0, classnum_init, 0, savetestpath)
    info["train_num"].append(classnum_init)
    info["test_num"].append(classnum_init)

    for i in range(TOTAL):
        startidx = classnum_init + i * classnum_per_batch
        endidx = classnum_init + (i + 1) * classnum_per_batch
        reserveclass(randlst, startidx, endidx, i + 1, savetestpath)
        info["train_num"].append(classnum_per_batch)
        info["test_num"].append(info["test_num"][-1] + classnum_per_batch)

    np.save('./save/cifar10_divide_info_{}_{}.npy'.format(classnum_init, classnum_per_batch), info)


# create a file for each class
def divide_cls(n, m):
    datainfopath = './save/cifar10_divide_info_{}_{}.npy'.format(n, m)
    datainfo = np.load(datainfopath, allow_pickle=True)
    label_map = datainfo.item()["label_map"]

    for i in range(10):
        clasdiv_obj, clsdiv_data_lst = set_empty_c10()

        for bsnum in range(1, 6):
            ori_train_path_ = ori_train_path + str(bsnum)
            trainobj = unpickle(ori_train_path_)
            trdata_lst = trainobj['data'].tolist()

            for idx in range(10000):
                if label_map[trainobj['labels'][idx]] == i:
                    clasdiv_obj['labels'].append(trainobj['labels'][idx])
                    clasdiv_obj['filenames'].append(trainobj['filenames'][idx])
                    clsdiv_data_lst.append(trdata_lst[idx])

        clasdiv_obj['data'] = np.array(clsdiv_data_lst)

        with open(modeify_DATASET_ROOT + 'train_class_' + str(i), 'wb') as p:
            pickle.dump(clasdiv_obj, p)

    for i in range(10):
        filepath = modeify_DATASET_ROOT + "train_class_" + str(i)
        trainobj = unpickle(filepath)
        if len(trainobj['labels']) != 5000:
            print("class", i, "error")

    return True


class GenerateExemplars():
    def __init__(self, n, m, inc_idx):
        datainfopath = './save/cifar10_divide_info_{}_{}.npy'.format(n, m)
        datainfo = np.load(datainfopath, allow_pickle=True)
        self.train_num_lst = datainfo.item()["train_num"]  # [2, 2, 2, 2, 2]
        self.test_num_lst = datainfo.item()["test_num"]  # [2, 4, 6, 8, 10]

        self.inc_idx = inc_idx
        self.newclassnum = self.train_num_lst[inc_idx]
        self.oldclassnum = self.test_num_lst[inc_idx] - self.train_num_lst[inc_idx]
        self.totalclassnum = self.test_num_lst[inc_idx]

        print("self.oldclassnum: ", format(self.oldclassnum))
        print("self.newclassnum: ", format(self.newclassnum))
        print("self.totalclassnum: ", format(self.totalclassnum))

    def combine(self, inc_idx):
        """
        For train and val
        combine the required classes into a file. The file is stored under /data/
        Args:
            inc_idx (int): 1, 2, 3, 4
        """

        cifar_10_train_filepath = modeify_DATASET_ROOT + 'train_class_'
        clasdiv_obj, clsdiv_data_lst = set_empty_c10()
        train_num = exemplar_train_lst[inc_idx]
        val_num = exemplar_val_lst[inc_idx]
        print("exemplar_train_num: {}, exemplar_val_num: {}".format(exemplar_train_lst[inc_idx],
                                                                    exemplar_val_lst[inc_idx]))

        # exemplar data(for balance validation)
        for i in range(self.totalclassnum):
            trainobj = unpickle(cifar_10_train_filepath + str(i))
            trdata_lst = trainobj['data'].tolist()
            clasdiv_obj['labels'] = clasdiv_obj['labels'] + trainobj['labels'][0:val_num]
            clasdiv_obj['filenames'] = clasdiv_obj['filenames'] + trainobj['filenames'][0:val_num]
            clsdiv_data_lst = clsdiv_data_lst + trdata_lst[0:val_num]
        clasdiv_obj['data'] = np.array(clsdiv_data_lst)
        with open(modeify_DATASET_ROOT + 'val_sample_' + str(self.inc_idx - 1), 'wb') as p:
            pickle.dump(clasdiv_obj, p)

        # exemplar data(for balance train)
        clasdiv_obj, clsdiv_data_lst = set_empty_c10()
        for i in range(self.oldclassnum):
            trainobj = unpickle(cifar_10_train_filepath + str(i))
            trdata_lst = trainobj['data'].tolist()
            clasdiv_obj['labels'] = clasdiv_obj['labels'] + trainobj['labels'][val_num: train_num + val_num]
            clasdiv_obj['filenames'] = clasdiv_obj['filenames'] + trainobj['filenames'][val_num: train_num + val_num]
            clsdiv_data_lst = clsdiv_data_lst + trdata_lst[val_num: train_num + val_num]
        clasdiv_obj['data'] = np.array(clsdiv_data_lst)
        with open(modeify_DATASET_ROOT + 'train_sample_' + str(self.inc_idx - 1), 'wb') as p:
            pickle.dump(clasdiv_obj, p)
        return True

    def combine2(self, inc_idx):
        train_num = exemplar_train_lst[inc_idx + 1]
        val_num = exemplar_val_lst[inc_idx + 1]
        cifar_10_train_filepath = modeify_DATASET_ROOT + 'train_class_'

        # train_x
        clasdiv_obj, clsdiv_data_lst = set_empty_c10()
        for i in range(self.totalclassnum - self.train_num_lst[inc_idx], self.totalclassnum):
            trainobj = unpickle(cifar_10_train_filepath + str(i))
            trdata_lst = trainobj['data'].tolist()
            clasdiv_obj['labels'] = clasdiv_obj['labels'] + trainobj['labels'][val_num:]
            clasdiv_obj['filenames'] = clasdiv_obj['filenames'] + trainobj['filenames'][val_num:]
            clsdiv_data_lst = clsdiv_data_lst + trdata_lst[val_num:]
        clasdiv_obj['data'] = np.array(clsdiv_data_lst)
        with open(modeify_DATASET_ROOT + 'train_' + str(self.inc_idx), 'wb') as p:
            pickle.dump(clasdiv_obj, p)

        return True


def main(k, n, m):
    init(k, n, m)
    group = (10 - n) // m + 1
    getdividedCifar(n, m, modeify_DATASET_ROOT)  # test
    divide_cls(n, m)

    for inc_idx in range(group):
        g = GenerateExemplars(n, m, inc_idx)
        if inc_idx != 0:
            g.combine(inc_idx)  # train_sample_ and val_sample_
        g.combine2(inc_idx)  # train_ and test_
    print("Finished!")


if __name__ == '__main__':
    print("dealcifar10.py", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, required=False, default=2000,
                        help='memory size')
    parser.add_argument('-n', type=int, required=True, default=2,
                        help='number of the init classes, please input like this: -n 2')
    parser.add_argument('-m', type=int, required=True, default=2,
                        help='number of the classes incrementally transferred to model ')
    args = parser.parse_args()

    main(args.k, args.n, args.m)
