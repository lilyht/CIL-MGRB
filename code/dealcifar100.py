from setup import *
import pickle
import numpy as np
import argparse
import util

'''
generate train_sample and val_sample
'''

random_seed = 1993
np.random.seed(random_seed)
origin_CIFAR_PATH_ROOT = '/HOME/scz1839/run/data/cifar-100/'  # path of original dataset
modeify_DATASET_ROOT = "./data/cifar100/cifar-100-python/"  # path of the divided datasets stored
origin_dataset_root = origin_CIFAR_PATH_ROOT + "cifar-100-python/"
cifar_100_train_perclass = modeify_DATASET_ROOT + "train_class_"
test_filepath = origin_CIFAR_PATH_ROOT + 'cifar-100-python/test'

exampler_train_lst = []
exampler_val_lst = []
datainfopath = ""
info = {"train_num": [], "test_num": []}


def init(k, n, m):
    global exampler_train_lst
    global exampler_val_lst
    global datainfopath
    datainfopath = './save/cifar100_divide_info_{}_{}.npy'.format(n, m)

    itera = (100 - n) // m + 1
    exampler_train_lst.append(-1)
    exampler_val_lst.append(-1)

    for i in range(itera):
        exampler_train_lst.append(18)
        exampler_val_lst.append(2)

    print(exampler_train_lst)
    print(exampler_val_lst)


def unpickle(file):
    # items in trainobj: filenames, batch_label, fine_labels, coarse_labels, data
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def reserveclass(trainlst, startidx, endidx, tag, savetestpath):
    '''
    Args:
        trainlst (list): the classes to be trained
        startidx (int): begin
        endidx (int): end
        tag (int): the tag_th batch
    '''

    testlst = trainlst
    classlst = trainlst[:startidx] + trainlst[endidx:]
    total = len(classlst)

    # divide testdata
    classlst = testlst[endidx:]
    total = len(classlst)
    testobj = unpickle(test_filepath)
    trdata_lst = testobj['data'].tolist()  # 50000*3072 (3072 = 32*32*3)

    for idx in range(10000):
        for i in range(total):
            if testobj['fine_labels'][idx] == classlst[i]:
                testobj['fine_labels'][idx] = -100  # mark with a special value
                testobj['coarse_labels'][idx] = -100
                testobj['filenames'][idx] = '#'
                trdata_lst[idx] = [-100]
                break

    while -100 in testobj['fine_labels']:
        testobj['fine_labels'].remove(-100)
    while -100 in testobj['coarse_labels']:
        testobj['coarse_labels'].remove(-100)
    while [-100] in trdata_lst:
        trdata_lst.remove([-100])
    while '#' in testobj['filenames']:
        testobj['filenames'].remove('#')

    testobj['data'] = np.array(trdata_lst)
    print("The dataset for test consists of {} 32x32 colour images in {} classes".format(
        len(testobj['fine_labels']), 100 - total))
    print("test classes in {} batch: {}\n".format(tag, testlst[:endidx]))
    info["test_batch_{}".format(tag)] = testlst[:endidx]
    with open(savetestpath + "/test_" + str(tag), 'wb') as p:
        pickle.dump(testobj, p)


def getdividedCifar(n, m, savetestpath):
    '''
    Args:
        n (int): the number of firstly trained classes
        m (int): the number of classes delivered to a batch
        isrand (bool): randomly divide or not
    '''
    TOTAL = (100 - n) // m
    randlst = np.arange(100)
    np.random.shuffle(randlst)
    randlst = randlst.tolist()
    print(randlst)

    info["order"] = randlst
    info["label_map"] = {}
    for idx in range(100):
        info["label_map"][randlst[idx]] = idx  # mapping

    # split test set
    info["train_num"].append(n)
    info["test_num"].append(n)
    reserveclass(randlst, 0, n, 0, savetestpath)
    for i in range(TOTAL):
        startidx = n + i * m
        endidx = n + (i + 1) * m
        reserveclass(randlst, startidx, endidx, i + 1, savetestpath)
        info["train_num"].append(m)
        info["test_num"].append(info["test_num"][-1] + m)

    # record coarse-grained labels
    cato = []
    info["coarse_label"] = []
    for i in range(100):
        for item in util.cifar100_label_names_sort:
            if item == util.cifar100_label_names[randlst[i]]:
                cato.append(util.cifar100_label_names_sort.index(item))
    for i in range(100):
        info["coarse_label"].append((cato[i] - 20) // 5)
    print("coarse labels: {}".format(info["coarse_label"]))

    np.save('./save/cifar100_divide_info_{}_{}.npy'.format(n, m), info)


def divide_cls(n, m):
    # split train set, created a file for each class
    datainfo = np.load(datainfopath, allow_pickle=True)
    label_map = datainfo.item()["label_map"]

    trainobj = unpickle(origin_dataset_root + "train")
    trdata_lst = trainobj['data'].tolist()
    print(len(trdata_lst))

    for i in range(100):
        clasdiv_obj, clsdiv_data_lst = set_empty()

        for idx in range(50000):
            if label_map[trainobj['fine_labels'][idx]] == i:
                clasdiv_obj['fine_labels'].append(trainobj['fine_labels'][idx])
                clasdiv_obj['coarse_labels'].append(trainobj['coarse_labels'][idx])
                clasdiv_obj['filenames'].append(trainobj['filenames'][idx])
                clsdiv_data_lst.append(trdata_lst[idx])

        clasdiv_obj['data'] = np.array(clsdiv_data_lst)

        with open(cifar_100_train_perclass + str(i), 'wb') as p:
            pickle.dump(clasdiv_obj, p)

    for i in range(100):
        filepath = cifar_100_train_perclass + str(i)
        trainobj = unpickle(filepath)
        if len(trainobj['fine_labels']) != 500:
            print("An error occurred in class", i)
    print("Finish splitting!")
    return True


class GenerateExemplars():
    def __init__(self, storage, inc_idx):
        datainfo = np.load(datainfopath, allow_pickle=True)
        # label_map = datainfo.item()["label_map"]
        self.train_num_lst = datainfo.item()["train_num"]  # [20, 20, 20, 20, 20]
        self.test_num_lst = datainfo.item()["test_num"]  # [20, 40, 60, 80, 100]
        self.storage = storage
        self.inc_idx = inc_idx
        self.newclassnum = self.train_num_lst[inc_idx]
        self.oldclassnum = self.test_num_lst[inc_idx] - self.train_num_lst[inc_idx]
        self.totalclassnum = self.test_num_lst[inc_idx]

        print("self.storage: ", format(self.storage))
        print("self.oldclassnum: ", format(self.oldclassnum))
        print("self.newclassnum: ", format(self.newclassnum))
        print("self.totalclassnum: ", format(self.totalclassnum))

    def combine(self, inc_idx):
        '''
        combine required classes into one file. The file is stored under /data/...
        Args: 
            inc_idx (int): 1, 2, 3, 4
        '''
        clasdiv_obj, clsdiv_data_lst = set_empty()
        train_num = exampler_train_lst[inc_idx]
        val_num = exampler_val_lst[inc_idx]
        print("exampler_train_num: {}, exampler_val_num: {}".format(exampler_train_lst[inc_idx],
                                                                    exampler_val_lst[inc_idx]))

        # exampler data of old classes (for balance validation)
        for i in range(self.totalclassnum):
            trainobj = unpickle(cifar_100_train_perclass + str(i))
            trdata_lst = trainobj['data'].tolist()
            clasdiv_obj['fine_labels'] = clasdiv_obj['fine_labels'] + trainobj['fine_labels'][0:val_num]
            clasdiv_obj['coarse_labels'] = clasdiv_obj['coarse_labels'] + trainobj['coarse_labels'][0:val_num]
            clasdiv_obj['filenames'] = clasdiv_obj['filenames'] + trainobj['filenames'][0:val_num]
            clsdiv_data_lst = clsdiv_data_lst + trdata_lst[0:val_num]
        clasdiv_obj['data'] = np.array(clsdiv_data_lst)
        with open(modeify_DATASET_ROOT + 'val_sample_' + str(self.inc_idx - 1), 'wb') as p:
            pickle.dump(clasdiv_obj, p)

        # exampler data of old classes (for training)
        clasdiv_obj, clsdiv_data_lst = set_empty()
        for i in range(self.oldclassnum):
            trainobj = unpickle(cifar_100_train_perclass + str(i))
            trdata_lst = trainobj['data'].tolist()
            clasdiv_obj['fine_labels'] = clasdiv_obj['fine_labels'] + trainobj['fine_labels'][
                                                                      val_num: train_num + val_num]
            clasdiv_obj['coarse_labels'] = clasdiv_obj['coarse_labels'] + trainobj['coarse_labels'][
                                                                          val_num: train_num + val_num]
            clasdiv_obj['filenames'] = clasdiv_obj['filenames'] + trainobj['filenames'][val_num: train_num + val_num]
            clsdiv_data_lst = clsdiv_data_lst + trdata_lst[val_num: train_num + val_num]
        clasdiv_obj['data'] = np.array(clsdiv_data_lst)
        with open(modeify_DATASET_ROOT + 'train_sample_' + str(self.inc_idx - 1), 'wb') as p:
            pickle.dump(clasdiv_obj, p)
        return True

    def combine2(self, inc_idx):
        train_num = exampler_train_lst[inc_idx + 1]
        val_num = exampler_val_lst[inc_idx + 1]
        # train_x
        clasdiv_obj, clsdiv_data_lst = set_empty()
        for i in range(self.totalclassnum - self.train_num_lst[inc_idx], self.totalclassnum):  # 全是新类
            trainobj = unpickle(cifar_100_train_perclass + str(i))
            trdata_lst = trainobj['data'].tolist()
            clasdiv_obj['fine_labels'] = clasdiv_obj['fine_labels'] + trainobj['fine_labels'][val_num:]
            clasdiv_obj['coarse_labels'] = clasdiv_obj['coarse_labels'] + trainobj['coarse_labels'][val_num:]
            clasdiv_obj['filenames'] = clasdiv_obj['filenames'] + trainobj['filenames'][val_num:]
            clsdiv_data_lst = clsdiv_data_lst + trdata_lst[val_num:]
        clasdiv_obj['data'] = np.array(clsdiv_data_lst)
        with open(modeify_DATASET_ROOT + 'train_' + str(self.inc_idx), 'wb') as p:
            pickle.dump(clasdiv_obj, p)

        return True


def main(k, n, m):
    init(k, n, m)
    group = (100 - n) // m + 1
    getdividedCifar(n, m, modeify_DATASET_ROOT)
    divide_cls(n, m)

    for inc_idx in range(group):
        g = GenerateExemplars(k, inc_idx)
        if inc_idx != 0:
            g.combine(inc_idx)  # generate train_sample_ and val_sample_
        g.combine2(inc_idx)  # generate train_ and test_
    print("Finished!")


if __name__ == '__main__':
    print("dealcifar100.py", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, required=False, default=2000,
                        help='memory size')
    parser.add_argument('-n', type=int, required=True, default=20,
                        help='number of the init classes')
    parser.add_argument('-m', type=int, required=True, default=20,
                        help='number of the classes incrementally transferred to model')
    args = parser.parse_args()

    main(args.k, args.n, args.m)
