import os
import cv2
import math
import torch
import pickle
import numpy as np
import scipy.io as sciio
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(17)  # torchvision.transforms

idx_num = {0: 85, 1: 96, 2: 83, 3: 91, 4: 86, 5: 86, 6: 86, 7: 108, 8: 87, 9: 140, 10: 138, 11: 288, 12: 652, 13: 135,
           14: 145, 15: 137, 16: 137, 17: 174, 18: 160, 19: 172, 20: 176, 21: 164, 22: 153, 23: 134, 24: 145, 25: 143,
           26: 338, 27: 158, 28: 136, 29: 135, 30: 134, 31: 165, 32: 135, 33: 135, 34: 134, 35: 145, 36: 151, 37: 141,
           38: 102, 39: 93, 40: 208, 41: 145, 42: 140, 43: 174, 44: 162, 45: 217, 46: 138, 47: 126, 48: 135, 49: 134,
           50: 154, 51: 160, 52: 140, 53: 146, 54: 200, 55: 136, 56: 149, 57: 140, 58: 139, 59: 126, 60: 161, 61: 162,
           62: 164, 63: 165, 64: 138, 65: 2681}
train_num_dict = {0: 85, 1: 96, 2: 83, 3: 91, 4: 86, 5: 86, 6: 86, 7: 108, 8: 87, 9: 140, 10: 138, 11: 288, 12: 652,
                  13: 135, 14: 145, 15: 137, 16: 137, 17: 174, 18: 160, 19: 172, 20: 176, 21: 164, 22: 153, 23: 134,
                  24: 145, 25: 143, 26: 338, 27: 158, 28: 136, 29: 135, 30: 134, 31: 165, 32: 135, 33: 135, 34: 134,
                  35: 145, 36: 151, 37: 141, 38: 102, 39: 93, 40: 208, 41: 145, 42: 140, 43: 174, 44: 162, 45: 217,
                  46: 138, 47: 126, 48: 135, 49: 134, 50: 154, 51: 160, 52: 140, 53: 146, 54: 200, 55: 136, 56: 149,
                  57: 140, 58: 139, 59: 126, 60: 161, 61: 162, 62: 164, 63: 165, 64: 138, 65: 2681}
val_num_dict = {0: 12, 1: 13, 2: 11, 3: 12, 4: 11, 5: 12, 6: 11, 7: 15, 8: 11, 9: 19, 10: 19, 11: 41, 12: 92, 13: 18,
                14: 20, 15: 19, 16: 19, 17: 24, 18: 22, 19: 23, 20: 25, 21: 22, 22: 21, 23: 19, 24: 20, 25: 20, 26: 47,
                27: 22, 28: 19, 29: 18, 30: 19, 31: 22, 32: 19, 33: 18, 34: 18, 35: 20, 36: 21, 37: 19, 38: 14, 39: 12,
                40: 29, 41: 20, 42: 19, 43: 24, 44: 23, 45: 30, 46: 19, 47: 17, 48: 18, 49: 19, 50: 21, 51: 22, 52: 19,
                53: 20, 54: 28, 55: 18, 56: 21, 57: 19, 58: 19, 59: 17, 60: 22, 61: 23, 62: 23, 63: 23, 64: 19, 65: 382}
test_num_dict = {0: 24, 1: 26, 2: 23, 3: 26, 4: 23, 5: 24, 6: 24, 7: 30, 8: 24, 9: 39, 10: 39, 11: 82, 12: 185, 13: 38,
                 14: 41, 15: 38, 16: 38, 17: 49, 18: 45, 19: 49, 20: 49, 21: 46, 22: 43, 23: 37, 24: 41, 25: 40, 26: 96,
                 27: 44, 28: 38, 29: 38, 30: 38, 31: 46, 32: 38, 33: 38, 34: 38, 35: 41, 36: 42, 37: 40, 38: 28, 39: 26,
                 40: 59, 41: 40, 42: 39, 43: 49, 44: 46, 45: 61, 46: 39, 47: 35, 48: 38, 49: 37, 50: 43, 51: 46, 52: 39,
                 53: 41, 54: 56, 55: 38, 56: 42, 57: 39, 58: 39, 59: 35, 60: 46, 61: 45, 62: 46, 63: 47, 64: 38,
                 65: 765}

workers = 2


def FaultDataset2d_load_data(args, modeltype, inc_idx):
    """
    Args:
        inc_idx (int): the number of the incremental learning process.
    """
    FARON_PATH = '/HOME/scz1839/run/data/FARON/FARON_3d_121.mat'
    datainfopath = './save/FARON_divide_info_{}_{}.npy'.format(args.n, args.m)
    datainfo = np.load(datainfopath, allow_pickle=True)
    order = datainfo.item()["order"]
    test_num_lst = datainfo.item()["test_num"]

    numinfopath = "./save/FARON_num_info_{}_{}.npy".format(args.n, args.m)
    num_info = np.load(numinfopath, allow_pickle=True)

    exemplar_num = num_info.item()["exemplar_num"][inc_idx]
    exemplar_valnum = num_info.item()["val_num"][inc_idx]

    data = sciio.loadmat(FARON_PATH)
    train_set = data.get('x_train')
    train_label = data.get('y_train')
    test_set = data.get('x_test')
    test_label = data.get('y_test')
    train_label = train_label - 1
    test_label = test_label - 1

    if modeltype == 'old':
        select_train_x = []
        select_train_y = []
        select_test_x = []
        select_test_y = []

        # original train_set：(20, 121, 12643)
        train_set_T = torch.from_numpy(train_set).permute(2, 0, 1)  # (12643, 20, 121)
        train_label_T = train_label  # (12643, 1)
        test_set_T = torch.from_numpy(test_set).permute(2, 0, 1)  # (3562, 20, 121)
        test_label_T = test_label  # (3562, 1)

        od = []
        od = order[0:test_num_lst[inc_idx]]

        # train x and y
        i = 0
        for item in train_label_T:
            if item[0] in od:
                select_train_x.append(train_set_T[i][:][:])
                select_train_y.append(train_label_T[i][:])
            i = i + 1
        print("len(select_train_x)", len(select_train_x))

        select_train_x = torch.tensor([item.cpu().detach().numpy() for item in select_train_x])
        select_train_x = select_train_x.permute(1, 2, 0)
        select_train_x = select_train_x.numpy()
        select_train_y = np.array(select_train_y)

        # test x and y
        i = 0
        for item in test_label_T:
            if item[0] in od:
                select_test_x.append(test_set_T[i][:][:])
                select_test_y.append(test_label_T[i][:])
            i = i + 1
        print("len(select_test_x)", len(select_test_x))
        select_test_x = torch.tensor([item.cpu().detach().numpy() for item in select_test_x])
        select_test_x = select_test_x.permute(1, 2, 0)
        select_test_x = select_test_x.numpy()
        select_test_y = np.array(select_test_y)

        return select_train_x, select_train_y, select_test_x, select_test_y

    elif modeltype == 'new':
        select_train_x = []
        select_train_y = []
        select_test_x = []
        select_test_y = []
        select_val_x = []
        select_val_y = []
        count = {}  # how many samples in each class
        i = 0
        for itm in range(66):
            count[i] = idx_num[itm]
            i = i + 1

        train_set_T = torch.from_numpy(train_set).permute(2, 0, 1)  # (12643, 20, 121)
        train_label_T = train_label  # (12643, 1)
        test_set_T = torch.from_numpy(test_set).permute(2, 0, 1)  # (3562, 20, 121)
        test_label_T = test_label  # (3562, 1)

        od = []
        od = order[0:test_num_lst[inc_idx + 1]]

        # train and val for x and y
        k = 0
        while k < 12643:
            # print("k:", k, "train_label_T[k][0]:", train_label_T[k][0])
            if train_label_T[k][0] in od[0:test_num_lst[inc_idx]]:

                # print("exemplar_num:", exemplar_num, ", count[train_label_T[k][0]:", count[train_label_T[k][0]])
                if count[train_label_T[k][0]] < exemplar_num:
                    # there are few samples in this class and cannot reach the average
                    print("class {} not enough, load {} sample instead...".format(train_label_T[k][0], count[
                        train_label_T[k][0]] - exemplar_valnum))
                    end_idx = k + count[train_label_T[k][0]]
                else:
                    end_idx = k + exemplar_num

                if count[train_label_T[k][0]] < exemplar_valnum:
                    # there are few samples in this class and cannot reach the average
                    print("class {} not enough, load {} sample instead...".format(train_label_T[k][0], count[
                        train_label_T[k][0]] - exemplar_valnum))
                    start_idx = k + exemplar_valnum // 2
                else:
                    start_idx = k + exemplar_valnum

                _ = train_set_T[start_idx:end_idx][:][:]
                for __ in _:
                    select_train_x.append(__)
                _ = train_label_T[start_idx:end_idx][:]
                for __ in _:
                    select_train_y.append(__)

                _ = train_set_T[k:start_idx][:][:]
                for __ in _:
                    select_val_x.append(__)
                _ = train_label_T[k:start_idx][:]
                for __ in _:
                    select_val_y.append(__)

            elif train_label_T[k][0] in od[test_num_lst[inc_idx]:test_num_lst[inc_idx + 1]]:

                exemplar_half_valnum = exemplar_valnum
                if count[train_label_T[k][0]] < exemplar_valnum:
                    # there are few samples in this class and cannot reach the average
                    print("class {} not enough, load {} sample instead...".format(train_label_T[k][0], count[
                        train_label_T[k][0]] - exemplar_valnum))
                    exemplar_half_valnum = exemplar_valnum // 2

                _ = train_set_T[k + exemplar_half_valnum:k + count[train_label_T[k][0]]][:][:]
                for __ in _:
                    select_train_x.append(__)
                _ = train_label_T[k + exemplar_half_valnum:k + count[train_label_T[k][0]]][:]
                for __ in _:
                    select_train_y.append(__)

                _ = train_set_T[k:k + exemplar_half_valnum][:][:]
                for __ in _:
                    select_val_x.append(__)
                _ = train_label_T[k:k + exemplar_half_valnum][:]
                for __ in _:
                    select_val_y.append(__)
            k = k + count[train_label_T[k][0]]

        print("len(select_train_x)", len(select_train_x))

        select_train_x = torch.tensor([item.cpu().detach().numpy() for item in select_train_x])
        select_train_x = select_train_x.permute(1, 2, 0)
        select_train_x = select_train_x.numpy()
        select_train_y = np.array(select_train_y)
        print("select_train_y", select_train_y.shape)

        select_val_x = torch.tensor([item.cpu().detach().numpy() for item in select_val_x])
        select_val_x = select_val_x.permute(1, 2, 0)
        select_val_x = select_val_x.numpy()
        select_val_y = np.array(select_val_y)

        # test x and y
        i = 0
        for item in test_label_T:
            if item[0] in od:
                select_test_x.append(test_set_T[i][:][:])
                select_test_y.append(test_label_T[i][:])
            i = i + 1
        print("len(select_test_x)", len(select_test_x))
        select_test_x = torch.tensor([item.cpu().detach().numpy() for item in select_test_x])
        select_test_x = select_test_x.permute(1, 2, 0)
        select_test_x = select_test_x.numpy()
        select_test_y = np.array(select_test_y)

        return select_train_x, select_train_y, select_val_x, select_val_y, select_test_x, select_test_y


class FaultDataset2d(Dataset):

    def __init__(self, data_array, label_array, transform=None):
        """
        Args:
            data_array (string): array of data which have been cached.
            label_array (string): array of labels which have been cached.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_array = data_array
        self.labels = np.array(label_array)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data_array[:, :, idx]
        label = self.labels[idx]
        return data, label


def get_FARON_dataset(args, modeltype, inc_idx):
    print("getting FARON dataset...")

    if modeltype == 'old':
        train_set, train_label, test_set, test_label = FaultDataset2d_load_data(args, modeltype, inc_idx)
        train_dataset = FaultDataset2d(train_set, train_label, inc_idx)
        test_dataset = FaultDataset2d(test_set, test_label, inc_idx)
        trainloader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=workers)
        testloader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=workers)
        print("Successfully load FARON")
        return trainloader, testloader

    elif modeltype == 'new':
        train_set, train_label, val_set, val_label, test_set, test_label = FaultDataset2d_load_data(args, modeltype,
                                                                                                    inc_idx)
        train_dataset = FaultDataset2d(train_set, train_label, inc_idx)
        val_dataset = FaultDataset2d(val_set, val_label, inc_idx)
        test_dataset = FaultDataset2d(test_set, test_label, inc_idx)
        trainloader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=workers)
        valloader = DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=True, num_workers=workers)
        testloader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=workers)
        print("Successfully load FARON")
        return trainloader, valloader, testloader
