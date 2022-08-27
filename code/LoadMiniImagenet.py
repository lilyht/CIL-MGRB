import os
import cv2
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(17)  # torchvision.transforms


class MiniImagenet_load_batch_newmodel(Dataset):
    """
    load dataset
    """

    def __init__(self, args, loadtype, modeltype, inc_idx, transform):
        self.args = args
        self.loadtype = loadtype
        self.inc_idx = inc_idx
        self.transform = transform
        self.mini_imagenet_mean = [0.485, 0.456, 0.406]
        self.mini_imagenet_std = [0.229, 0.224, 0.225]
        self.workers = 2

        self.mini_imagenet_path = '/HOME/scz1839/run/data/mini-imagenet/processed'
        datainfopath = './save/miniimagenet_divide_info_{}_{}.npy'.format(self.args.n, self.args.m)
        datainfo = np.load(datainfopath, allow_pickle=True)
        self.order = datainfo.item()["order"]
        self.label_map = datainfo.item()["label_map"]
        self.test_num_lst = datainfo.item()["test_num"]  # [20, 40, 60, 80, 100]
        self.id2clsname = datainfo.item()["id2clsname"]
        self.clsname2id = datainfo.item()["clsname2id"]

        # get exemplar_num, train_num, val_num
        numinfopath = './save/miniimagenet_num_info_{}_{}.npy'.format(self.args.n, self.args.m)
        num_info = np.load(numinfopath, allow_pickle=True)
        exemplar_num = num_info.item()["exemplar_num"][inc_idx]
        exemplar_valnum = num_info.item()["val_num"][inc_idx]

        train_labels_t = []
        train_image_names = []
        self.train_labels = []

        v_image_names = []
        self.v_labels = []

        if modeltype == 'old':
            od = []
            od = self.order[0:self.test_num_lst[self.inc_idx]]
            for item in od:
                train_labels_t.append(self.id2clsname[item])
            i = 0
            for label in train_labels_t:
                # The fields in xxboxes.txt are picture name, X, Y, H and W
                txt_path = self.mini_imagenet_path + '/train/' + label + '/' + label + '_boxes.txt'
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip('\n').split('\t')[0])

                train_image_names.append(image_name[exemplar_valnum:])
                for idx in range(500 - exemplar_valnum):
                    self.train_labels.append(self.label_map[self.clsname2id[label]])
                i = i + 1

            labels = np.arange(100)

            # test loader
            test_labels_t = []
            self.test_labels = []
            test_names = []

            od = []
            od = self.order[0:self.test_num_lst[self.inc_idx]]
            for item in od:
                test_labels_t.append(self.id2clsname[item])

            for label in test_labels_t:
                txt_path = self.mini_imagenet_path + '/test/' + label + '/' + label + '_boxes.txt'
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip('\n').split('\t')[0])

                test_names.append(image_name[0:100])
                for idx in range(100):
                    self.test_labels.append(self.label_map[self.clsname2id[label]])
                i = i + 1

        elif modeltype == 'new':
            # train & val
            od = []
            od = self.order[0:self.test_num_lst[self.inc_idx + 1]]
            for item in od:
                train_labels_t.append(self.id2clsname[item])

            i = 0
            for label in train_labels_t:
                txt_path = self.mini_imagenet_path + '/train/' + label + '/' + label + '_boxes.txt'
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip('\n').split('\t')[0])

                # for training set: for new classes, [exemplar_valnum:]; else, [exemplar_valnum:exemplar_num]
                # for validation set: [0:exemplar_valnum]
                if i >= self.test_num_lst[self.inc_idx]:
                    train_image_names.append(image_name[exemplar_valnum:])
                    for idx in range(500 - exemplar_valnum):
                        self.train_labels.append(self.label_map[self.clsname2id[label]])
                else:
                    train_image_names.append(image_name[exemplar_valnum:exemplar_num])
                    for idx in range(exemplar_valnum, exemplar_num):
                        self.train_labels.append(self.label_map[self.clsname2id[label]])

                v_image_names.append(image_name[0:exemplar_valnum])
                for idx in range(exemplar_valnum):
                    self.v_labels.append(self.label_map[self.clsname2id[label]])

                i = i + 1

            # test loader
            test_labels_t = []
            self.test_labels = []
            test_names = []

            od = []
            od = self.order[0:self.test_num_lst[self.inc_idx + 1]]
            for item in od:
                test_labels_t.append(self.id2clsname[item])

            for label in test_labels_t:
                txt_path = self.mini_imagenet_path + '/test/' + label + '/' + label + '_boxes.txt'
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip('\n').split('\t')[0])

                test_names.append(image_name[0:100])
                for idx in range(100):
                    self.test_labels.append(self.label_map[self.clsname2id[label]])
                i = i + 1

        if modeltype == 'old':
            # the number of samples of the old and new classes is different, so reshape together will occur an error.
            if loadtype == 'train':
                print("load train")
                # old classes
                i = 0
                self.images = []
                for label in train_labels_t[0:self.test_num_lst[self.inc_idx]]:
                    image = []
                    for image_name in train_image_names[i]:
                        image_path = os.path.join(self.mini_imagenet_path + '/train',
                                                  label, image_name)
                        image.append(cv2.imread(image_path))
                    self.images.append(image)
                    i = i + 1
                self.images = np.array(self.images)
                print("self.images.shape: {}".format(self.images.shape))
                self.images = self.images.reshape(-1, 84, 84, 3)

            elif loadtype == 'test':
                i = 0
                print("load test")
                self.test_images = []
                for label in test_labels_t[0:self.test_num_lst[self.inc_idx]]:
                    image = []

                    for image_name in test_names[i]:
                        image_path = os.path.join(self.mini_imagenet_path + '/test',
                                                  label, image_name)
                        image.append(cv2.imread(image_path))
                    self.test_images.append(image)
                    i = i + 1
                self.test_images = np.array(self.test_images)
                print(self.test_images.shape)
                self.test_images = self.test_images.reshape(-1, 84, 84, 3)

        elif modeltype == 'new':
            if loadtype == 'train':
                print("load train")
                # old classes
                i = 0
                self.images = []
                for label in train_labels_t[0:self.test_num_lst[self.inc_idx]]:
                    image = []
                    for image_name in train_image_names[i]:
                        image_path = os.path.join(self.mini_imagenet_path + '/train',
                                                  label, image_name)
                        image.append(cv2.imread(image_path))
                    self.images.append(image)
                    i = i + 1
                self.images = np.array(self.images)
                print(self.images.shape)
                self.images = self.images.reshape(-1, 84, 84, 3)

                # new classes
                i = self.test_num_lst[self.inc_idx]
                self.images2 = []
                for label in train_labels_t[self.test_num_lst[self.inc_idx]:self.test_num_lst[self.inc_idx + 1]]:
                    image = []
                    for image_name in train_image_names[i]:
                        image_path = os.path.join(self.mini_imagenet_path + '/train',
                                                  label, image_name)
                        image.append(cv2.imread(image_path))
                    self.images2.append(image)
                    i = i + 1
                self.images2 = np.array(self.images2)
                self.images2 = self.images2.reshape(-1, 84, 84, 3)
                self.images = np.concatenate((self.images, self.images2), axis=0)

            elif loadtype == 'val':
                print("load val")
                i = 0
                self.images = []
                for label in train_labels_t[0:self.test_num_lst[self.inc_idx + 1]]:
                    image = []
                    for image_name in v_image_names[i]:
                        image_path = os.path.join(self.mini_imagenet_path + '/train',
                                                  label, image_name)
                        image.append(cv2.imread(image_path))
                    self.images.append(image)
                    i = i + 1
                self.images = np.array(self.images)
                self.images = self.images.reshape(-1, 84, 84, 3)

            elif loadtype == 'test':
                print("load test")
                self.test_images = []
                i = 0
                for label in test_labels_t[0:self.test_num_lst[self.inc_idx + 1]]:
                    image = []
                    for image_name in test_names[i]:
                        image_path = os.path.join(self.mini_imagenet_path + '/test',
                                                  label, image_name)
                        image.append(cv2.imread(image_path))
                    self.test_images.append(image)
                    i = i + 1
                self.test_images = np.array(self.test_images)
                print(self.test_images.shape)
                self.test_images = self.test_images.reshape(-1, 84, 84, 3)

    def __getitem__(self, index):
        label = []
        image = []
        if self.loadtype == 'train':
            label = self.train_labels[index]
            image = self.images[index]  # <class 'numpy.ndarray'>  (64, 64, 3)
        if self.loadtype == 'val':
            label = self.v_labels[index]
            image = self.images[index]  # <class 'numpy.ndarray'>
        if self.loadtype == 'test':
            label = self.test_labels[index]
            image = self.test_images[index]

        # plt.imshow(image,cmap = 'gray')
        # plt.show()

        return self.transform(image), label

    def __len__(self):
        len = 0
        if self.loadtype == 'train':
            len = self.images.shape[0]
        if self.loadtype == 'val':
            len = self.images.shape[0]
        if self.loadtype == 'test':
            len = self.test_images.shape[0]
        return len


def get_mini_imagenet_dataset(args, modeltype, inc_idx):
    """Get all the data in the mini imagenet dataset
    return trainloader, valloader and list of classes in the dataset(
    for instances [0, 1, 2, ..., 100])
    """
    print("getting mini imagenet dataset...")

    mini_imagenet_mean = [0.485, 0.456, 0.406]
    mini_imagenet_std = [0.229, 0.224, 0.225]
    workers = 2

    if modeltype == 'old':
        train_dataset = MiniImagenet_load_batch_newmodel(args, 'train', 'old', inc_idx,
                                                         transform=transforms.Compose([
                                                             transforms.ToPILImage(),
                                                             transforms.Resize([84, 84]),
                                                             # transforms.RandomCrop(32, padding=4),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.RandomRotation(15),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mini_imagenet_mean,
                                                                                  mini_imagenet_std)]))
        # x, y = train_dataset.__getitem__(1769)
        # print(y)

        test_dataset = MiniImagenet_load_batch_newmodel(args, 'test', 'old', inc_idx,
                                                        transform=transforms.Compose([
                                                            transforms.ToPILImage(),
                                                            # transforms.Resize([32, 32]),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mini_imagenet_mean,
                                                                                 mini_imagenet_std)]))

        trainloader = DataLoader(dataset=train_dataset, batch_size=args.bs,
                                 shuffle=True, num_workers=workers)
        testloader = DataLoader(dataset=test_dataset, batch_size=args.bs,
                                shuffle=True, num_workers=workers)
        print("Successfully load mini imagenet")
        return trainloader, testloader

    elif modeltype == 'new':
        train_dataset = MiniImagenet_load_batch_newmodel(args, 'train', 'new', inc_idx,
                                                         transform=transforms.Compose([
                                                             transforms.ToPILImage(),
                                                             # transforms.Resize([32, 32]),
                                                             # transforms.RandomCrop(32, padding=4),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.RandomRotation(15),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mini_imagenet_mean,
                                                                                  mini_imagenet_std)]))

        val_dataset = MiniImagenet_load_batch_newmodel(args, 'val', 'new', inc_idx,
                                                       transform=transforms.Compose([
                                                           transforms.ToPILImage(),
                                                           # transforms.Resize([32, 32]),
                                                           # transforms.RandomCrop(32, padding=4),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.RandomRotation(15),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mini_imagenet_mean,
                                                                                mini_imagenet_std)]))

        test_dataset = MiniImagenet_load_batch_newmodel(args, 'test', 'new', inc_idx,
                                                        transform=transforms.Compose([
                                                            transforms.ToPILImage(),
                                                            # transforms.Resize([32, 32]),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mini_imagenet_mean,
                                                                                 mini_imagenet_std)]))
        #         x, y = test_dataset.__getitem__(4000)
        #         print(y)
        #         print(test_dataset.__len__())

        trainloader = DataLoader(dataset=train_dataset, batch_size=args.bs,
                                 shuffle=True, num_workers=workers)
        valloader = DataLoader(dataset=val_dataset, batch_size=args.bs,
                               shuffle=True, num_workers=workers)
        testloader = DataLoader(dataset=test_dataset, batch_size=args.bs,
                                shuffle=True, num_workers=workers)
        print("Successfully load mini imagenet")
        return trainloader, valloader, testloader

# trainloader1, testloader1, clsnum1 = get_mini_imagenet_dataset(args, 'old', 3)
# trainloader2, valloader2, testloader2, clsnum2 = get_mini_imagenet_dataset(args, 'new', 1)
