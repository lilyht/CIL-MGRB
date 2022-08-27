import os
import cv2
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(17)  # torchvision.transforms

workers = 2
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Imagenet100_load_batch_newmodel(Dataset):
    def __init__(self, args, loadtype, modeltype, inc_idx, transform):
        self.args = args
        self.loadtype = loadtype
        self.inc_idx = inc_idx
        self.transform = transform
        self.imagenet100_path = '/HOME/scz1839/run/data/data/seed_1993_subset_100_imagenet/data'
        datainfopath = './save/imagenet100_divide_info_{}_{}.npy'.format(self.args.n, self.args.m)
        datainfo = np.load(datainfopath, allow_pickle=True)
        self.order = datainfo.item()["order"]
        self.label_map = datainfo.item()["label_map"]
        self.test_num_lst = datainfo.item()["test_num"]  # [20, 40, 60, 80, 100]
        self.id2clsname = datainfo.item()["id2clsname"]
        self.clsname2id = datainfo.item()["clsname2id"]

        # get exemplar_num, train_num, val_num
        numinfopath = './save/imagenet100_num_info_{}_{}.npy'.format(self.args.n, self.args.m)
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
            if inc_idx == 0:
                # train
                i = 0
                for label in train_labels_t:
                    txt_path = self.imagenet100_path + '/train/' + label + '/' + label + '_boxes.txt'
                    image_name = []
                    with open(txt_path) as txt:
                        for line in txt:
                            image_name.append(line.strip('\n').split('\t')[0])

                    # for training set, for old classes, [exemplar_valnum:]
                    train_image_names.append(image_name[exemplar_valnum:])
                    am = len(image_name[exemplar_valnum:])
                    for idx in range(am):
                        self.train_labels.append(self.label_map[self.clsname2id[label]])
                    i = i + 1

            else:
                i = 0
                for label in train_labels_t:
                    txt_path = self.imagenet100_path + '/train/' + label + '/' + label + '_boxes.txt'
                    image_name = []
                    with open(txt_path) as txt:
                        for line in txt:
                            image_name.append(line.strip('\n').split('\t')[0])

                    train_image_names.append(image_name[exemplar_valnum:exemplar_num])
                    for idx in range(exemplar_valnum, exemplar_num):
                        self.train_labels.append(self.label_map[self.clsname2id[label]])
                    i = i + 1
            labels = np.arange(100)

            # test
            test_labels_t = []
            self.test_labels = []
            test_names = []

            od = []
            od = self.order[0:self.test_num_lst[self.inc_idx]]
            for item in od:
                test_labels_t.append(self.id2clsname[item])

            for label in test_labels_t:
                txt_path = self.imagenet100_path + '/val/' + label + '/' + label + '_boxes.txt'
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip('\n').split('\t')[0])

                test_names.append(image_name[0:50])
                for idx in range(50):
                    self.test_labels.append(self.label_map[self.clsname2id[label]])
                i = i + 1

        elif modeltype == 'new':
            # train and val
            od = []
            od = self.order[0:self.test_num_lst[self.inc_idx + 1]]
            for item in od:
                train_labels_t.append(self.id2clsname[item])

            i = 0
            for label in train_labels_t:
                txt_path = self.imagenet100_path + '/train/' + label + '/' + label + '_boxes.txt'
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip('\n').split('\t')[0])

                # for training set: for new classes, [exemplar_valnum:]; else, [exemplar_valnum:exemplar_num]
                # for validation set: [0:exemplar_valnum]
                if i >= self.test_num_lst[self.inc_idx]:
                    train_image_names.append(image_name[exemplar_valnum:])  # 0~val是n类加载进来；之后全部加载进来，代表的是新类全部加载进来
                    am = len(image_name[exemplar_valnum:])
                    for idx in range(am):
                        self.train_labels.append(self.label_map[self.clsname2id[label]])
                else:
                    train_image_names.append(image_name[exemplar_valnum:exemplar_num])
                    for idx in range(exemplar_valnum, exemplar_num):
                        self.train_labels.append(self.label_map[self.clsname2id[label]])

                v_image_names.append(image_name[0:exemplar_valnum])
                for idx in range(exemplar_valnum):
                    self.v_labels.append(self.label_map[self.clsname2id[label]])

                i = i + 1
            labels = np.arange(100)

            # test
            test_labels_t = []
            self.test_labels = []
            test_names = []

            od = []
            od = self.order[0:self.test_num_lst[self.inc_idx + 1]]
            for item in od:
                test_labels_t.append(self.id2clsname[item])

            for label in test_labels_t:
                txt_path = self.imagenet100_path + '/val/' + label + '/' + label + '_boxes.txt'
                image_name = []
                with open(txt_path) as txt:
                    for line in txt:
                        image_name.append(line.strip('\n').split('\t')[0])

                test_names.append(image_name[0:50])
                for idx in range(50):
                    self.test_labels.append(self.label_map[self.clsname2id[label]])
                i = i + 1

        if modeltype == 'old':
            if loadtype == 'train':
                print("load train")
                # old classes
                i = 0
                self.images = []
                self.images = np.array(self.images)
                for label in train_labels_t[0:self.test_num_lst[self.inc_idx]]:
                    image = []
                    for image_name in train_image_names[i]:
                        image_path = os.path.join(self.imagenet100_path + '/train',
                                                  label, image_name)
                        img = cv2.imread(image_path)
                        img_256x256 = cv2.resize(img, (256, 256))
                        image.append(img_256x256)
                    self.images2 = np.array(image)
                    self.images2 = self.images2.reshape(-1, 256, 256, 3)
                    if i == 0:
                        self.images = self.images2
                    else:
                        self.images = np.concatenate((self.images, self.images2), axis=0)
                    i = i + 1
                print("self.images.shape: {}".format(self.images.shape))

            elif loadtype == 'test':
                i = 0
                print("load test")
                self.test_images = []
                for label in test_labels_t[0:self.test_num_lst[self.inc_idx]]:
                    image = []

                    for image_name in test_names[i]:
                        image_path = os.path.join(self.imagenet100_path + '/val',
                                                  label, image_name)
                        img = cv2.imread(image_path)
                        img_256x256 = cv2.resize(img, (256, 256))
                        image.append(img_256x256)
                    self.test_images.append(image)
                    i = i + 1
                self.test_images = np.array(self.test_images)
                print(self.test_images.shape)
                self.test_images = self.test_images.reshape(-1, 256, 256, 3)

        elif modeltype == 'new':
            if loadtype == 'train':
                print("load train")
                # old classes
                i = 0
                self.images = []
                self.images = np.array(self.images)

                for label in train_labels_t[0:self.test_num_lst[self.inc_idx + 1]]:
                    image = []
                    for image_name in train_image_names[i]:
                        image_path = os.path.join(self.imagenet100_path + '/train',
                                                  label, image_name)
                        img = cv2.imread(image_path)
                        img_256x256 = cv2.resize(img, (256, 256))
                        image.append(img_256x256)

                    self.images2 = np.array(image)
                    self.images2 = self.images2.reshape(-1, 256, 256, 3)
                    if i == 0:
                        self.images = self.images2
                    else:
                        self.images = np.concatenate((self.images, self.images2), axis=0)

                    i = i + 1
                print("self.images.shape: {}".format(self.images.shape))

            elif loadtype == 'val':
                print("load val")
                i = 0
                self.images = []

                for label in train_labels_t[0:self.test_num_lst[self.inc_idx + 1]]:
                    image = []
                    for image_name in v_image_names[i]:
                        image_path = os.path.join(self.imagenet100_path + '/train',
                                                  label, image_name)
                        img = cv2.imread(image_path)
                        img_256x256 = cv2.resize(img, (256, 256))
                        image.append(img_256x256)  # 256*256*3
                    self.images.append(image)
                    i = i + 1
                self.images = np.array(self.images)
                self.images = self.images.reshape(-1, 256, 256, 3)

            elif loadtype == 'test':
                print("load test")
                self.test_images = []
                i = 0
                for label in test_labels_t[0:self.test_num_lst[self.inc_idx + 1]]:
                    image = []
                    for image_name in test_names[i]:
                        image_path = os.path.join(self.imagenet100_path + '/val',
                                                  label, image_name)
                        img = cv2.imread(image_path)
                        img_256x256 = cv2.resize(img, (256, 256))
                        image.append(img_256x256)
                    self.test_images.append(image)
                    i = i + 1
                self.test_images = np.array(self.test_images)
                print(self.test_images.shape)
                self.test_images = self.test_images.reshape(-1, 256, 256, 3)

    def __getitem__(self, index):
        label = []
        image = []
        if self.loadtype == 'train':
            label = self.train_labels[index]
            image = self.images[index]  # <class 'numpy.ndarray'>
        if self.loadtype == 'val':
            label = self.v_labels[index]
            image = self.images[index]
        if self.loadtype == 'test':
            label = self.test_labels[index]
            image = self.test_images[index]

        plt.imshow(image, cmap='gray')
        plt.show()

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


def get_imagenet100_dataset(args, modeltype, inc_idx):
    """Get all the data in the imagenet-100 dataset
    return trainloader, valloader and list of classes in the dataset(
    for instances [0, 1, 2, ..., 100])
    because there are no groundtruth for test data, I will use the
    validation part as the test data
    """
    print("getting imagenet100 dataset...")

    if modeltype == 'old':
        train_dataset = Imagenet100_load_batch_newmodel(args, 'train', 'old', inc_idx,
                                                        transform=transforms.Compose([
                                                            transforms.ToPILImage(),
                                                            transforms.RandomResizedCrop(224),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomRotation(15),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean, std)]))

        #         x, y = train_dataset.__getitem__(12748)
        #         print(y)

        test_dataset = Imagenet100_load_batch_newmodel(args, 'test', 'old', inc_idx,
                                                       transform=transforms.Compose([
                                                           transforms.ToPILImage(),
                                                           transforms.Resize(256),
                                                           transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean, std)]))

        #         x, y = test_dataset.__getitem__(499)
        #         print(y)

        trainloader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=workers)
        testloader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=workers)
        print("Successfully load imagenet-100")
        return trainloader, testloader

    elif modeltype == 'new':
        train_dataset = Imagenet100_load_batch_newmodel(args, 'train', 'new', inc_idx,
                                                        transform=transforms.Compose([
                                                            transforms.ToPILImage(),
                                                            transforms.RandomResizedCrop(224),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomRotation(15),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean, std)]))

        val_dataset = Imagenet100_load_batch_newmodel(args, 'val', 'new', inc_idx,
                                                      transform=transforms.Compose([
                                                          transforms.ToPILImage(),
                                                          transforms.Resize([224, 224]),
                                                          transforms.RandomResizedCrop(224),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.RandomRotation(15),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(mean, std)]))

        test_dataset = Imagenet100_load_batch_newmodel(args, 'test', 'new', inc_idx,
                                                       transform=transforms.Compose([
                                                           transforms.ToPILImage(),
                                                           transforms.Resize(256),
                                                           transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean, std)]))

        trainloader = DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=workers)
        valloader = DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=True, num_workers=workers)
        testloader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=True, num_workers=workers)
        print("Successfully load imagenet100")
        return trainloader, valloader, testloader

# trainloader1, testloader1, clsnum1 = get_imagenet100_dataset(args, 'old', 3)
# trainloader2, valloader2, testloader2, clsnum2 = get_imagenet100_dataset(args, 'new', 1)
