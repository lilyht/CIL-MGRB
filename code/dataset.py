import cv2
import ast
import shutil
from tqdm import tqdm
import torch
import torchvision.transforms as transforms

import datasets_cifar
import modified_datasets_cifar


# base_path = "."
base_path = "/HOME/scz1839/run"
CIFAR10_PATH = base_path + "/data/cifar10"
CIFAR100_PATH = base_path + "/data/cifar-100"

mean = [0.5071, 0.4866, 0.4409]
std = [0.2673, 0.2564, 0.2762]

mini_imagenet_mean = [0.485, 0.456, 0.406]
mini_imagenet_std = [0.229, 0.224, 0.225]
mean_ilsvrc12 = [0.485, 0.456, 0.406]
std_ilsvrc12 = [0.229, 0.224, 0.225]
mean_inat19 = [0.454, 0.474, 0.367]
std_inat19 = [0.237, 0.230, 0.249]
workers = 2


def split_cifar100_dataset_load_batch(args, inc_idx, load_train=True):
    CIFAR100_PATH = "./data/cifar100"
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    if load_train:
        cifar100_training = modified_datasets_cifar.CIFAR100(root=CIFAR100_PATH, inc_idx=inc_idx, train=0,
                                                             download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(cifar100_training, batch_size=args.bs,
                                                  shuffle=True, num_workers=workers)
    else:
        trainloader = None

    cifar100_testing = modified_datasets_cifar.CIFAR100(root=CIFAR100_PATH, inc_idx=inc_idx, train=1,
                                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar100_testing, batch_size=100, shuffle=False, num_workers=workers)
    return trainloader, testloader


def split_cifar10_dataset_load_batch(args, inc_idx, load_train=True):
    CIFAR10_PATH = "./data/cifar10"
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if load_train:
        cifar10_training = modified_datasets_cifar.CIFAR10(root=CIFAR10_PATH, inc_idx=inc_idx, train=0,
                                                           download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(cifar10_training, batch_size=args.bs,
                                                  shuffle=True, num_workers=workers)
    else:
        trainloader = None

    cifar10_testing = modified_datasets_cifar.CIFAR10(root=CIFAR10_PATH, inc_idx=inc_idx, train=1,
                                                      download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar10_testing, batch_size=100, shuffle=False, num_workers=workers)
    return trainloader, testloader


def cifar100_valid_dataset(args, inc_idx):
    CIFAR100_PATH = './data/cifar100'
    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    cifar100_validation = modified_datasets_cifar.CIFAR100(root=CIFAR100_PATH, inc_idx=inc_idx, train=2,
                                                           download=True, transform=transform_valid)
    validloader = torch.utils.data.DataLoader(cifar100_validation, batch_size=args.bs,
                                              shuffle=True, num_workers=workers)
    return validloader


def cifar10_valid_dataset(args, inc_idx):
    CIFAR10_PATH = "./data/cifar10"
    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    cifar10_validation = modified_datasets_cifar.CIFAR10(root=CIFAR10_PATH, inc_idx=inc_idx, train=2,
                                                         download=True, transform=transform_valid)
    validloader = torch.utils.data.DataLoader(cifar10_validation, batch_size=args.bs,
                                              shuffle=True, num_workers=workers)
    return validloader


def cifar10_dataset_joint(args, inc_idx, load_train=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    if load_train:
        cifar10_training = datasets_cifar.CIFAR10(root=CIFAR10_PATH, inc_idx=inc_idx, train=0,
                                                  download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(cifar10_training, batch_size=args.bs,
                                                  shuffle=True, num_workers=workers)
    else:
        trainloader = None
    cifar10_testing = datasets_cifar.CIFAR10(root=CIFAR10_PATH, inc_idx=inc_idx, train=1,
                                             download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar10_testing, batch_size=100,
                                             shuffle=False, num_workers=workers)
    return trainloader, testloader


def cifar100_dataset_joint(args, inc_idx, load_train=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    if load_train:
        cifar100_training = datasets_cifar.CIFAR100(root=CIFAR100_PATH, inc_idx=inc_idx, train=0,
                                                    download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(cifar100_training, batch_size=args.bs,
                                                  shuffle=True, num_workers=workers)
    else:
        trainloader = None
    cifar100_testing = datasets_cifar.CIFAR100(root=CIFAR100_PATH, inc_idx=inc_idx, train=1,
                                               download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar100_testing, batch_size=100,
                                             shuffle=False, num_workers=workers)
    return trainloader, testloader
