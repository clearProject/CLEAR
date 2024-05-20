import os
import json
import numpy as np
import torch
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
from dataset import ScoreDataset, ImagenetA
import sys
import pdb
import pickle
import argparse
from tqdm import tqdm
from itertools import chain
import torch.nn as nn
import transformers
from collections import defaultdict
import sklearn
from sentence_transformers import SentenceTransformer
import clip


def clean_label(true_labels):
    true_labels = np.array(true_labels)
    if np.min(true_labels) > 0:
        true_labels -= np.min(true_labels)
    return true_labels


def get_labels(dataset):
    if dataset == 'cub':
        with open("./data/CUB_200_2011/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)
        train_test_split = pd.read_csv(os.path.join('./data/', 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        train_test_split = train_test_split['is_training_img'].values
        train_indices = np.where(train_test_split == 1)
        test_indices = np.where(train_test_split == 0)
        train_labels, test_labels = true_labels[train_indices], true_labels[test_indices]

    elif dataset == 'cifar100':
        with open("./data/cifar-100-python/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)
        train_labels, test_labels = true_labels[:-10000], true_labels[-10000:]

    elif dataset == 'cifar10' or dataset == 'cifar10-p':
        with open("./data/cifar-10-batches-py/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)
        train_labels, test_labels = true_labels[:-10000], true_labels[-10000:]

    elif dataset == 'food':
        with open("./data/food-101/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)
        train_labels, test_labels = true_labels[:-25250], true_labels[-25250:]

    elif dataset == 'flower':
        with open("./data/flowers-102/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)

        train_labels, test_labels = true_labels[:-1020], true_labels[-1020:]

    elif dataset == 'imagenet':
        with open("./data/imagenet/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 50000 + 1281167

        train_labels, test_labels = true_labels[:1281167], true_labels[-50000:]

    elif dataset == 'imagenet-a':
        with open("./data/imagenet/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 50000 + 1281167

        train_labels, test_labels = true_labels[:1281167], true_labels[-50000:]

        train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        def filter_labels(labels):
            idxes = np.where((labels < 398) & (labels!=69))
            return labels[idxes]

        train_labels = filter_labels(train_labels)

        train_labels[np.where(train_labels>69)] -= 1

        testset = ImagenetA(root='./data/imagenet-a')
        test_labels = testset.labels

    elif dataset == 'imagenet-animal':
        with open("./data/imagenet/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 50000 + 1281167

        train_labels, test_labels = true_labels[:1281167], true_labels[-50000:]

        train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        def filter_labels(labels):
            idxes = np.where((labels < 398) & (labels!=69))
            return labels[idxes]

        train_labels = filter_labels(train_labels)
        test_labels = filter_labels(test_labels)

        train_labels[np.where(train_labels>69)] -= 1
        test_labels[np.where(test_labels>69)] -= 1

    else:
        raise NotImplementedError

    return train_labels, test_labels


def get_image_dataloader(dataset, preprocess, preprocess_eval=None, shuffle=False):

    if dataset == 'cub':
        # Load dataset
        from dataset import Cub2011
        train_dataset = Cub2011(root='./data/', mode='train', transform=preprocess)
        test_dataset = Cub2011(root='./data/', mode='test', transform=preprocess)

        print("Train dataset:", len(train_dataset))
        print("Test dataset:", len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=96, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)

    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=preprocess)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=preprocess)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

    elif dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    elif dataset == 'food':
        trainset = torchvision.datasets.Food101(root='./data/', split='train', download=True, transform=preprocess)
        testset = torchvision.datasets.Food101(root='./data/', split='test', download=True, transform=preprocess)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

    elif dataset == 'flower':
        trainset = torchvision.datasets.Flowers102(root='./data/', split='train', download=True, transform=preprocess)
        testset = torchvision.datasets.Flowers102(root='./data/', split='val', download=True, transform=preprocess)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

    elif dataset == 'imagenet' or dataset == 'imagenet-animal' or dataset == 'imagenet-a':
        trainset = torchvision.datasets.ImageNet(root='./data/imagenet', split='train', transform=preprocess)
        testset = torchvision.datasets.ImageNet(root='./data/imagenet', split='val', transform=preprocess)

        if dataset == 'imagenet-animal' or dataset == 'imagenet-a':

            def filter_dataset(dataset):
                targets = np.array(dataset.targets)
                idxes = np.where((targets < 398) & (targets != 69))
                dataset.targets = targets[idxes].tolist()
                dataset.samples = [dataset.samples[i] for i in idxes[0]]

            filter_dataset(trainset)
            filter_dataset(testset)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False)

        if dataset == 'imagenet-a':
            testset = ImagenetA(root='./data/imagenet-a', preprocess=preprocess)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    else:
        raise NotImplementedError

    return train_loader, test_loader


def get_output_dim(dataset):
    return len(np.unique(get_labels(dataset)[0]))


def get_folder_name(dataset):
    if dataset == 'cub':
        # return 'cub'
        return 'CUB_200_2011'
    elif dataset == 'cifar100':
        return 'cifar-100-python'
    elif dataset == 'cifar10':
        return 'cifar-10-batches-py'
    elif dataset == 'flower':
        return "flowers-102"
    elif dataset == 'food':
        return "food-101"
    elif dataset == 'imagenet' or dataset == 'imagenet-animal':
        return "imagenet"  # "Imagenet"
    elif dataset == 'imagenet-a':
        return "imagenet-a"
    else:
        raise NotImplementedError


def get_attributes(cfg, args):

    if cfg['attributes'] == 'cub':
        class2concepts = json.load(open("./data/CUB_200_2011/" + "concepts.json", "r"))
        return [concept for concepts in class2concepts.values() for concept in concepts]

    elif cfg['attributes'] == 'flower':
        class2concepts = json.load(open("./data/flowers-102/" + "concepts.json", "r"))
        return [concept for concepts in class2concepts.values() for concept in concepts]

    elif cfg['attributes'] == 'food':
        class2concepts = json.load(open("./data/food-101/" + "concepts.json", "r"))
        return [concept for concepts in class2concepts.values() for concept in concepts]

    elif cfg['attributes'] == 'imagenet':
        class2concepts = json.load(open("./data/imagenet/" + "concepts.json", "r"))
        return [concept for concepts in class2concepts.values() for concept in concepts]

    elif cfg['attributes'] == 'imagenet-animal':
        class2concepts = json.load(open("./data/imagenet/" + "concepts.json", "r"))
        imagenet_labels_cls2num = json.load(open("./data/imagenet/" + "imagenet-labels_text-to-num.json", "r"))
        imagenet_labels_cls2num = {key.replace('_', ' ').lower(): value for key, value in
                                   imagenet_labels_cls2num.items()}  # update to the same format as the concepts
        filtered_class2concepts = {
            cls: concepts for cls, concepts in class2concepts.items()
            if cls in imagenet_labels_cls2num and imagenet_labels_cls2num[cls] < 398 and imagenet_labels_cls2num[
                cls] != 69
        }
        return [concept for concepts in filtered_class2concepts.values() for concept in concepts]

    elif cfg['attributes'] == 'cifar10':
        class2concepts = json.load(open("./data/cifar-10-batches-py/" + "concepts.json", "r"))
        return [concept for concepts in class2concepts.values() for concept in concepts]

    elif cfg['attributes'] == 'cifar100':
        class2concepts = json.load(open("./data/cifar-100-python/" + "concepts.json", "r"))
        return [concept for concepts in class2concepts.values() for concept in concepts]

    else:
        raise NotImplementedError


def get_prefix(cfg):
    if cfg['attributes'] == 'cbm':
        return ""
    if cfg['dataset'] == 'cub':
        return "The bird has "
    elif cfg['dataset'] == 'cifar100':
        return "A photo of an object with "
    elif cfg['dataset'] == 'cifar10':
        return "A blur photo of an object with"
    elif cfg['dataset'] == 'cifar10-p':
        return "A photo of an object with "
    elif cfg['dataset'] == 'flower':
        return "A photo of the flower with "
    elif cfg['dataset'] == 'food':
        return "A photo of the food with "
    elif cfg['dataset'] == 'imagenet':
        return "A photo of an object with "
    elif cfg['dataset'] in ['imagenet-animal', 'imagenet-a']:
        return "A photo of an animal with "
    else:
        raise NotImplementedError


def update_num_attributes(args):
    config_num_att_dict = {
        'configs/cifar10.yaml': {'1': 8, '2': 16, '3': 32, '4': 64, '5': 128, '6': 256, '7': 10, '8': 20, '9': 'full'},
        'configs/cub.yaml': {'1': 8, '2': 16, '3': 32, '4': 64, '5': 128, '6': 256, '7': 200, '8': 400, '9': 'full'},
        'configs/cifar100_bn.yaml': {'1': 8, '2': 16, '3': 32, '4': 64, '5': 128, '6': 256, '7': 100, '8': 200,
                                     '9': 'full'},
        'configs/food_bn.yaml': {'1': 8, '2': 16, '3': 32, '4': 64, '5': 128, '6': 256, '7': 101, '8': 202, '9': 'full'},
        'configs/flower.yaml': {'1': 8, '2': 16, '3': 32, '4': 64, '5': 128, '6': 256, '7': 102, '8': 204, '9': 'full'},
        'configs/imagenet-animal_bn.yaml': {'1': 8, '2': 16, '3': 32, '4': 64, '5': 128, '6': 256, '7': 397, '8': 794,
                                            '9': 'full'},
    }
    att_dict = config_num_att_dict[args.config]
    args.num_attributes = att_dict[str(args.num_attributes)]
