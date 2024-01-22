import os
import os.path as osp

import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MiniImageNet(Dataset):

    def __init__(self, args, train, base_sess, root='./data', shotpercls=False, doubleaug=False,
                 transform=None, index_path=None, index=None, shot=None):
        if train:
            setname = 'train'
        else:
            setname = 'test'
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.IMAGE_PATH = os.path.join(root, 'miniimagenet/images')
        self.SPLIT_PATH = os.path.join(root, 'miniimagenet/split')
        self.doubleaug = doubleaug

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb

        if train:
            image_size = 84

            tf_list = [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([  # added for sup
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2)
            ]
            default_tf = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]

            AL_train_transforms = [
                transforms.Resize([92, 92]),
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]



            if 0 in args.aug_type1:
                tf1 = default_tf
            else:
                tf1 = [tf_list[i - 1] for i in args.aug_type1] + default_tf

            if 0 in args.aug_type2:
                tf2 = default_tf
            else:
                tf2 = [tf_list[i - 1] for i in args.aug_type2] + default_tf

            self.transform = transforms.Compose(tf1)
            self.transform2 = transforms.Compose(tf2)
            #self.transform = transforms.Compose(AL_train_transforms)
            #self.transform2 = transforms.Compose(AL_train_transforms)


            """
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            self.transform2 = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
             """
            #if base_sess:
            #    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            #else:
            #    self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)

            if not shotpercls:
                if base_sess:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromClasses_shot(self.data, self.targets, index, shot)
            else:
                if base_sess:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromClasses_shot(self.data, self.targets, index, shot)
        else:
            image_size = 84
            #"""
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            # """
            """
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            """
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def SelectfromClasses_shot(self, data, targets, index,shot):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            ind_cl_shot = np.random.permutation(ind_cl)[:shot]
            for j in ind_cl_shot:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        tmp_zip = list(zip(data_tmp, targets_tmp))
        random.shuffle(tmp_zip)
        data_, targets_ = zip(*tmp_zip)
        return list(data_), list(targets_)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.doubleaug is False:
            path, targets = self.data[i], self.targets[i]
            image = self.transform(Image.open(path).convert('RGB'))
            return image, targets
        else:
            path, targets = self.data[i], self.targets[i]
            image = self.transform(Image.open(path).convert('RGB'))
            image2 = self.transform2(Image.open(path).convert('RGB'))
            return [image,image2], targets


if __name__ == '__main__':
    txt_path = "../../data/index_list/mini_imagenet/session_1.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '~/data'
    batch_size_base = 400
    trainset = MiniImageNet(root=dataroot, train=True, transform=None, index_path=txt_path)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
    print(trainloader.dataset.data.shape)
    # txt_path = "../../data/index_list/cifar100/session_2.txt"
    # # class_index = open(txt_path).read().splitlines()
    # class_index = np.arange(base_class)
    # trainset = CIFAR100(root=dataroot, train=True, download=True, transform=None, index=class_index,
    #                     base_sess=True)
    # cls = np.unique(trainset.targets)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
    #                                           pin_memory=True)
