import numpy as np
import torch
import copy
from dataloader.sampler import CategoriesSampler, PairBatchSampler
from torchvision.datasets.vision import VisionDataset
from utils_s import *

def get_dataloader(args, procD, clsD, bookD, session=None):
    if session==None:
        session = procD['session']
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args, clsD)
        return trainset, trainloader, testloader
    else:
        trainset, trainloader, testloader, new_testloader, prev_testloader, new_all_testloader, base_testloader \
            = get_new_dataloader(args, session, procD, clsD, bookD)
        return trainset, trainloader, testloader, new_testloader, prev_testloader, new_all_testloader, base_testloader


def get_base_dataloader(args, clsD):
    class_index = np.array(clsD['tasks'][0].cpu())
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(args, train=True, shotpercls=args.shotpercls, base_sess=True,
                                       root=args.dataroot, doubleaug=args.base_doubleaug, download=True, index=class_index)
        testset = args.Dataset.CIFAR100(args, train=False, shotpercls=args.shotpercls, base_sess=True,
                                      root=args.dataroot, download=False, index=class_index)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(args, train=True, shotpercls=args.shotpercls, base_sess=True,
                                       root=args.dataroot, doubleaug=args.base_doubleaug, index=class_index)
        testset = args.Dataset.CUB200(args, train=False, shotpercls=args.shotpercls, base_sess=True,
                                      root=args.dataroot, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(args, train=True, shotpercls=args.shotpercls, base_sess=True,
                                       root=args.dataroot, doubleaug=args.base_doubleaug, index=class_index)
        testset = args.Dataset.MiniImageNet(args, train=False, shotpercls=args.shotpercls, base_sess=True,
                                      root=args.dataroot, index=class_index)


    if args.base_dataloader_mode == 'plain':
        #trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
        #                                          num_workers=8, pin_memory=True, drop_last=True)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                                  num_workers=8, pin_memory=True)
    elif args.base_dataloader_mode == 'episodic':
        episampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                    args.episode_shot + args.episode_query)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=episampler,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)
    elif args.base_dataloader_mode == 'pair':
        pairsampler = PairBatchSampler(trainset.targets, args.batch_size_base // 2 )
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=pairsampler,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        raise NotImplementedError


    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader



def get_new_dataloader(args, session, procD, clsD, bookD):
    base_class_index = np.array(clsD['tasks'][0].cpu())
    #txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    class_new_tr, class_new_te, prev_class_list_te, new_all_class_list_te = get_session_classes(args, clsD, bookD, session)
    if args.dataset == 'cifar100':
        #trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
        #                                 index=class_index, base_sess=False)
        trainset = args.Dataset.CIFAR100(args, train=True, shotpercls=args.shotpercls, base_sess=False,
                                       root=args.dataroot, download=False, index=class_new_tr, shot = args.shot)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(args, train=True, shotpercls=args.shotpercls, base_sess=False,
                                       root=args.dataroot, index=class_new_tr, shot = args.shot)

    if args.dataset == 'mini_imagenet':
        #trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
        #                                     index_path=txt_path)
        trainset = args.Dataset.MiniImageNet(args, train=True, shotpercls=args.shotpercls, base_sess=False,
                                       root=args.dataroot, index=class_new_tr, shot = args.shot)


    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        if not args.use_coreset:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new,
                                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new,
                                                      shuffle=False, num_workers=args.num_workers, pin_memory=True)


    # test on all encountered classes

    if args.dataset == 'cifar100':
        #testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
        #                                index=class_new, base_sess=False)
        testset = args.Dataset.CIFAR100(args, train=False, base_sess=False,
                                      root=args.dataroot, download=False, index=class_new_te)
        new_testset = args.Dataset.CIFAR100(args, train=False, base_sess=False,
                                        root=args.dataroot, download=False, index=class_new_tr)
        prev_testset = args.Dataset.CIFAR100(args, train=False, base_sess=False,
                                            root=args.dataroot, download=False, index=prev_class_list_te)
        new_all_testset = args.Dataset.CIFAR100(args, train=False, base_sess=False,
                                             root=args.dataroot, download=False, index=new_all_class_list_te)
        base_testset = args.Dataset.CIFAR100(args, train=False, base_sess=False,
                                                root=args.dataroot, download=False, index=base_class_index)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(args, train=False, base_sess=False,
                                      root=args.dataroot, index=class_new_te)
        new_testset = args.Dataset.CUB200(args, train=False, base_sess=False,
                                      root=args.dataroot, index=class_new_tr)
        prev_testset = args.Dataset.CUB200(args, train=False, base_sess=False,
                                          root=args.dataroot, index=prev_class_list_te)
        new_all_testset = args.Dataset.CUB200(args, train=False, base_sess=False,
                                          root=args.dataroot, index=new_all_class_list_te)
        base_testset = args.Dataset.CUB200(args, train=False, base_sess=False,
                                              root=args.dataroot, index=base_class_index)
    if args.dataset == 'mini_imagenet':
        #testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
        #                                    index=class_new)
        testset = args.Dataset.MiniImageNet(args, train=False, base_sess=False,
                                      root=args.dataroot, index=class_new_te)
        new_testset = args.Dataset.MiniImageNet(args, train=False, base_sess=False,
                                            root=args.dataroot, index=class_new_tr)
        prev_testset = args.Dataset.MiniImageNet(args, train=False, base_sess=False,
                                            root=args.dataroot, index=prev_class_list_te)
        new_all_testset = args.Dataset.MiniImageNet(args, train=False, base_sess=False,
                                            root=args.dataroot, index=new_all_class_list_te)
        base_testset = args.Dataset.MiniImageNet(args, train=False, base_sess=False,
                                                    root=args.dataroot, index=base_class_index)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    new_testloader = torch.utils.data.DataLoader(dataset=new_testset, batch_size=args.test_batch_size,
                                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    prev_testloader = torch.utils.data.DataLoader(dataset=prev_testset, batch_size=args.test_batch_size,
                                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    new_all_testloader = torch.utils.data.DataLoader(dataset=new_all_testset, batch_size=args.test_batch_size,
                                                  shuffle=False, num_workers=args.num_workers, pin_memory=True)
    base_testloader = torch.utils.data.DataLoader(dataset=base_testset, batch_size=args.test_batch_size,
                                                     shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader, new_testloader, prev_testloader, new_all_testloader, base_testloader


def get_session_classes(args, clsD, bookD, session):
    class_list_tr =np.array(clsD['tasks'][session].cpu())
    #class_list_te = np.array(args.bookD[session]['seen'].cpu())
    class_list_te = np.array(bookD[session]['seen_unsort'].cpu())
    prev_class_list_te = np.array(bookD[session]['prev_unsort'].cpu())
    new_all_class_list_te = np.array(bookD[session]['seen_unsort'][args.base_class:].cpu())
    return class_list_tr, class_list_te, prev_class_list_te, new_all_class_list_te

def get_dataloader_coreset(args, coreset, coreset_labels):
    session = args.proc_book['session']
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session, coreset,coreset_labels)
    return trainset, trainloader, testloader


class dataset_withcoreset(VisionDataset):
    def __init__(self, args, dataloader, coreset, indexes, model, clsD):
        loader = copy.deepcopy(dataloader)
        #loader.
        features, labels = tot_datalist(args, dataloader, model, doubleaug=False, map= clsD['seen_unsort_map'])

        #data = dataloader.dataset.data
        #targets = dataloader.dataset.targets
        session = (len(indexes)-args.base_class)//args.way + 1
        _coreset = torch.cat([coreset[i] for i in range(session)], dim=0) # 0~sess-1
        _coreset = _coreset.view(-1, _coreset.shape[-1])
        label_coreset = torch.cat([torch.ones(args.shot)*k for k in range(len(indexes))],dim=0).cuda()
        assert len(_coreset) == len(label_coreset)

        self.data = torch.cat([features, _coreset], dim=0)
        self.targets = torch.cat([labels, label_coreset],dim=0).long()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self):
        return len(self.data)







