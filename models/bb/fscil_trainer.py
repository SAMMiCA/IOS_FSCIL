import os.path as osp
import torch.nn as nn
from copy import deepcopy
import seaborn as sns
import abc
import torch.nn.functional as F
import pdb
import torch.backends.cudnn as cudnn


from utils_s import *
from dataloader.data_utils_s import *
from tensorboardX import SummaryWriter
from .Network import MYNET
from tqdm import tqdm
from SupConLoss import SupConLoss

class FSCILTrainer(object, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        pass

    def init_vars(self, args):
        # Setting arguments
        #if args.start_session == 0:
        if args.load_dir == None:
            args.secondphase = False
        else:
            args.secondphase = True

        if args.dataset == 'cifar100':
            import dataloader.cifar100.cifar_s as Dataset
            args.num_classes = 100
            args.width = 32
            args.height = 32

            if args.base_class == None and args.way == None:
                args.base_class = 60
                args.way = 5
                # args.shot = 5
                args.sessions = 9
            elif not args.base_class == None and not args.way == None:
                args.sessions = 1 + int((args.num_classes - args.base_class) / args.way)
            else:
                raise NotImplementedError

        if args.dataset == 'cub200':
            import dataloader.cub200.cub200_s as Dataset
            args.num_classes = 200
            args.width = 224
            args.height = 224

            if args.base_class == None and args.way == None:
                args.base_class = 100
                args.way = 10
                # args.shot = 5
                args.sessions = 11
            elif not args.base_class == None and not args.way == None:
                args.sessions = 1 + int((args.num_classes - args.base_class) / args.way)
            else:
                raise NotImplementedError

        if args.dataset == 'mini_imagenet':
            import dataloader.miniimagenet.miniimagenet_s as Dataset
            args.num_classes = 100
            args.width = 84
            args.height = 84

            if args.base_class == None and args.way == None:
                args.base_class = 60
                args.way = 5
                # args.shot = 5
                args.sessions = 9
            elif not args.base_class == None and not args.way == None:
                args.sessions = 1 + int((args.num_classes - args.base_class) / args.way)
            else:
                raise NotImplementedError

        args.Dataset = Dataset
        args.eps = 1e-7

        if args.base_dataloader_mode != 'episodic':
            args.shotpercls = False
        else:
            args.shotpercls = True




        # Setting save path
        #mode = args.base_mode + '-' + args.new_mode
        mode = args.fw_mode
        if not args.no_rbfc:
            mode = mode + '-' + 'rbfc'

        bfzb = 'T' if args.base_freeze_backbone else 'F'
        ifzb = 'T' if args.inc_freeze_backbone else 'F'

        bdg = 'T' if args.base_doubleaug else 'F'
        idg = 'T' if args.inc_doubleaug else 'F'

        rcnm = 'T' if args.rpclf_normmean else 'F'

        if args.schedule == 'Milestone':
            mile_stone = str(args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            schedule = 'MS_%s'%(mile_stone)
        elif args.schedule == 'Step':
            schedule = 'Step_%d'%(args.step)
        elif args.schedule == 'Cosine':
            schedule = 'schecos'
        if args.schedule_new == 'Milestone':
            mile_stone_new = str(args.milestones_new).replace(" ", "").replace(',', '_')[1:-1]
            schedule_new = 'MS_%s'%(mile_stone_new)
        elif args.schedule_new == 'Step':
            raise NotImplementedError
            #schedule = 'Step_%d'%(args.step)
        elif args.schedule_new == 'Cosine':
            schedule_new = 'sch_c'

        if args.batch_size_base > 256:
            args.warm = True
        else:
            args.warm = False
        if args.warm:
            #args.model_name = '{}_warm'.format(opt.model_name)
            args.warmup_from = 0.01
            args.warm_epochs = 10
            if args.schedule == 'Cosine':
                eta_min = args.lr_base * (args.gamma ** 3)
                args.warmup_to = eta_min + (args.lr_base - eta_min) * (
                        1 + math.cos(math.pi * args.warm_epochs / args.epochs_base)) / 2
            else:
                args.warmup_to = args.lr_base



        # sch_c: shcedule cosine
        # bdm: base_dataloader_mode
        # l: lambda
        # sd : seed
        # SC: == supcon, use_supconloss T
        # BCF: base classifier fine-tune, == base_clf_ft T
        # Epf, Lrf: epoch, lr for fine-tune
        # RN == resnet, SCRN == Supconresnet
        # 2P == 2ndphase, st == start
        # hd == head_type, D == head_dim
        # SCA == supcon_angle
        # CK: cskd
        # CRN: CIFAR_RESNET
        # EM: encmlp
        # G: gauss, bt: tukey_beta, ns: num_sampled

        str_loss = '-'
        if args.use_celoss:
            str_loss += 'ce'
            if args.fw_mode == 'fc_cosface':
                str_loss += '_s%.1fm%.1f-'%(args.s, args.m)
        if args.use_supconloss:
            SCA_ = 'T' if args.supcon_angle else 'F'
            scstr_ = 'SC-SCA_%s-' % (SCA_)
            str_loss += scstr_
        #if args.use_cskdloss:
        #    str_loss += 'CK-l%.1f-'%(args.lamda)
        if args.use_cskdloss_1:
            str_loss += 'CK1-l%.1f-'%(args.lamda)
        if args.use_cskdloss_2:
            str_loss += 'CK2-l%.1f-'%(args.lamda)
        auglist1_ = ''.join(str(x) for x in args.aug_type1)
        auglist2_ = ''.join(str(x) for x in args.aug_type2)
        str_loss += 'aug_%s,%s-' % (auglist1_, auglist2_)

        if args.use_head:
            str_loss += 'hd%sD_%d-'%(args.head_type, args.head_dim)
        if args.use_encmlp:
            str_loss += 'EM%d_D%d-'%(args.encmlp_layers, args.encmlp_dim)

        hyper_name_list = 'Epob_%d-Epon_%d-Lrb_%.4f-Lrn_%.4f-%s-%s-Gam_%.2f-Dec_%.5f-Bs_%d-Mom_%.2f' \
                                    'bsc_%d-way_%d-shot_%d-bfzb_%s-ifzb_%s-bdg_%s-idg_%s-rcnm_%s-bdm_%s' \
                                    '-sd_%d' % (
                            args.epochs_base, args.epochs_new, args.lr_base, args.lr_new, schedule, schedule_new, \
                            args.gamma,
                            args.decay, args.batch_size_base, args.momentum, args.base_class, args.way, args.shot, bfzb,
                            ifzb,
                            bdg, idg, rcnm, args.base_dataloader_mode,
                            args.seed)
        hyper_name_list += str_loss
        if args.base_clf_ft:
            hyper_name_list = hyper_name_list + 'BCF'
            hyper_name_list = hyper_name_list + '-Epf_%d-Lrf_%.4f' %(args.epochs_base_clf, args.lr_base_clf)

        #if args.warm:
        #    hyper_name_list += '-warm'

        if not args.secondphase:
            save_path = '%s/' % args.dataset
            save_path = save_path + '%s/' % args.project
            save_path = save_path + '%s-st_%d/' % (mode, args.start_session)
            #save_path += 'mlp-ftenc'
            if args.dataset == 'cifar100':
                if args.use_cifar_resnet18:
                    if args.use_cifar_resnet18_opt1:
                        save_path = save_path + 'RN18_1-'
                    elif args.use_cifar_resnet18_opt2:
                        save_path = save_path + 'RN18_2-'
                    elif args.use_cifar_resnet18_opt3:
                        save_path = save_path + 'RN18_3-'
                    else:
                        save_path = save_path + 'RN18-'
                elif args.use_supcon_resnet18:
                    save_path = save_path + 'SCRN18-'
                else:
                    save_path = save_path + 'RN20-'
            if args.dataset == 'mini_imagenet':
                if args.use_cifar_resnet18_mini:
                    save_path = save_path + 'CRN18-'
            save_path = save_path + hyper_name_list
            save_path = os.path.join('checkpoint', save_path)
            ensure_path(save_path)
            args.save_path = save_path
        else:
            hyper_name_list = '2Pst_%d-%s'%(args.start_session, hyper_name_list)
            save_path = os.path.join(args.load_dir, hyper_name_list)
            ensure_path(save_path)
            args.save_path = save_path

        if args.gauss:
            args.save_path += '-G_bt%.1f-ns%d'%(args.tukey_beta, args.num_sampled)


        # Setting dictionaries
        # gaussD
        gaussD = {}

        #clsD initialize
        clsD = {}

        #clsD, procD
        if args.secondphase == False:
            args.task_class_order = np.random.permutation(np.arange(args.num_classes)).tolist() #######
            #args.task_class_order = np.arange(args.num_classes).tolist()  #######
            clsD['tasks'], clsD['class_maps'] = init_maps(args)
            #self.args.task_class_order = (np.arange(self.args.num_classes)).tolist()

            procD = {}
            #varD['proc_book'] = {}

            # train statistics
            procD['trlog'] = {}
            procD['trlog']['train_loss'] = []
            procD['trlog']['val_loss'] = []
            procD['trlog']['test_loss'] = []
            procD['trlog']['train_acc'] = []
            procD['trlog']['val_acc'] = []
            procD['trlog']['test_acc'] = []
            procD['trlog']['max_acc_epoch'] = 0
            procD['trlog']['max_acc'] = [0.0] * args.sessions
            procD['trlog']['max_acc_base_rbfc'] = 0.0
            procD['trlog']['max_acc_base_after_ft'] = 0.0
            procD['trlog']['new_max_acc'] = [0.0] * args.sessions # first will be left 0.0
            procD['trlog']['new_all_max_acc'] = [0.0] * args.sessions  # first will be left 0.0
            procD['trlog']['base_max_acc'] = [0.0] * args.sessions  # first will be left 0.0
            procD['trlog']['prev_max_acc'] = [0.0] * args.sessions  # first will be left 0.0
            procD['trlog']['prev_new_clf_ratio'] = [0.0] * args.sessions  # first will be left 0.0
            procD['trlog']['new_new_clf_ratio'] = [0.0] * args.sessions  # first will be left 0.0
            procD['session'] = -1

            bookD = book_val(args)

        else:
            assert args.load_dir != None

            # load objs
            obj_dir = os.path.join(args.load_dir, 'saved_dicts')
            with open(obj_dir, 'rb') as f:
                dict_ = pickle.load(f)
                procD = dict_['procD']
                bookD = dict_['bookD']

            procD['session'] = args.start_session - 1
            # epoch, step is init to -1 for every sessions so no worry.
            if args.start_session > 0:
                procD['trlog']['max_acc'][args.start_session:] = [0.0] * (args.sessions-args.start_session)
                procD['trlog']['new_max_acc'][args.start_session:] = [0.0] * (args.sessions-args.start_session)
                procD['trlog']['new_all_max_acc'][args.start_session:] = [0.0] * (args.sessions-args.start_session)
                procD['trlog']['base_max_acc'][args.start_session:] = [0.0] * (args.sessions-args.start_session)
                procD['trlog']['prev_max_acc'][args.start_session:] = [0.0] * (args.sessions-args.start_session)
                procD['trlog']['prev_new_clf_ratio'][args.start_session:] = [0.0] * (args.sessions-args.start_session)
                procD['trlog']['new_new_clf_ratio'][args.start_session:] = [0.0] * (args.sessions-args.start_session)
            else:
                procD['trlog']['new_max_acc'][args.start_session:] = [0.0] * (args.sessions-args.start_session)
                procD['trlog']['new_all_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['base_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['prev_max_acc'][args.start_session:] = [0.0] * (args.sessions-args.start_session)

            tasks = bookD[0]['tasks'] # Note that bookD[i]['tasks'] is same for each i
            class_order_ = []
            for i in range(len(tasks)):
                class_order_ += tasks[i].tolist()
            # clsD to up-to-date
            args.task_class_order = class_order_
            clsD['tasks'], clsD['class_maps'] = init_maps(args)
            #for i in range(args.start_session):

            if args.start_session == 0 or args.start_session == 1:
                pass
            else:
                clsD = inc_maps(args, clsD, procD, args.start_session-1)

        return args, procD, gaussD, clsD, bookD


    def get_optimizer_base(self, args, model):
        """
        #이걸 각색할것.
        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                     {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        #이것도 참고
        self.parameters_to_train = []
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        """
        """
        self.parameters_to_train = []
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["encoder"].parameters())
        """
        if args.base_freeze_backbone:
            for param in model.encoder.module.parameters():
                param.requires_grad = False
        # for param in self.model.module.fc.parameters():
        #    param.requires_grad = False

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.module.parameters()),
        #                             lr=0.0003, weight_decay=0.0008)

        #optimizer = torch.optim.SGD(model.parameters(), #####
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
        #optimizer = torch.optim.SGD(model.module.parameters(),
                                    #args.lr_base, momentum=0.9, nesterov=True, weight_decay=args.decay)
                                    args.lr_base, momentum=0.9, weight_decay=args.decay)
        # optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
        #                            weight_decay=self.args.decay)
        if args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        elif args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                             gamma=args.gamma)
        elif args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_base)

        return model, optimizer, scheduler

    """
    def get_dataloader(self, session):

        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader
    """
    def get_optimizer_ft(self, args, model):

        #for param in model.module.encoder.parameters():
        #    param.requires_grad = False
        if args.use_head:
            for param in model.head.parameters():
                param.requires_grad = False

        """
        enc_param_list = list(kv[0] for kv in model.module.encoder.named_parameters())
        enc_param_list = ['encoder.' + k for k in enc_param_list]

        enc_params = list(filter(lambda kv: kv[0] in enc_param_list and kv[1].requires_grad,
                                 model.module.named_parameters()))
        else_params = list(filter(lambda kv: kv[0] not in enc_param_list and kv[1].requires_grad,
                                  model.module.named_parameters()))

        enc_params = [i[1] for i in enc_params]
        else_params = [i[1] for i in else_params]
        
        
        optimizer = torch.optim.SGD([{'params': enc_params, 'lr': args.lr_new_enc},
                                     {'params': else_params, 'lr': args.lr_new}],
                                    # momentum=0.9, dampening=0.9, weight_decay=self.args.decay)
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        """
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    args.lr_base_clf, momentum=0.9, nesterov=True, weight_decay=args.decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_base_clf)
        return model, optimizer, scheduler

    def get_optimizer_new(self, args, model):
        #assert self.args.angle_mode is not None

        if args.inc_freeze_backbone:
            for param in model.encoder.parameters():
                #param.requires_grad = False
                param.requires_grad = False

        if args.use_head:
            for param in model.head.parameters():
                param.requires_grad = False

        #for param in self.model.module.fc.parameters():
        #    param.requires_grad = False

        #set_trainable_param(self.model.module.angle_w, [i for i in range(self.args.proc_book['session']+1)])


        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.module.parameters()),
        #                            self.args.lr_new, momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.module.parameters()),
        #                            self.args.lr_new, momentum=0.9, dampening=0.9,weight_decay=0) #Last ver.

        enc_param_list = list(kv[0] for kv in model.encoder.named_parameters())
        enc_param_list = ['encoder.' + k for k in enc_param_list]

        enc_params = list(filter(lambda kv: kv[0] in enc_param_list and kv[1].requires_grad,
                                 model.named_parameters()))
        else_params = list(filter(lambda kv: kv[0] not in enc_param_list and kv[1].requires_grad,
                                  model.named_parameters()))

        enc_params = [i[1] for i in enc_params]
        else_params = [i[1] for i in else_params]

        optimizer = torch.optim.SGD([{'params': enc_params, 'lr': args.lr_new_enc},
                                     {'params': else_params, 'lr': args.lr_new}],
                                    #momentum=0.9, dampening=0.9, weight_decay=self.args.decay)
                                    momentum=0.9, dampening=0.9, weight_decay=0)

        if args.schedule_new == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_new, gamma=args.gamma)
        elif args.schedule_new == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones_new,
                                                             gamma=args.gamma)
        elif args.schedule_new == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_base)

        return model, optimizer, scheduler


    def base_train(self, args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler, epoch, supcon_criterion=None):
        tl = Averager()
        ta = Averager()
        # self.model = self.model.train()
        model.train()
        # standard classification for pretrain
        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):
            procD['step'] += 1
            warmup_learning_rate(args, epoch, i, len(trainloader), optimizer)

            if args.base_doubleaug is False:
                data, train_label = [_.cuda() for _ in batch]
                target_cls = clsD['class_maps'][0][train_label]
            else:
                #data = batch[0][0].cuda()
                #train_label = batch[1].cuda()
                data = torch.cat((batch[0][0],batch[0][1]),dim=0).cuda()
                  #train_label = batch[1].repeat(2).cuda()
                train_label = batch[1].cuda()
                train_label = train_label.repeat(2)
                target_cls = clsD['class_maps'][0][train_label]

            model.set_mode('encoder')
            data = model(data)
            bsz_ = data.shape[0]
            bsz_half = bsz_ // 2
            data_f = data[:bsz_half]
            data_b = data[bsz_half:]
            targets_ = target_cls[:bsz_half]
            # tasks 대신 seen_unsort / seen_unsort_map으로 바꾼 후 아래랑 합쳐도 무방.
            if not args.base_doubleaug:
                assert args.use_supconloss == False
                assert args.use_cskdloss_1 == False
                assert args.use_cskdloss_2 == False

                if not args.base_dataloader_mode == 'pair':
                    assert args.use_celoss == True
                    model.set_mode(args.fw_mode)
                    if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                        logits_cls, _ = model(data, target_cls, sess=procD['session'], doenc=False)
                    else:
                        logits_cls = model(data, sess=procD['session'], doenc=False)
                    loss = F.cross_entropy(logits_cls, target_cls)
                    acc = count_acc(logits_cls, target_cls)
                    total_loss = loss
                else:
                    model.set_mode(args.fw_mode)
                    if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                        logits_cls_f, _ = model(data_f, targets_, sess=procD['session'],doenc=False)
                        with torch.no_grad():
                            logits_cls_b, _ = model(data_b, targets_, sess=procD['session'],doenc=False)
                    else:
                        logits_cls_f = model(data_f, sess=procD['session'],doenc=False)
                        with torch.no_grad():
                            logits_cls_b = model(data_b, sess=procD['session'],doenc=False)

                    #total_loss=0.0
                    total_loss = torch.tensor(0.0).cuda()
                    if args.use_celoss:
                        loss_pair_ce = F.cross_entropy(logits_cls_f, targets_)
                        total_loss += loss_pair_ce
                    """
                    if args.use_cskdloss_1:
                        loss_cskd = kdloss(logits_cls_f, logits_cls_b.detach()) * args.lamda
                        total_loss += loss_cskd
                    if args.use_cskdloss_2:
                        raise NotImplementedError
                    """
                    if total_loss==0:
                        print("loss is zero loss args needed")
                    acc = count_acc(logits_cls_f, targets_)
            else:

                model.set_mode(args.fw_mode)
                if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                    #logits_cls, wf = model(data, target_cls, sess=procD['session'], doenc=False) #####
                    logits_cls, _ = model(data, target_cls, sess=procD['session'], doenc=False)  #####
                else:
                    logits_cls = model(data, sess=procD['session'],doenc=False)

                total_loss = torch.tensor(0.0).cuda()
                if args.use_celoss:
                    #"""
                    loss_ce = F.cross_entropy(logits_cls, target_cls)
                    total_loss += loss_ce
                    #""" ####
                    """
                    numerator = args.s * (torch.diagonal(wf.transpose(0, 1)[target_cls]) - args.m)
                    excl = torch.cat(
                        [torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(target_cls)], dim=0)
                    denominator = torch.exp(numerator) + torch.sum(torch.exp(args.s * excl), dim=1)
                    L = numerator - torch.log(denominator)
                    #total_loss += -torch.mean(L)
                    cosfaceloss1 = -torch.mean(L[:args.batch_size_base])
                    cosfaceloss2 = -torch.mean(L[args.batch_size_base:])
                    total_loss += 0.5*cosfaceloss1 + 0.5 + cosfaceloss2
                    """
                if args.use_supconloss:
                    #args.use_head asserted true
                    model.set_mode('head')
                    features = model(data, doenc=False)
                    f1, f2 = torch.split(features, [bsz_half, bsz_half], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    loss_supcon = supcon_criterion(features, targets_)
                    total_loss += loss_supcon

                """
                if args.use_cskdloss_1:
                    model.set_mode(args.fw_mode)
                    if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                        logits_cls_f, _ = model(data_f, targets_, sess=0, doenc=False)
                        with torch.no_grad():
                            logits_cls_b, _ = model(data_b, targets_, sess=0, doenc=False)
                    else:
                        logits_cls_f = model(data_f, sess=0, doenc=False)
                        with torch.no_grad():
                            logits_cls_b = model(data_b, sess=0, doenc=False)

                    # loss_pair_ce = F.cross_entropy(logits_cls_f, targets_)
                    loss_cskd = kdloss(logits_cls_f, logits_cls_b.detach()) * args.lamda
                    total_loss += loss_cskd
                if args.use_cskdloss_2:
                    data2 = data_b.clone().detach()
                    data3 = deepcopy(data2)
                    ll = targets_.contiguous().view(-1, 1)
                    map = torch.eq(ll, ll.T).float().cuda() - torch.eye(len(targets_)).cuda()
                    for i in range(len(targets_)):
                        if map[i].sum() == 0:
                            pass
                        else:
                            idx_ = (map[i] == 1.0).nonzero(as_tuple=True)[0]
                            ii = idx_[torch.randperm(len(idx_))[:1]]
                            data2[i] = data3[ii]

                    model.set_mode(args.fw_mode)
                    if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                        logits_cls_f, _ = model(data_f, targets_, sess=0, doenc=False)
                        with torch.no_grad():
                            logits_cls_b, _ = model(data2, targets_, sess=0, doenc=False)
                    else:
                        logits_cls_f = model(data_f, sess=0, doenc=False)
                        with torch.no_grad():
                            logits_cls_b = model(data2, sess=0, doenc=False)

                    # loss_pair_ce = F.cross_entropy(logits_cls_f, targets_)
                    loss_cskd = kdloss(logits_cls_f, logits_cls_b.detach()) * args.lamda
                    total_loss += loss_cskd
                """
                acc = count_acc(logits_cls, target_cls)

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            #loss.backward()
            total_loss.backward()
            optimizer.step()
            # self.step += 1
        tl = tl.item()
        ta = ta.item()
        return tl, ta


    def base_train_ft(self, args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler, epoch):
        tl = Averager()
        ta = Averager()
        model.train()
        # standard classification for pretrain

        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):
            if args.base_doubleaug is False:
                data, train_label = [_.cuda() for _ in batch]
                target_cls = clsD['class_maps'][0][train_label]
            else:
                data = torch.cat((batch[0][0],batch[0][1]),dim=0).cuda()
                train_label = batch[1].cuda()
                train_label = train_label.repeat(2)
                target_cls = clsD['class_maps'][0][train_label]

            if not args.base_doubleaug:
                assert args.use_supconloss == False

                if not args.base_dataloader_mode == 'pair':
                    model.set_mode(args.fw_mode)
                    if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                        logits_cls, _ = model(data, target_cls, sess=procD['session'])
                    else:
                        logits_cls = model(data, sess=procD['session'])
                    loss = F.cross_entropy(logits_cls, target_cls)
                    acc = count_acc(logits_cls, target_cls)
                    total_loss = loss
                else:
                    total_loss = torch.tensor(0.0).cuda()

                    bsz_ = data.size(0)
                    targets_ = target_cls[:bsz_ // 2]
                    model.set_mode(args.fw_mode)
                    if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                        logits_cls_f, _ = model(data[:bsz_//2], targets_, sess=procD['session'])
                    else:
                        logits_cls_f = model(data[:bsz_//2], sess=procD['session'])

                    loss_pair_ce = F.cross_entropy(logits_cls_f, targets_)
                    total_loss += loss_pair_ce
                    if args.use_cskdloss_1 or args.use_cskdloss_2:
                        raise NotImplementedError
                    acc = count_acc(logits_cls_f, targets_)

            else:

                model.set_mode('encoder')
                data = model(data)
                bsz_ = data.shape[0]
                bsz_half = bsz_ // 2
                data_f = data[:bsz_half]
                data_b = data[bsz_half:]
                targets_ = target_cls[:bsz_half]
                model.set_mode(args.fw_mode)
                if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                    # logits_cls, wf = model(data, target_cls, sess=procD['session'], doenc=False) #####
                    logits_cls, _ = model(data, target_cls, sess=procD['session'], doenc=False)  #####
                else:
                    logits_cls = model(data, sess=procD['session'], doenc=False)

                total_loss = torch.tensor(0.0).cuda()
                #if args.use_celoss:
                    # """
                loss_ce = F.cross_entropy(logits_cls, target_cls)
                total_loss += loss_ce
                #### CE is default setting for fine-tuning

                if args.use_cskdloss_1:
                    model.set_mode(args.fw_mode)
                    if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                        logits_cls_f, _ = model(data_f, targets_, sess=0, doenc=False)
                        with torch.no_grad():
                            logits_cls_b, _ = model(data_b, targets_, sess=0, doenc=False)
                    else:
                        logits_cls_f = model(data_f, sess=0, doenc=False)
                        with torch.no_grad():
                            logits_cls_b = model(data_b, sess=0, doenc=False)

                    # loss_pair_ce = F.cross_entropy(logits_cls_f, targets_)
                    loss_cskd = kdloss(logits_cls_f, logits_cls_b.detach()) * args.lamda
                    total_loss += loss_cskd
                if args.use_cskdloss_2:
                    data2 = data_b.clone().detach()
                    data3 = deepcopy(data2)
                    ll = targets_.contiguous().view(-1, 1)
                    map = torch.eq(ll, ll.T).float().cuda() - torch.eye(len(targets_)).cuda()
                    for i in range(len(targets_)):
                        if map[i].sum() == 0:
                            pass
                        else:
                            idx_ = (map[i] == 1.0).nonzero(as_tuple=True)[0]
                            ii = idx_[torch.randperm(len(idx_))[:1]]
                            data2[i] = data3[ii]

                    model.set_mode(args.fw_mode)
                    if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                        logits_cls_f, _ = model(data_f, targets_, sess=0, doenc=False)
                        with torch.no_grad():
                            logits_cls_b, _ = model(data2, targets_, sess=0, doenc=False)
                    else:
                        logits_cls_f = model(data_f, sess=0, doenc=False)
                        with torch.no_grad():
                            logits_cls_b = model(data2, sess=0, doenc=False)

                    # loss_pair_ce = F.cross_entropy(logits_cls_f, targets_)
                    loss_cskd = kdloss(logits_cls_f, logits_cls_b.detach()) * args.lamda
                    total_loss += loss_cskd
                acc = count_acc(logits_cls, target_cls)

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            #loss.backward()
            total_loss.backward()
            optimizer.step()
            # self.step += 1
        tl = tl.item()
        ta = ta.item()
        return tl, ta



    def new_train(self, args, model, procD, clsD, gaussD, trainloader, optimizer, kdloss, scheduler, epoch):
        tl = Averager()
        ta = Averager()
        # self.model = self.model.train()
        model.train()
        # standard classification for pretrain

        if args.gauss:
            beta = args.tukey_beta

            assert args.batch_size_new == 0
            model.set_mode('encoder')
            for batch in trainloader:
                data, train_label = [_.cuda() for _ in batch]
                target_cls = clsD['class_maps'][procD['session']][train_label]
                data, label = [_.cuda() for _ in batch]
                cur_feat = model(data)
                cur_label = target_cls
            with torch.no_grad():
                # torch cpu
                base_means = gaussD['base_mean']
                base_cov = gaussD['base_cov']
                num_sampled = args.num_sampled

                data_ = cur_feat.cpu()
                label_ = cur_label.cpu()
                sampled_data = []
                sampled_label = []
                for i in range(len(data_)):
                    mean, cov = distribution_calibration(torch.pow(data_[i],beta), base_means, base_cov, k=2)

                    sampled = torch.tensor(np.random.multivariate_normal(mean=mean,cov=cov, size=num_sampled)).float()
                    sampled = torch.pow(sampled,1/beta)
                    sampled_data.append(sampled)
                    sampled_label.extend([label_[i]] * num_sampled)
                #sampled_data = torch.cat([sampled_data[:]]).reshape(len(data_) * num_sampled, -1)
                sampled_data = torch.cat(sampled_data,dim=0)
                sampled_label = torch.tensor(sampled_label)

            # X_aug = torch.cat([data_, sampled_data],dim=0).cuda()
            # Y_aug = torch.cat([label_, sampled_label],dim=0).cuda()
            cur_feat_aug = torch.cat([cur_feat, sampled_data.cuda().detach()], dim=0)
            cur_label_aug = torch.cat([cur_label, sampled_label.cuda().detach()], dim=0)
            #cur_label_aug_cls = self.args.cls_book['seen_unsort_map'][torch.tensor(cur_label_aug, dtype=int)]

            # tasks 대신 seen_unsort / seen_unsort_map으로 바꾼 후 아래랑 합쳐도 무방.
            model.set_mode(args.fw_mode)
            if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                logits_cls, _ = model(cur_feat_aug, cur_label_aug, sess=procD['session'], doenc=False)
            else:
                logits_cls = model(cur_feat_aug, sess=procD['session'], doenc=False)
            loss = F.cross_entropy(logits_cls, cur_label_aug)
            acc = count_acc(logits_cls, cur_label_aug)
            total_loss = loss

            optimizer.zero_grad()
            #loss.backward()
            total_loss.backward()
            optimizer.step()

            return total_loss, acc
        else:
            tqdm_gen = tqdm(trainloader)
            for i, batch in enumerate(tqdm_gen, 1):
                procD['step'] += 1
                data, train_label = [_.cuda() for _ in batch]
                target_cls = clsD['class_maps'][procD['session']][train_label]

                # tasks 대신 seen_unsort / seen_unsort_map으로 바꾼 후 아래랑 합쳐도 무방.
                if not args.inc_dataloader_mode == 'pair':
                    model.set_mode(args.fw_mode)
                    if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                        logits_cls, _ = model(data, target_cls, sess=procD['session'])
                    else:
                        logits_cls = model(data, sess=procD['session'])
                    loss = F.cross_entropy(logits_cls, target_cls)
                    acc = count_acc(logits_cls, target_cls)
                    total_loss = loss
                else:
                    bsz_ = data.size(0)
                    targets_ = target_cls[:bsz_ // 2]

                    model.set_mode(args.fw_mode)
                    if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                        logits_cls_f, _ = model(data[:bsz_ // 2], targets_, sess=procD['session'])
                        with torch.no_grad():
                            logits_cls_b, _ = model(data[bsz_ // 2:], targets_, sess=procD['session'])
                    else:
                        logits_cls_f = model(data[:bsz_ // 2], sess=procD['session'])
                        with torch.no_grad():
                            logits_cls_b = model(data[bsz_ // 2:], sess=procD['session'])

                    loss_pair_ce = F.cross_entropy(logits_cls_f, targets_)
                    loss_pair_kdl = kdloss(logits_cls_f, logits_cls_b.detach())
                    total_loss = loss_pair_ce
                    total_loss += loss_pair_kdl * args.lamda
                    acc = count_acc(logits_cls_f, targets_)

                lrc = scheduler.get_last_lr()[0]
                tqdm_gen.set_description(
                    'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(),
                                                                                        acc))
                tl.add(total_loss.item())
                ta.add(acc)

                optimizer.zero_grad()
                # loss.backward()
                total_loss.backward()
                optimizer.step()
                # self.step += 1
            tl = tl.item()
            ta = ta.item()
            return tl, ta

    def test(self, args, model, procD, clsD, testloader):
        epoch = procD['epoch']
        session = procD['session']

        model.eval()
        vl = Averager()
        va = Averager()
        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = [_.cuda() for _ in batch]
                if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                    model.set_mode('fc_cos')
                else:
                    model.set_mode(args.fw_mode)
                logits = model(data, sess=procD['session'])

                if session == 0:
                    #logits_cls = logits[:, clsD['tasks'][0]]
                    logits_cls = logits
                    target_cls = clsD['class_maps'][0][test_label]
                    # tasks 대신 seen_unsort / seen_unsort_map으로 바꾼 후 아래랑 합쳐도 무방.
                    loss = F.cross_entropy(logits_cls, target_cls)
                    acc = count_acc(logits_cls, target_cls)
                else:
                    #logits = logits[:, clsD['seen_unsort']]
                    target_cls = clsD['seen_unsort_map'][test_label]
                    # seen, seen_map results same.
                    loss = F.cross_entropy(logits, target_cls)
                    acc = count_acc(logits, target_cls)

                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()
        if session == 0:
            print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        else:
            print('session {}, test, loss={:.4f} acc={:.4f}'.format(session, vl, va))

        return vl, va

    def test2(self, args, model, procD, clsD, trainloader, testloader):
        epoch = procD['epoch']
        session = procD['session']
        model2 = deepcopy(model)
        model2.eval()
        vl = Averager()
        va = Averager()
        model2 = self.replace_clf(args, model2, procD, clsD, trainloader, testloader.dataset.transform, args.rbfc_opt2)

        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = [_.cuda() for _ in batch]
                if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                    model2.set_mode('fc_cos')
                else:
                    model2.set_mode(args.fw_mode)
                logits = model2(data, sess=procD['session'])

                if session == 0:
                    #logits_cls = logits[:, clsD['tasks'][0]]
                    logits_cls = logits
                    target_cls = clsD['class_maps'][0][test_label]
                    # tasks 대신 seen_unsort / seen_unsort_map으로 바꾼 후 아래랑 합쳐도 무방.
                    loss = F.cross_entropy(logits_cls, target_cls)
                    acc = count_acc(logits_cls, target_cls)
                else:
                    #logits = logits[:, clsD['seen_unsort']]
                    target_cls = clsD['seen_unsort_map'][test_label]
                    # seen, seen_map results same.
                    loss = F.cross_entropy(logits, target_cls)
                    acc = count_acc(logits, target_cls)

                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()
        if session == 0:
            print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        else:
            print('session {}, test, loss={:.4f} acc={:.4f}'.format(session, vl, va))

        return vl, va



    def newtest(self,  args, model, procD, clsD, testloader):
        epoch = procD['epoch']
        session = procD['session']

        model.eval()
        vl = Averager()
        va = Averager()
        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            sum_ = 0
            num_ = 0
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = [_.cuda() for _ in batch]
                if args.fw_mode == 'fc_cosface' or args.fw_mode == 'fc_arcface':
                    model.set_mode('fc_cos')
                else:
                    model.set_mode(args.fw_mode)

                # target_cls = self.args.cls_book['class_maps'][0][test_label]
                target_cls = clsD['seen_unsort_map'][test_label]
                logits = model(data, target_cls, session)
                # loss = F.cross_entropy(logits_cls, target_cls)
                # acc = new_count_acc(cosine_cls, target_cls)
                pred = torch.argmax(logits, dim=1)
                sum_ += torch.sum(pred > args.base_class)
                num_ += len(pred)

        return sum_ / num_


    def main(self, args):
        timer = Timer()
        args, procD, gaussD, clsD, bookD = self.init_vars(args)

        if args.use_supconloss:
            assert args.use_head == True
        if args.rbfc_opt2:
            assert args.no_rbfc == False

        #model = MYNET(args, mode=args.base_mode)
        # model = SupConResNet(name=opt.model)
        model = MYNET(args, fw_mode=args.fw_mode)
        # model = MYNET(args, fw_mode=args.fw_mode)
        # model = SupConResNet(name=opt.model, head='linear')
        criterion = SupConLoss(temperature=args.supcontemp, supcon_angle=args.supcon_angle)

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model.encoder = torch.nn.DataParallel(model.encoder, list(range(args.num_gpu)))
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True
        supcon_criterion = SupConLoss(temperature=args.supcontemp, supcon_angle=args.supcon_angle)
        kdloss = KDLoss(args.s)

        writer = SummaryWriter(os.path.join(args.log_path, args.project))
        # for tsne plotting
        if args.plot_tsne:
            n_components = 2
            perplexity = 30
            sns.set_style('darkgrid')
            sns.set_palette('muted')
            sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
            #draw_base_cls_num = 60
            draw_base_cls_num = 15
            draw_n_per_basecls = 100

        t_start_time = time.time()

        if args.load_dir is not None:
            assert args.model_name is not None
            print('Loading init parameters from: %s_%s' %(args.load_dir, args.model_name))
            model_dir = os.path.join(args.load_dir, args.model_name)
            best_model_dict = torch.load(model_dir)['params']
            ##### TBD to remove
            #"""
            #best_model_dict = torch.load(model_dir)['model']
            """
            new_state_dict = {}
            for k, v in best_model_dict.items():
                #k = k.replace("module.", "")
                kk = k.split('.')
                str_ = kk[1] + '.' + kk[0]
                for i in range(2,len(kk)):
                    str_ += ('.'+kk[i])
                new_state_dict[str_] = v
                #new_state_dict[k] = v
            best_model_dict = new_state_dict
            """
            msg = model.load_state_dict(best_model_dict, strict=False)
            print(msg)
            #"""
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            best_model_dict = deepcopy(model.state_dict())

        # init train statistics
        result_list = [args]
        # args.seesions: total session num. == len(self.tasks)
        natsa_ = []

        if args.angle_exp:
            l_inter_angles_base = [];
            l_angle_intra_mean_base = []; l_angle_intra_std_base = []
            l_angle_feat_fc_base = []; l_angle_feat_fc_base_std = []
            l_inter_angles_inc = []
            l_angle_intra_mean_inc = []; l_angle_intra_std_inc = []
            l_angle_feat_fc_inc = []; l_angle_feat_fc_inc_std = []
            l_angle_base_feats_new_clf = []
            l_angle_base_clfs_new_feat = []
            l_base_inc_fc_angle = []
            l_inc_inter_fc_angle = []
            l_angle_featmean_fc_base = []; l_angle_featmean_fc_inc = []

        # init train statistics
        result_list = [args]

        # args.seesions: total session num. == len(self.tasks)
        for session in range(args.start_session, args.sessions):
            procD['step'] = -1
            procD['epoch'] = -1
            procD['session'] += 1
            clsD = inc_maps(args, clsD, procD, procD['session'])

            if session == 0:
                train_set, trainloader, testloader = get_dataloader(args, procD, clsD, bookD)
            else:
                train_set, trainloader, testloader, new_testloader, prev_testloader, new_all_testloader, base_testloader\
                    = get_dataloader(args, procD, clsD, bookD)

            #model.load_state_dict(best_model_dict)
            ### TBD To be removed, or not.
            model.load_state_dict(best_model_dict, strict=False)

            if session == 0:  # load base class train img label
                if args.secondphase is False:
                    if args.angle_exp:
                        init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc, \
                            init_angle_feat_fc_std, init_angle_featmean_fc = \
                            base_angle_exp(args, trainloader, args.base_doubleaug, procD, clsD, model, testloader.dataset.transform)
                        te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std, te_init_angle_feat_fc, \
                            te_init_angle_feat_fc_std, te_init_angle_featmean_fc = \
                            base_angle_exp(args, testloader, False, procD, clsD, model)

                    print('new classes for this session:\n', np.unique(train_set.targets))
                    model, optimizer, scheduler = self.get_optimizer_base(args, model)

                    angles = []
                    for epoch in range(args.epochs_base):
                        procD['epoch'] += 1
                        start_time = time.time()
                        # train base sess

                        if args.angle_exp:
                            angle = get_intra_avg_angle_from_loader(args, trainloader, args.base_doubleaug, procD, clsD, model)
                            angles.append(angle)

                        tl, ta = self.base_train(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler, epoch, supcon_criterion)
                        # test model with all seen class
                        tsl, tsa = self.test(args, model, procD, clsD, testloader) ####
                        #tsl, tsa = self.test2(args, model, procD, clsD, trainloader, testloader)  ####

                        # save better model
                        if (tsa * 100) >= procD['trlog']['max_acc'][session]:
                            procD['trlog']['max_acc'][session] = float('%.3f' % (tsa * 100))
                            procD['trlog']['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=model.state_dict()), save_model_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                            best_model_dict = deepcopy(model.state_dict())
                            print('********A better model is found!!**********')
                            print('Saving model to :%s' % save_model_dir)
                        print('best epoch {}, best test acc={:.3f}'.format(procD['trlog']['max_acc_epoch'],
                                                                           procD['trlog']['max_acc'][session]))

                        procD['trlog']['train_loss'].append(tl)
                        procD['trlog']['train_acc'].append(ta)
                        procD['trlog']['test_loss'].append(tsl)
                        procD['trlog']['test_acc'].append(tsa)
                        lrc = scheduler.get_last_lr()[0]
                        result_list.append(
                            'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                        print('This epoch takes %d seconds' % (time.time() - start_time),
                              '\nstill need around %.2f mins to finish this session' % (
                                      (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                        writer.add_scalar('Session {0} - Loss/train'.format(session), tl, epoch)
                        writer.add_scalar('Session {0} - Acc/val_ncm'.format(session), tsa, epoch)
                        writer.add_scalar('Session {0} - Learning rate/train'.format(session), lrc, epoch)
                        scheduler.step()

                        if epoch ==args.epochs_base-1 or epoch % args.save_freq == 0:
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) \
                                                          + '_epo' + str(epoch) + '_acc.pth')
                            torch.save(dict(params=model.state_dict()), save_model_dir)


                    if args.plot_tsne:
                        base_tsne_idx = torch.arange(args.base_class)[torch.randperm(args.base_class)[:draw_base_cls_num]]
                        palette = np.array(sns.color_palette("hls", args.base_class))
                        data_, label_ = tot_datalist(args, trainloader, model, args.base_doubleaug, \
                                                     clsD['seen_unsort_map'], gpu=False)
                        data_, label_ = selec_datalist(args, data_, label_, base_tsne_idx, draw_n_per_basecls)
                        draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base train')

                        data_, label_ = tot_datalist(args, testloader, model, False, clsD['seen_unsort_map'], gpu=False)
                        data_, label_ = selec_datalist(args, data_, label_, base_tsne_idx, draw_n_per_basecls)
                        draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base test')

                    if args.angle_exp:
                        afterbase_inter_angles, afterbase_intra_angle_mean, afterbase_intra_angle_std, afterbase_angle_feat_fc, \
                            afterbase_angle_feat_fc_std, afterbase_angle_featmean_fc = \
                            base_angle_exp(args, trainloader, args.base_doubleaug, procD, clsD, model, testloader.dataset.transform)
                        te_afterbase_inter_angles, te_afterbase_intra_angle_mean, te_afterbase_intra_angle_std, \
                        te_afterbase_angle_feat_fc, te_afterbase_angle_feat_fc_std, te_afterbase_angle_featmean_fc = \
                            base_angle_exp(args, testloader, False, procD, clsD, model)

                    #fig, (ax1,ax2,ax3) = plt.subplots(1,3)
                    #ax1.plot(range(args.epochs_base), angles)
                    #ax2.plot(range(args.epochs_base), tas)
                    #ax3.plot(range(args.epochs_base), tsas)
                    #plt.show()
                    result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                        session, procD['trlog']['max_acc_epoch'], procD['trlog']['max_acc'][session], ))

                if not args.no_rbfc:
                    if args.plot_tsne:
                        base_tsne_idx = torch.arange(args.base_class)[torch.randperm(args.base_class)[:draw_base_cls_num]]
                        palette = np.array(sns.color_palette("hls", args.base_class))
                        data_, label_ = tot_datalist(args, trainloader, model, args.base_doubleaug, \
                                                     clsD['seen_unsort_map'], gpu=False)
                        data_, label_ = selec_datalist(args, data_, label_, base_tsne_idx, draw_n_per_basecls)
                        draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base train')

                        data_, label_ = tot_datalist(args, testloader, model, False, clsD['seen_unsort_map'], gpu=False)
                        data_, label_ = selec_datalist(args, data_, label_, base_tsne_idx, draw_n_per_basecls)
                        draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base test')

                    #model.load_state_dict(best_model_dict)
                    #model = self.replace_base_fc(args, model, clsD, train_set, testloader.dataset.transform)
                    model = self.replace_clf(args, model, procD, clsD, trainloader, testloader.dataset.transform, args.rbfc_opt2)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_base_rbfc_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    best_model_dict = deepcopy(model.state_dict())
                    torch.save(dict(params=model.state_dict()), best_model_dir)

                    if args.plot_tsne:
                        base_tsne_idx = torch.arange(args.base_class)[torch.randperm(args.base_class)[:draw_base_cls_num]]
                        palette = np.array(sns.color_palette("hls", args.base_class))
                        data_, label_ = tot_datalist(args, trainloader, model, args.base_doubleaug, \
                                                     clsD['seen_unsort_map'], gpu=False)
                        data_, label_ = selec_datalist(args, data_, label_, base_tsne_idx, draw_n_per_basecls)
                        draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base train')

                        data_, label_ = tot_datalist(args, testloader, model, False, clsD['seen_unsort_map'], gpu=False)
                        data_, label_ = selec_datalist(args, data_, label_, base_tsne_idx, draw_n_per_basecls)
                        draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base test')

                    #model.module.set_mode('fc_cos')
                    print('After all epochs, test')
                    tsl, tsa = self.test(args, model, procD, clsD, testloader)
                    result_list.append(
                        'rbfc, test_loss:%.5f,test_acc:%.5f' % (
                            tsl, tsa))

                    procD['trlog']['max_acc_base_rbfc'] = float('%.3f' % (tsa * 100))
                    #if (tsa * 100) >= procD['trlog']['max_acc'][session]:
                    print('The rbfc test acc of base session={:.3f}'.format(procD['trlog']['max_acc_base_rbfc']))
                    if args.angle_exp:
                        afterrbfc_inter_angles, afterrbfc_intra_angle_mean, afterrbfc_intra_angle_std, \
                            afterrbfc_angle_feat_fc, afterrbfc_angle_feat_fc_std, afterrbfc_angle_featmean_fc = \
                            base_angle_exp(args, trainloader, args.base_doubleaug, procD, clsD, model, testloader.dataset.transform)
                        te_afterrbfc_inter_angles, te_afterrbfc_intra_angle_mean, te_afterrbfc_intra_angle_std, \
                            te_afterrbfc_angle_feat_fc, te_afterrbfc_angle_feat_fc_std, te_afterrbfc_angle_featmean_fc = \
                            base_angle_exp(args, testloader, False, procD, clsD, model)
                        #print('aia:', afterrbfc_inter_angles)
                        #print('taia:',te_afterrbfc_inter_angles)
                    result_list.append('Session {}, Test rbfc ,\nbest test Acc {:.4f}\n'.format(
                        session, procD['trlog']['max_acc_base_rbfc'], ))



                if args.base_clf_ft:
                    model, optimizer, scheduler = self.get_optimizer_ft(args, model)
                    trainloader.dataset.transform = testloader.dataset.transform

                    for epoch in range(args.epochs_base_clf):
                        tl, ta = self.base_train_ft(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,epoch)
                        # test model with all seen class

                        tsl, tsa = self.test(args, model, procD, clsD, testloader)

                        lrc = scheduler.get_last_lr()[0]
                        result_list.append(
                            'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))

                        scheduler.step()
                        if epoch % 10 == 0:
                            result_list.append(
                            'Session {}, after f.t. epoch {}, test Acc {:.3f}\n'.format(session, epoch, tsa))


                    save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_afterft_acc.pth')
                    torch.save(dict(params=model.state_dict()), save_model_dir)
                    best_model_dict = deepcopy(model.state_dict())
                    print('Saving model to :%s' % save_model_dir)
                    print('test acc={:.3f}'.format(tsa))

                    procD['trlog']['max_acc_base_after_ft'] = float('%.3f' % (tsa * 100))
                    # if (tsa * 100) >= procD['trlog']['max_acc'][session]:
                    print('The after_ft test acc of base session={:.3f}'.format(tsa))
                    result_list.append('Session {}, Test after_ft ,\n test Acc {:.4f}\n'.format(
                        session, tsa))

                if args.gauss:
                    gaussD['base_mean'], gaussD['base_cov'] = \
                        learn_gauss(args, trainloader, model, clsD, procD)


            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                model, optimizer, scheduler = self.get_optimizer_new(args, model)
                transform_ = trainloader.dataset.transform
                trainloader.dataset.transform = testloader.dataset.transform
                model = self.replace_clf(args, model, procD, clsD, trainloader, testloader.dataset.transform, args.rbfc_opt2)
                trainloader.dataset.transform = transform_




                for epoch in range(args.epochs_new):
                    procD['epoch'] += 1
                    start_time = time.time()
                    # train base sess

                    tl, ta = self.new_train(args, model, procD, clsD, gaussD, trainloader, optimizer, kdloss, scheduler, epoch)
                    # test model with all seen class
                    tsl, tsa = self.test(args, model, procD, clsD, testloader)

                    # save better model
                    """
                    if (tsa * 100) >= procD['trlog']['max_acc'][session]:
                        procD['trlog']['max_acc'][session] = float('%.3f' % (tsa * 100))
                        procD['trlog']['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        best_model_dict = deepcopy(model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                     """
                    print('epoch {}, test acc={:.3f}'.format(epoch, tsa))
                    lrc = scheduler.get_last_lr()[0]
                    if epoch % 10 == 0:
                        result_list.append(
                            'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                    scheduler.step()

                # model.module.set_mode(args.new_mode)
                model.eval()

                print('Incremental session, test')
                tsl, tsa = self.test(args, model, procD, clsD, testloader)
                ntsl, ntsa = self.test(args, model, procD, clsD, new_testloader)
                ptsl, ptsa = self.test(args, model, procD, clsD, prev_testloader)

                _, btsa = self.test(args, model, procD, clsD, base_testloader)
                _, natsa = self.test(args, model, procD, clsD, new_all_testloader)

                prev_new_clf_ratio = self.newtest(args, model, procD, clsD, prev_testloader)
                new_new_clf_ratio = self.newtest(args, model, procD, clsD, new_testloader)

                # save model
                procD['trlog']['max_acc'][session] = float('%.3f' % (tsa * 100))
                procD['trlog']['new_max_acc'][session] = float('%.3f' % (ntsa * 100))
                procD['trlog']['new_all_max_acc'][session] = float('%.3f' % (natsa * 100))
                procD['trlog']['base_max_acc'][session] = float('%.3f' % (btsa * 100))
                procD['trlog']['prev_max_acc'][session] = float('%.3f' % (ptsa * 100))
                procD['trlog']['prev_new_clf_ratio'][session] = float('%.3f' % (prev_new_clf_ratio * 100))
                procD['trlog']['new_new_clf_ratio'][session] = float('%.3f' % (new_new_clf_ratio * 100))

                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=model.state_dict()), save_model_dir)
                best_model_dict = deepcopy(model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(procD['trlog']['max_acc'][session]))

                result_list.append(
                    'Session {}, test Acc {:.3f}\n'.format(session, procD['trlog']['max_acc'][session]))


                if args.plot_tsne:
                    #if session !=  args.sessions -1:
                    #    continue
                    base_tsne_idx = torch.arange(args.base_class)[torch.randperm(args.base_class)[:draw_base_cls_num]]
                    palette = np.array(sns.color_palette("hls", args.base_class + args.way * session))

                    data_base_, label_base_ = tot_datalist(args, base_testloader, model, False, clsD['seen_unsort_map'], gpu=False)
                    data_base_, label_base_ = selec_datalist(args, data_base_, label_base_, base_tsne_idx, draw_n_per_basecls)
                    #draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base test')

                    new_tsne_idx = torch.arange(args.way * session) + args.base_class
                    new_tsne_idx = new_tsne_idx[-20:]
                    data_new_, label_new_ = tot_datalist(args, new_all_testloader, model, False, clsD['seen_unsort_map'], gpu=False)
                    data_new_, label_new_ = selec_datalist(args, data_new_, label_new_, new_tsne_idx, draw_n_per_basecls)

                    data_ = torch.cat((data_base_, data_new_), dim=0)
                    label_ = torch.cat((label_base_, label_new_), dim=0)
                    combine_tsne_idx = torch.cat((base_tsne_idx, new_tsne_idx), dim=0)
                    #draw_tsne(data_, label_, n_components, perplexity, palette, combine_tsne_idx, 'new test')
                    palette2 = np.array(sns.color_palette("hls", args.way * session))
                    lll = label_new_ - args.base_class
                    iii = new_tsne_idx - args.base_class
                    #draw_tsne(data_new_, label_new_, n_components, perplexity, palette2, new_tsne_idx, 'new test')
                    draw_tsne(data_new_, lll, n_components, perplexity, palette2, iii, 'new test')


                #if session == 1:
                if args.angle_exp:
                    inter_angles_base, angle_intra_mean_base, angle_intra_std_base, angle_feat_fc_base, angle_feat_fc_base_std,\
                    inter_angles_inc, angle_intra_mean_inc, angle_intra_std_inc, angle_feat_fc_inc, angle_feat_fc_inc_std,\
                    angle_base_feats_new_clf, angle_base_clfs_new_feat, base_inc_fc_angle, inc_inter_fc_angle, \
                    angle_featmean_fc_base, angle_featmean_fc_inc \
                    = inc_angle_exp(args, base_testloader, new_testloader, args.base_doubleaug, procD, clsD, model)

                    l_inter_angles_base.append(inter_angles_base)
                    l_angle_intra_mean_base.append(angle_intra_mean_base)
                    l_angle_intra_std_base.append(angle_intra_std_base)
                    l_angle_feat_fc_base.append(angle_feat_fc_base)
                    l_angle_feat_fc_base_std.append(angle_feat_fc_base_std)
                    l_inter_angles_inc.append(inter_angles_inc)
                    l_angle_intra_mean_inc.append(angle_intra_mean_inc)
                    l_angle_intra_std_inc.append(angle_intra_std_inc)
                    l_angle_feat_fc_inc.append(angle_feat_fc_inc)
                    l_angle_feat_fc_inc_std.append(angle_feat_fc_inc_std)
                    l_angle_base_feats_new_clf.append(angle_base_feats_new_clf)
                    l_angle_base_clfs_new_feat.append(angle_base_clfs_new_feat)
                    l_base_inc_fc_angle.append(base_inc_fc_angle)
                    l_inc_inter_fc_angle.append(inc_inter_fc_angle)
                    l_angle_featmean_fc_base.append(angle_featmean_fc_base)
                    l_angle_featmean_fc_inc.append(angle_featmean_fc_inc)

                    #print(inter_angles_base, angle_intra_std_base, angle_feat_fc_base, inter_angles_inc, angle_intra_std_inc, \
                    #angle_feat_fc_inc, angle_base_feats_new_clf, angle_base_clfs_new_feat, base_inc_fc_angle, inc_inter_fc_angle)

        result_list.append('Base Session Best Epoch {}\n'.format(procD['trlog']['max_acc_epoch']))
        result_list.append(procD['trlog']['max_acc'])
        print('max_acc:', procD['trlog']['max_acc'])
        print('max_acc_base_rbfc:', procD['trlog']['max_acc_base_rbfc'])
        print('max_acc_base_after_ft:', procD['trlog']['max_acc_base_after_ft'])
        print('new_max_acc:', procD['trlog']['new_max_acc'])
        print('new_all_max_acc:', procD['trlog']['new_all_max_acc'])
        print('base_max_acc:', procD['trlog']['base_max_acc'])
        print('prev_max_acc:', procD['trlog']['prev_max_acc'])
        print('prev_new_clf_ratio:', procD['trlog']['prev_new_clf_ratio'])
        print('new_new_clf_ratio:', procD['trlog']['new_new_clf_ratio'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
        save_obj(args.save_path, procD, clsD, bookD)

        print('Base Session Best epoch:', procD['trlog']['max_acc_epoch'])

        if args.angle_exp:
            print('base exp result')
            print(init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc, init_angle_feat_fc_std, \
                  init_angle_featmean_fc)
            print(te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std, te_init_angle_feat_fc,\
                  te_init_angle_feat_fc_std, te_init_angle_featmean_fc)
            print(afterbase_inter_angles, afterbase_intra_angle_mean, afterbase_intra_angle_std,\
                  afterbase_angle_feat_fc, afterbase_angle_feat_fc_std, afterbase_angle_featmean_fc)
            print(te_afterbase_inter_angles, te_afterbase_intra_angle_mean, te_afterbase_intra_angle_std, \
                  te_afterbase_angle_feat_fc, te_afterbase_angle_feat_fc_std, te_afterbase_angle_featmean_fc)
            print(afterrbfc_inter_angles, afterrbfc_intra_angle_mean, afterrbfc_intra_angle_std, \
                  afterrbfc_angle_feat_fc, afterrbfc_angle_feat_fc_std, afterrbfc_angle_featmean_fc)
            print(te_afterrbfc_inter_angles, te_afterrbfc_intra_angle_mean, te_afterrbfc_intra_angle_std, \
                  te_afterrbfc_angle_feat_fc, te_afterrbfc_angle_feat_fc_std, te_afterrbfc_angle_featmean_fc)
            print('inc exp result')
            print(l_inter_angles_base)
            print(l_angle_intra_mean_base)
            print(l_angle_intra_std_base)
            print(l_angle_feat_fc_base)
            print(l_angle_feat_fc_base_std)
            print(l_inter_angles_inc)
            print(l_angle_intra_mean_inc)
            print(l_angle_intra_std_inc)
            print(l_angle_feat_fc_inc)
            print(l_angle_feat_fc_inc_std)
            print(l_angle_base_feats_new_clf)
            print(l_angle_base_clfs_new_feat)
            print(l_base_inc_fc_angle)
            print(l_inc_inter_fc_angle)
            print(l_angle_featmean_fc_base)
            print(l_angle_featmean_fc_inc)
            print('angles')
            print(angles)


    def replace_clf(self, args, model, procD, clsD, trainloader, transform, opt2=False):
        # replace fc.weight with the embedding average of train data
        session = procD['session']
        model.eval()

        loader = deepcopy(trainloader)
        loader.dataset.transform = transform
        if session == 0:
            embedding_list, label_list = tot_datalist(args, loader, model, doubleaug=args.base_doubleaug, map=None,
                                                      gpu=False)
        else:
            embedding_list, label_list = tot_datalist(args, loader, model, doubleaug=args.inc_doubleaug, map=None,
        #                                          map=clsD['class_maps'][session], gpu=False)
                                                   gpu=False)
        label_list = label_list.cuda()
        mean_list = []

        for class_index in clsD['tasks'][session]:
        #for class_index in range(len(model.module.angle_w[session].data)):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            if args.rpclf_normmean:
                embedding_this = F.normalize(embedding_this).mean(0)
            else:
                embedding_this = embedding_this.mean(0)
            mean_list.append(embedding_this)

        mean_list = torch.stack(mean_list, dim=0)

        bsc_ = args.base_class
        way_ = args.way
        if 'fc' in args.fw_mode:
            #model.module.fc.weight.data[clsD['tasks'][session]] = proto_list.cuda()
            if session == 0:
                model.fc.weight.data[:bsc_] = mean_list.cuda()
            else:
                model.fc.weight.data[bsc_+ way_*(session-1) : bsc_+ way_ * session] = mean_list.cuda()
        else:
            model.angle_w[session].data = mean_list.cuda()

        return model
