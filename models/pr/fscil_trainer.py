import os.path as osp
import torch.nn as nn
from copy import deepcopy
import seaborn as sns
import abc
import torch.nn.functional as F
import pdb
import torch.backends.cudnn as cudnn
import torchvision
import cv2

from utils_s import *
from dataloader.data_utils_s import *
from tensorboardX import SummaryWriter
from .Network import *
from tqdm import tqdm
from SupConLoss import SupConLoss
#from src.datasets.common import get_dataloader, maybe_dictionarize
#from src.datasets.registry import get_dataset
from src.utils import *
from english_words import english_words_lower_set
from clip_iu.loss import ClipLoss
from src.datasets.templates import get_templates
from yacs.config import CfgNode as CN
from PIL import Image

from captum.attr import visualization
import clip_iu.clip as clip
from clip_iu.simple_tokenizer import SimpleTokenizer as _Tokenizer



class FSCILTrainer(object, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        pass

    def init_vars(self, args):
        # Setting arguments
        # if args.start_session == 0:
        cfg = CN()
        cfg.MODEL = CN()
        cfg.TRAINER = CN()
        cfg.DATASET = CN()
        cfg.OPTIM = CN()

        cfg.OPTIM.WARMUP_EPOCH = 1
        cfg.OPTIM.WARMUP_CONS_LR = 1e-5

        cfg.TRAINER.COCOOP = CN()
        cfg.TRAINER.COCOOP.N_CTX = 4  # number of context vectors
        cfg.TRAINER.COCOOP.CTX_INIT = "a photo of a"  # initialization words
        cfg.TRAINER.COCOOP.PREC = "fp32"  # fp16, fp32, amp
        cfg.TRAINER.NAME = 'CoCoOp'

        cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

        args.cfg = cfg




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

        if args.use_coreset:
            args.coreset = {}



        # Setting save path
        # mode = args.base_mode + '-' + args.new_mode
        mode = args.fw_mode

        bfzb = 'T' if args.base_freeze_backbone else 'F'
        ifzb = 'T' if args.inc_freeze_backbone else 'F'

        bdg = 'T' if args.base_doubleaug else 'F'
        idg = 'T' if args.inc_doubleaug else 'F'
        _SP = 'T' if args.use_single_prompt else 'F'

        ft_type = 'ce'

        if args.schedule == 'Milestone':
            mile_stone = str(args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            schedule = 'MS_%s' % (mile_stone)
        elif args.schedule == 'Step':
            schedule = 'Step_%d' % (args.step)
        elif args.schedule == 'Cosine':
            schedule = 'schecos'
        elif args.schedule == 'Custom_Cosine':
            schedule = 'cc'

        if args.schedule_new == 'Milestone':
            mile_stone_new = str(args.milestones_new).replace(" ", "").replace(',', '_')[1:-1]
            schedule_new = 'MS_%s' % (mile_stone_new)
        elif args.schedule_new == 'Step':
            raise NotImplementedError
            # schedule = 'Step_%d'%(args.step)
        elif args.schedule_new == 'Cosine':
            schedule_new = 'sch_c'
        elif args.schedule_new == 'Custom_Cosine':
            schedule_new = 'cc'

        if args.batch_size_base > 256:
            args.warm = True
        else:
            args.warm = False
        if args.warm:
            # args.model_name = '{}_warm'.format(opt.model_name)
            args.warmup_from = 0.01
            args.warm_epochs = 10
            if args.schedule == 'Cosine':
                eta_min = args.lr_base * (args.gamma ** 3)
                args.warmup_to = eta_min + (args.lr_base - eta_min) * (
                        1 + math.cos(math.pi * args.warm_epochs / args.epochs_base)) / 2
            else:
                args.warmup_to = args.lr_base

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        # sch_c: shcedule cosine
        # bdm: base_dataloader_mode
        # l: lambda
        # sd : seed
        # Epf, Lrf: epoch, lr for fine-tune
        # RN == resnet, SCRN == Supconresnet
        # 2P == 2ndphase, st == start
        # hd == head_type, D == head_dim
        # CRN: CIFAR_RESNET
        # cc: custom cosine
        # CR: use_coreset

        str_loss = '-'
        if args.use_celoss:
            str_loss += 'ce'
            if args.fw_mode == 'fc_cosface':
                str_loss += '_s%.1fm%.1f-' % (args.s, args.m)



        """
        auglist1_ = ''.join(str(x) for x in args.aug_type1)
        auglist2_ = ''.join(str(x) for x in args.aug_type2)
        str_loss += 'aug_%s,%s-' % (auglist1_, auglist2_)
        """

        if args.use_coreset:
            str_loss += 'CR-'


        hyper_name_list = 'Model_%s-Epob_%d-Epon_%d-Lrb_%.5f-Lrn_%.5f-%s-%s-Gam_%.2f-Dec_%.5f-wd_%.1f-ls_%.1f-Bs_%d-Mom_%.2f' \
                          'bsc_%d-way_%d-shot_%d-bfzb_%s-ifzb_%s-bdg_%s-idg_%s-bdm_%s' \
                          '-prm_%s-prv_%s-nprb_%d-npri_%d-kdim_%d-topk_%d-sd_%d-SP_%s' % (
                              args.model_type, args.epochs_base, args.epochs_new, args.lr_base, args.lr_new, schedule, schedule_new, \
                              args.gamma, args.decay, args.wd, args.ls,
                              args.batch_size_base, args.momentum, args.base_class, args.way, args.shot,
                              bfzb, ifzb, bdg, idg, args.base_dataloader_mode,
                              args.prompt_mode, args.clspr_ver, args.n_prpool_base, args.n_prpool_inc,
                              args.key_dim, args.topk_pool, args.seed, _SP)

        hyper_name_list += str_loss
        if args.secondphase:
            hyper_name_list = '2nd%d-'%(args.start_session) + hyper_name_list

        # if args.warm:
        #    hyper_name_list += '-warm'

        save_path = '%s/' % args.dataset
        save_path = save_path + '%s/' % args.project

        final_save_path = save_path + hyper_name_list
        final_save_path = os.path.join('checkpoint', final_save_path)
        args.final_save_path = final_save_path
        ensure_path(final_save_path)



        # Setting dictionaries
        # clsD initialize
        clsD = {}

        # clsD, procD
        if args.secondphase == False:
            args.task_class_order = np.random.permutation(np.arange(args.num_classes)).tolist()  #######
            # args.task_class_order = np.arange(args.num_classes).tolist()  #######
            clsD['tasks'], clsD['class_maps'] = init_maps(args)
            # self.args.task_class_order = (np.arange(self.args.num_classes)).tolist()

            procD = {}
            # varD['proc_book'] = {}

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
            procD['trlog']['new_max_acc'] = [0.0] * args.sessions  # first will be left 0.0
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
                procD['trlog']['max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['new_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['new_all_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['base_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['prev_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['prev_new_clf_ratio'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['new_new_clf_ratio'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
            else:
                procD['trlog']['new_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['new_all_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['base_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)
                procD['trlog']['prev_max_acc'][args.start_session:] = [0.0] * (args.sessions - args.start_session)

            tasks = bookD[0]['tasks']  # Note that bookD[i]['tasks'] is same for each i
            class_order_ = []
            for i in range(len(tasks)):
                class_order_ += tasks[i].tolist()
            # clsD to up-to-date
            args.task_class_order = class_order_
            clsD['tasks'], clsD['class_maps'] = init_maps(args)
            # for i in range(args.start_session):

            if args.start_session == 0 or args.start_session == 1:
                pass
            else:
                clsD = inc_maps(args, clsD, procD, args.start_session - 1)
        return args, procD, clsD, bookD


    def get_optimizer_base(self, args, model, num_batches):

        if args.base_freeze_backbone:

            for param in model.module.clip_encoder.parameters():
                param.requires_grad = False

        name_to_update1 = "prompt_learner"
        #name_to_update2 = "cra"
        #name_to_update3 = "msa"
        for name, param in model.module.named_parameters():
            #if (name_to_update1 not in name) and (name_to_update2 not in name) and (name_to_update3 not in name):
            if (name_to_update1 not in name):
                param.requires_grad_(False)
            else:
                if ('keys' in name) or ('key_prompts' in name):
                    if ('keys_0' in name) or ('key_prompts_0' in name):
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                elif 'clswise_pr_embed' in name:
                    if 'clswise_pr_embed_0' in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
            if param.requires_grad:
                print(name)

        # Double check
        enabled = set()
        for name, param in model.module.named_parameters():
            if param.requires_grad:
                enabled.add(name)


        print(f"Parameters to be updated: {enabled}")

        params = [p for p in model.module.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr_base, momentum=args.momentum, weight_decay=args.decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs_base)
        scheduler = ConstantWarmupScheduler(optimizer, scheduler, args.cfg.OPTIM.WARMUP_EPOCH,args.cfg.OPTIM.WARMUP_CONS_LR)
        """
        if args.base_freeze_backbone:
            assert args.use_encmlp
            optimizer = torch.optim.AdamW(params, lr=args.lr_encmlp, weight_decay=args.wd)
        else:
            if not args.use_encmlp:
                optimizer = torch.optim.AdamW(params, lr=args.lr_base, weight_decay=args.wd)
            else:
                optimizer = torch.optim.AdamW([{'params': model.module.clip_encoder.parameters(),
                                                'lr': args.lr_base},
                                         {'params': model.module.encmlp.parameters(), 'lr': args.lr_encmlp}],
                                         weight_decay=args.wd)

        if args.schedule == 'Custom_Cosine':
            scheduler = cosine_lr(optimizer, args.lr_base, args.warmup_length, args.epochs_base * num_batches)
        elif args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        elif args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                             gamma=args.gamma)
        elif args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_base)
        else:
            raise NotImplementedError

        return model, optimizer, scheduler
        """

        return model, optimizer, scheduler


    def get_optimizer_new(self, args, model, num_batches, sess):
        # assert self.args.angle_mode is not None
        if args.inc_freeze_backbone:
            for param in model.module.clip_encoder.parameters():
                param.requires_grad = False

        name_to_update1 = "prompt_learner"
        #name_to_update2 = "cra"
        #name_to_update3 = "msa"
        ban_to_update1 = "merge_net"
        ban_to_update2 = 'lin_net'
        for name, param in model.module.named_parameters():
            #if (name_to_update1 not in name) and (name_to_update2 not in name) and (name_to_update3 not in name):
            if (name_to_update1 not in name):
                param.requires_grad_(False)
            else:
                if ('keys' in name) or ('key_prompts' in name):
                    if (('keys_'+str(sess)) in name) or (('key_prompts_'+str(sess)) in name):
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                elif 'clswise_pr_embed' in name:
                    if ('clswise_pr_embed_'+str(sess)) in name:
                        param.requires_grad_(True)
                    else:
                        param.requires_grad_(False)
                else:
                    param.requires_grad_(True)

            if (ban_to_update1 in name) or (ban_to_update2 in name):
                param.requires_grad_(False)
            if param.requires_grad:
                print(name)

        # Double check
        enabled = set()
        for name, param in model.module.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        params = [p for p in model.module.parameters() if p.requires_grad]
        #optimizer = torch.optim.SGD(params, lr=args.lr_base, momentum=args.momentum, weight_decay=args.decay)
        #optimizer = torch.optim.AdamW(params, lr=args.lr_new, weight_decay=args.wd)
        optimizer = torch.optim.SGD(params, lr=args.lr_new, momentum=args.momentum, weight_decay=args.wd)


        if args.schedule_new == 'Custom_Cosine':
            scheduler = cosine_lr(optimizer, args.lr_new, args.warmup_length, args.epochs_new * num_batches)
        elif args.schedule_new == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        elif args.schedule_new == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                             gamma=args.gamma)
        elif args.schedule_new == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_new)

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs_new)
        #scheduler = ConstantWarmupScheduler(optimizer, scheduler, args.cfg.OPTIM.WARMUP_EPOCH,args.cfg.OPTIM.WARMUP_CONS_LR)
        return model, optimizer, scheduler



    def base_train(self, args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler, epoch,
                   loss_fn, supcon_criterion=None):

        tl = Averager()
        ta = Averager()
        # self.model = self.model.train()
        model.train()
        num_batches = len(trainloader)

        # standard classification for pretrain
        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):

            step = i + epoch * num_batches
            #if args.schedule=='Custom_Cosine':
            #    scheduler(step)
            procD['step'] += 1

            if args.base_doubleaug is False:
                data, train_label = [_.cuda() for _ in batch]
                target_cls = clsD['class_maps'][0][train_label]
            else:
                # data = batch[0][0].cuda()
                # train_label = batch[1].cuda()
                data = torch.cat((batch[0][0], batch[0][1]), dim=0).cuda()
                # train_label = batch[1].repeat(2).cuda()
                train_label = batch[1].cuda()
                train_label = train_label.repeat(2)
                target_cls = clsD['class_maps'][0][train_label]

            inputs = data
            labels = target_cls
            """
            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time
            """
            if not args.prompt_mode=='clspr':
                logits = model(inputs, sess=procD['session'], train=True)
            else:
                logits = model(inputs, sess=procD['session'], train=True, labels=labels)
            loss = loss_fn(logits, labels)



            params = [p for p in model.module.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            if i % args.print_every == 0:
                percent_complete = 100 * i / len(trainloader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(trainloader)}]\t"
                    f"Loss: {loss.item():.6f}", flush=True
                )

            total_loss = loss


            tl.add(total_loss.item())

            optimizer.zero_grad()
            # loss.backward()
            total_loss.backward()
            optimizer.step()

            acc = count_acc(logits, labels)
            tqdm_gen.set_description(
                'Session 0, epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, total_loss.item(), acc))
            ta.add(acc)

        tl = tl.item()
        ta = ta.item()
        return tl, ta



    def new_train(self, args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler, epoch,
                   loss_fn, supcon_criterion=None):

        tl = Averager()
        ta = Averager()
        # self.model = self.model.train()
        model.train()
        num_batches = len(trainloader)

        # standard classification for pretrain
        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):
            step = i + epoch * num_batches

            if args.inc_doubleaug is False:
                data, train_label = [_.cuda() for _ in batch]
                #target_cls = clsD['class_maps'][procD['session']][train_label]
                if not args.use_coreset:
                    target_cls = clsD['seen_unsort_map'][train_label]
                else:
                    target_cls = train_label
            else:
                # data = batch[0][0].cuda()
                # train_label = batch[1].cuda()
                data = torch.cat((batch[0][0], batch[0][1]), dim=0).cuda()
                # train_label = batch[1].repeat(2).cuda()
                train_label = batch[1].cuda()
                train_label = train_label.repeat(2)
                #target_cls = clsD['class_maps'][procD['session']][train_label]
                target_cls = clsD['seen_unsort_map'][train_label]

            inputs = data
            labels = target_cls

            if not args.prompt_mode == 'clspr':
                print(inputs.shape)
                if not args.use_coreset:
                    logits = model(inputs, sess=procD['session'], train=True)
                else:
                    logits = model(inputs, sess=procD['session'], train=True, encode=False)
            else:
                print(inputs.shape)
                if not args.use_coreset:
                    logits = model(inputs, sess=procD['session'], train=True, labels=labels)
                else:
                    logits = model(inputs, sess=procD['session'], train=True, encode=False, labels=labels)
            loss = loss_fn(logits, labels)

            params = [p for p in model.module.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params, 1.0)

            total_loss = loss

            tl.add(total_loss.item())

            optimizer.zero_grad()
            # loss.backward()
            total_loss.backward()
            optimizer.step()

            acc = count_acc(logits, labels)
            tqdm_gen.set_description(
                'Session {}, epo {}, total loss={:.4f} acc={:.4f}'.format(procD['session'], epoch, total_loss.item(), acc))
            ta.add(acc)

        tl = tl.item()
        ta = ta.item()
        return tl, ta



    def test(self, args, model, procD, clsD, testloader):
        epoch = procD['epoch']
        session = procD['session']

        model = model.cuda()
        model.eval()
        vl = Averager()
        va = Averager()

        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = [_.cuda() for _ in batch]
                #logits = get_logits(data, model)
                logits = model(data, procD['session'])


                if session == 0:
                    # logits_cls = logits[:, clsD['tasks'][0]]
                    logits_cls = logits
                    target_cls = clsD['class_maps'][0][test_label]
                    # tasks 대신 seen_unsort / seen_unsort_map으로 바꾼 후 아래랑 합쳐도 무방.
                    loss = F.cross_entropy(logits_cls, target_cls)
                    acc = count_acc(logits_cls, target_cls)
                else:
                    # logits = logits[:, clsD['seen_unsort']]
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



    def main(self, args):
        timer = Timer()
        args, procD, clsD, bookD = self.init_vars(args)


        print(args.task_class_order)

        # model = MYNET(args, mode=args.base_mode)
        # model = SupConResNet(name=opt.model)

        if args.dataset == 'cifar100':
            tmp_trainset = args.Dataset.CIFAR100(args, train=True, shotpercls=args.shotpercls, base_sess=True,
                                                 root=args.dataroot, doubleaug=args.base_doubleaug, download=True,
                                                 index=np.array(args.task_class_order))
            args.dataset_label2txt = {v: k for k, v in tmp_trainset.class_to_idx.items()}
        elif args.dataset == 'mini_imagenet':
            tmp_trainset = args.Dataset.MiniImageNet(args, train=True, shotpercls=args.shotpercls, base_sess=True,
                                                     root=args.dataroot, doubleaug=args.base_doubleaug,
                                                     index=np.array(args.task_class_order))
            mini_dic_loc = 'dataloader/miniimagenet/imagenet_label_textdic'
            with open(mini_dic_loc, 'rb') as f:
                mini_dic = pickle.load(f)
            args.dataset_label2txt = {}
            for i in range(args.num_classes):
                args.dataset_label2txt[i] = mini_dic[tmp_trainset.wnids[i]]
        elif args.dataset == 'cub200':
            tmp_trainset = args.Dataset.CUB200(args, train=True, shotpercls=args.shotpercls, base_sess=True,
                                               root=args.dataroot, doubleaug=args.base_doubleaug,
                                               index=np.array(args.task_class_order))
            mini_dic = {}
            for key, value in tmp_trainset.id2image.items():
                _str = value.split('/')[0]
                strs = _str.split('.')
                mini_dic[int(strs[0])-1] = strs[1]
            args.dataset_label2txt = mini_dic
        else:
            raise NotImplementedError

        _classnames = [args.dataset_label2txt[i] for i in range(args.num_classes)]
        #args.classnames = _classnames[args.task_class_order]
        args.classnames = [_classnames[i] for i in args.task_class_order]

        zeroshot_clipmodel = CLIP_Model(args, keep_lang=True)




        model = MYNET(args, fw_mode=args.fw_mode)

        # model = MYNET(args, fw_mode=args.fw_mode)
        # model = SupConResNet(name=opt.model, head='linear')
        criterion = SupConLoss(temperature=args.supcontemp, supcon_angle=args.supcon_angle)

        model = nn.DataParallel(model, list(range(args.num_gpu)))
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        model.train()

        if args.use_gradcam:
            print('gradcam')
            #img_pth = '../ILFR/data/miniimagenet/images/n0153282900001210.jpg'
            img_pth = '../ILFR/data/miniimagenet/images/' + args.img_name
            #img_pth = './glasses.png'
            if 'RN' in args.model_type:
                gradcam_transform = transforms.Compose([
                    transforms.Resize([92, 92]),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
                text_fc = model.module.get_text_classifiers(gradcam_transform(Image.open(img_pth)).cuda(), sess=0, train=True)
                #text_fc = model.module.get_text_classifiers(gradcam_transform(Image.open(img_pth)).cuda(), sess=2,
                                                            #train=True)
                save_gradcam(img_pth, model.module.image_encoder, text_fc, args.classnames, 'hello.png', model.module.image_encoder.layer4[-1])
            else:
                _clip_model = model.module.clip_model
                for i in range(len(_clip_model.visual.transformer.resblocks)):
                    _clip_model.visual.transformer.resblocks[i].heatmap = True
                for i in range(len(_clip_model.transformer.resblocks)):
                    _clip_model.transformer.resblocks[i].heatmap = True

                img = Image.open(img_pth).convert("RGB")
                preprocess = model.module.val_preprocess

                images = []
                images.append(preprocess(img))
                img_pp = preprocess.transforms[1](preprocess.transforms[0](img))

                images = []
                #pth = '../../n0153282900001268.jpg'
                image = Image.open(img_pth).convert("RGB")
                images.append(preprocess(image))

                #plt.tight_layout()
                #plt.show()

                start_layer = -1  # @param {type:"number"}

                # @title Number of layers for text Transformer
                start_layer_text = -1  # @param {type:"number"}

                _tokenizer = _Tokenizer()

                device = "cuda" if torch.cuda.is_available() else "cpu"

                class color:
                    PURPLE = '\033[95m'
                    CYAN = '\033[96m'
                    DARKCYAN = '\033[36m'
                    BLUE = '\033[94m'
                    GREEN = '\033[92m'
                    YELLOW = '\033[93m'
                    RED = '\033[91m'
                    BOLD = '\033[1m'
                    UNDERLINE = '\033[4m'
                    END = '\033[0m'

                #img_path = "CLIP/glasses.png"
                #texts = ["a man with eyeglasses"]
                #texts = ["a photo of a bird"]
                #texts = ["a photo of a wing bird"]
                texts = [args.given_text]
                text = clip.tokenize(texts).to(device)

                start_layer = -1  # @param {type:"number"}
                # @title Number of layers for text Transformer
                start_layer_text = -1  # @param {type:"number"}
                R_text, R_image = interpret(model=_clip_model, image=preprocess(image).cuda(), texts=text, device=device, start_layer=start_layer,
                                            start_layer_text=start_layer_text)
                batch_size = text.shape[0]
                for i in range(batch_size):
                    show_heatmap_on_text(texts[i], text[i], R_text[i])
                    show_image_relevance(R_image[i], preprocess(img).unsqueeze(0).to(device), orig_image=Image.open(img_pth))
                    plt.show()

                for i in range(len(_clip_model.visual.transformer.resblocks)):
                    _clip_model.visual.transformer.resblocks[i].heatmap = False
                for i in range(len(_clip_model.transformer.resblocks)):
                    _clip_model.transformer.resblocks[i].heatmap = False




        if args.load_dir is not None:
            print('Loading init parameters from: %s_%s' % (args.load_dir, args.model_name))
            args.load_model = os.path.join(args.load_dir, args.model_name)
            load_model_weight = torch.load(args.load_model)['params']
            load_missing_keys = model.load_state_dict(load_model_weight, strict=False)
            print('load_missing_keys:', load_missing_keys)


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
            # draw_base_cls_num = 60
            draw_base_cls_num = 15
            draw_n_per_basecls = 100

        t_start_time = time.time()


        # init train statistics
        result_list = [args]
        # args.seesions: total session num. == len(self.tasks)
        natsa_ = []

        if args.angle_exp:
            l_inter_angles_base = [];
            l_angle_intra_mean_base = [];
            l_angle_intra_std_base = []
            l_angle_feat_fc_base = [];
            l_angle_feat_fc_base_std = []
            l_inter_angles_inc = []
            l_angle_intra_mean_inc = [];
            l_angle_intra_std_inc = []
            l_angle_feat_fc_inc = [];
            l_angle_feat_fc_inc_std = []
            l_angle_base_feats_new_clf = []
            l_angle_base_clfs_new_feat = []
            l_base_inc_fc_angle = []
            l_inc_inter_fc_angle = []
            l_angle_featmean_fc_base = [];
            l_angle_featmean_fc_inc = []

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
                trainloader.dataset.transform = model.module.train_preprocess
                trainloader.dataset.transform2 = model.module.train_preprocess
                testloader.dataset.transform = model.module.val_preprocess
            else:
                if not args.use_coreset:
                    train_set, trainloader, testloader, new_testloader, prev_testloader, new_all_testloader, base_testloader \
                        = get_dataloader(args, procD, clsD, bookD)
                else:
                    train_set_nocoreset, trainloader_nocoreset_noshuff, testloader, new_testloader, prev_testloader, new_all_testloader, base_testloader \
                        = get_dataloader(args, procD, clsD, bookD)
                    trainloader_nocoreset_noshuff.dataset.transform = model.module.train_preprocess
                    trainset_withcoreset =  dataset_withcoreset(args, trainloader_nocoreset_noshuff, args.coreset,
                                                torch.arange(args.base_class + args.way*(session-1)), model, clsD)
                    #trainloader_withcoreset = torch.utils.data.DataLoader(dataset=trainset_withcoreset,
                    #                          batch_size=args.batch_size_new, num_workers=args.num_workers, pin_memory=True)
                    trainloader_withcoreset = torch.utils.data.DataLoader(dataset=trainset_withcoreset,batch_size=args.batch_size_new, shuffle=True)
                    trainloader = trainloader_withcoreset
                """
                else:
                    train_set, trainloader_nocoreset, trainloader_withcoreset, testloader, new_testloader, prev_testloader, new_all_testloader, base_testloader \
                        = get_dataloader(args, procD, clsD, bookD, args.coreset, model)
                    trainloader_withcoreset.transform = model.module.train_preprocess
                    trainloader = trainloader_withcoreset
                """

                trainloader.dataset.transform = model.module.train_preprocess
                trainloader.dataset.transform2 = model.module.train_preprocess
                testloader.dataset.transform = model.module.val_preprocess
                new_testloader.dataset.transform = model.module.val_preprocess
                prev_testloader.dataset.transform = model.module.val_preprocess
                base_testloader.dataset.transform = model.module.val_preprocess
                new_all_testloader.dataset.transform = model.module.val_preprocess

            args.print_every = 100

            # Should erase this part. but how to use preprocess_fn on existing dataloader? train & test.
            num_batches = len(trainloader)

            loss_fn = torch.nn.CrossEntropyLoss()




            if session == 0:  # load base class train img label
                if not args.secondphase:
                    if args.angle_exp:
                        init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc, \
                        init_angle_feat_fc_std, init_angle_featmean_fc = \
                            base_angle_exp(args, trainloader, args.base_doubleaug, procD, clsD, model,
                                           testloader.dataset.transform)
                        te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std, te_init_angle_feat_fc, \
                        te_init_angle_feat_fc_std, te_init_angle_featmean_fc = \
                            base_angle_exp(args, testloader, False, procD, clsD, model)
                        print('sess 0 bef train')
                        print(init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc, \
                              init_angle_feat_fc_std, init_angle_featmean_fc)
                        print(te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std,
                              te_init_angle_feat_fc, \
                              te_init_angle_feat_fc_std, te_init_angle_featmean_fc)
                        print('text classifiers inter angle: %d'%(get_inter_angle(model.module.textual_classifier.weight)))

                    print('new classes for this session:\n', np.unique(train_set.targets))
                    model, optimizer, scheduler = self.get_optimizer_base(args, model, num_batches)

                    angles = []
                    for epoch in range(args.epochs_base):
                        procD['epoch'] += 1
                        start_time = time.time()
                        # train base sess

                        if args.angle_exp:
                            angle = get_intra_avg_angle_from_loader(args, trainloader, args.base_doubleaug, procD, clsD,
                                                                    model)
                            angles.append(angle)

                        #tl, ta = self.base_train(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                        #                         epoch, supcon_criterion)
                        tl, ta = self.base_train(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                                             epoch, loss_fn, supcon_criterion)
                        procD['trlog']['train_loss'].append(tl)
                        procD['trlog']['train_acc'].append(ta)
                        result_list.append(
                            'epoch:%03d,training_loss:%.5f,training_acc:%.5f' % (epoch, tl, ta))
                        writer.add_scalar('Session {0} - Loss/train'.format(session), tl, epoch)

                        """
                        # test model with all seen class
                        tsl, tsa = self.test(args, model, procD, clsD, testloader)  ####
                        procD['trlog']['test_loss'].append(tsl)
                        procD['trlog']['test_acc'].append(tsa)
                        result_list.append(
                            'epoch:%03d,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, tsl, tsa))
                        writer.add_scalar('Session {0} - Acc/val_ncm'.format(session), tsa, epoch)
                        """
                        #scheduler.step()

                        if epoch == args.epochs_base-1:
                            save_model_dir = os.path.join(args.final_save_path, 'session' + str(session) \
                                                          + '_epo' + str(epoch) + '_acc.pth')
                            torch.save(dict(params=model.state_dict()), save_model_dir)

                            save_obj(args.final_save_path, procD, clsD, bookD)
                            print('save path is %s'%(args.final_save_path))

                """
                # test model with all seen class
                tsl, tsa = self.test(args, model, procD, clsD, testloader)  ####
                #tsl, tsa = 0.0, 0.0
                #tsl, tsa = self.test2(args, model, procD, clsD, trainloader, testloader)  ####
                procD['trlog']['max_acc'][session] = float('%.3f' % (tsa * 100))
                procD['trlog']['test_loss'].append(tsl)
                procD['trlog']['test_acc'].append(tsa)
                result_list.append(
                    'test_loss:%.5f,test_acc:%.5f' % (tsl, tsa))
                writer.add_scalar('Session {0} - Acc/val_ncm'.format(session), tsa)
                """


                ####

                if args.use_gradcam:
                    print('gradcam')
                    #img_pth = '../ILFR/data/miniimagenet/images/n0153282900001210.jpg'
                    img_pth = '../ILFR/data/miniimagenet/images/'+args.img_name
                    #img_pth = './glasses.png'
                    _clip_model = model.module.clip_model
                    for i in range(len(_clip_model.visual.transformer.resblocks)):
                        _clip_model.visual.transformer.resblocks[i].heatmap = True
                    for i in range(len(_clip_model.transformer.resblocks)):
                        _clip_model.transformer.resblocks[i].heatmap = True

                    img = Image.open(img_pth).convert("RGB")
                    preprocess = model.module.val_preprocess

                    images = []
                    images.append(preprocess(img))
                    img_pp = preprocess.transforms[1](preprocess.transforms[0](img))

                    images = []
                    # pth = '../../n0153282900001268.jpg'
                    image = Image.open(img_pth).convert("RGB")
                    images.append(preprocess(image))

                    # plt.tight_layout()
                    # plt.show()

                    start_layer = -1  # @param {type:"number"}

                    # @title Number of layers for text Transformer
                    start_layer_text = -1  # @param {type:"number"}

                    _tokenizer = _Tokenizer()

                    device = "cuda" if torch.cuda.is_available() else "cpu"

                    class color:
                        PURPLE = '\033[95m'
                        CYAN = '\033[96m'
                        DARKCYAN = '\033[36m'
                        BLUE = '\033[94m'
                        GREEN = '\033[92m'
                        YELLOW = '\033[93m'
                        RED = '\033[91m'
                        BOLD = '\033[1m'
                        UNDERLINE = '\033[4m'
                        END = '\033[0m'

                    # img_path = "CLIP/glasses.png"
                    #texts = ["a man with eyeglasses"]
                    #texts = ["a photo of a bird"]
                    #text = clip.tokenize(texts).to(device)
                    gradcam_transform = transforms.Compose([
                        transforms.Resize([92, 92]),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
                    text_fc = model.module.get_text_classifiers(gradcam_transform(Image.open(img_pth)).cuda(), sess=session, train=True)
                    #text_ = text_fc[68].unsqueeze(0) # house bird number
                    text_ = text_fc[args.class_idx].unsqueeze(0)  # house bird number
                    start_layer = -1  # @param {type:"number"}
                    # @title Number of layers for text Transformer
                    start_layer_text = -1  # @param {type:"number"}
                    R_text, R_image = interpret(model=_clip_model, image=preprocess(image).cuda(), texts=text_,
                                                device=device, start_layer=start_layer,
                                                start_layer_text=start_layer_text, no_txt_encode=True)
                    batch_size = text.shape[0]
                    for i in range(batch_size):
                        show_heatmap_on_text(texts[i], text[i], R_text[i])
                        show_image_relevance(R_image[i], preprocess(img).unsqueeze(0).to(device),
                                             orig_image=Image.open(img_pth))
                        plt.show()

                    for i in range(len(_clip_model.visual.transformer.resblocks)):
                        _clip_model.visual.transformer.resblocks[i].heatmap = False
                    for i in range(len(_clip_model.transformer.resblocks)):
                        _clip_model.transformer.resblocks[i].heatmap = False

                ###







                if args.epochs_base == 0:
                    epoch = -1
                    if args.angle_exp:
                        init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc, \
                        init_angle_feat_fc_std, init_angle_featmean_fc = \
                            base_angle_exp(args, trainloader, args.base_doubleaug, procD, clsD, model,
                                           testloader.dataset.transform)
                        te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std, te_init_angle_feat_fc, \
                        te_init_angle_feat_fc_std, te_init_angle_featmean_fc = \
                            base_angle_exp(args, testloader, False, procD, clsD, model)
                        print('sess 0 epoc_base 0')
                        print(init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc, \
                              init_angle_feat_fc_std, init_angle_featmean_fc )
                        print(te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std, te_init_angle_feat_fc, \
                              te_init_angle_feat_fc_std, te_init_angle_featmean_fc)
                    """
                    # Duplicated so remove
                    tsl, tsa = self.test(args, model, procD, clsD, testloader)  ####
                    procD['trlog']['max_acc'][session] = float('%.3f' % (tsa * 100))
                    procD['trlog']['test_loss'].append(tsl)
                    procD['trlog']['test_acc'].append(tsa)
                    result_list.append(
                        'epoch:%03d, test_loss:%.5f,test_acc:%.5f' % (
                            epoch, tsl, tsa))
                    writer.add_scalar('Session {0} - Acc/val_ncm'.format(session), tsa, epoch)
                    writer.add_scalar('Session {0} - Learning rate/train'.format(session), epoch)
                    
                    """
                    # scheduler.step()

                    save_obj(args.final_save_path, procD, clsD, bookD)
                    save_model_dir = os.path.join(args.final_save_path, 'session' + str(session) \
                                                  + '_epob0_model' + str(epoch) + '_acc.pth')
                    torch.save(dict(params=model.state_dict()), save_model_dir)

                if args.plot_tsne:
                    base_tsne_idx = torch.arange(args.base_class)[
                        torch.randperm(args.base_class)[:draw_base_cls_num]]
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
                        base_angle_exp(args, trainloader, args.base_doubleaug, procD, clsD, model,
                                       testloader.dataset.transform)
                    te_afterbase_inter_angles, te_afterbase_intra_angle_mean, te_afterbase_intra_angle_std, \
                    te_afterbase_angle_feat_fc, te_afterbase_angle_feat_fc_std, te_afterbase_angle_featmean_fc = \
                        base_angle_exp(args, testloader, False, procD, clsD, model)
                    print('sess 0 after train')
                    print(afterbase_inter_angles, afterbase_intra_angle_mean, afterbase_intra_angle_std, afterbase_angle_feat_fc, \
                          afterbase_angle_feat_fc_std, afterbase_angle_featmean_fc)
                    print(te_afterbase_inter_angles, te_afterbase_intra_angle_mean, te_afterbase_intra_angle_std, \
                          te_afterbase_angle_feat_fc, te_afterbase_angle_feat_fc_std, te_afterbase_angle_featmean_fc)
                # fig, (ax1,ax2,ax3) = plt.subplots(1,3)
                # ax1.plot(range(args.epochs_base), angles)
                # ax2.plot(range(args.epochs_base), tas)
                # ax3.plot(range(args.epochs_base), tsas)
                # plt.show()
                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, procD['trlog']['max_acc_epoch'], procD['trlog']['max_acc'][session], ))
                if args.use_coreset:
                    indexes = torch.arange(args.base_class)
                    args.coreset[session] = get_coreset(args, indexes, model, trainloader, session, clsD, testloader.dataset.transform)
                    if args.check_implementation:
                        for key in args.coreset:
                            print('sess %d coreset' %session, key, args.coreset[key].shape)


            else:  # incremental learning sessions

                print("training session: [%d]" % session)
                num_batches = len(trainloader)
                model, optimizer, scheduler = self.get_optimizer_new(args, model, num_batches, session)
                transform_ = trainloader.dataset.transform
                trainloader.dataset.transform = testloader.dataset.transform
                trainloader.dataset.transform = transform_

                for epoch in range(args.epochs_new):
                    procD['epoch'] += 1
                    start_time = time.time()
                    tl, ta = self.new_train(args, model, procD, clsD, trainloader, optimizer, kdloss, scheduler,
                                                 epoch, loss_fn, supcon_criterion)


                print('Incremental session, test')

                model.eval()

                tsl, tsa = self.test(args, model, procD, clsD, testloader)
                #ntsl, ntsa = self.test(args, model, procD, clsD, new_testloader)
                #ptsl, ptsa = self.test(args, model, procD, clsD, prev_testloader)

                _, btsa = self.test(args, model, procD, clsD, base_testloader)
                _, natsa = self.test(args, model, procD, clsD, new_all_testloader)
                ntsa = natsa
                ptsa = btsa

                # save model
                procD['trlog']['max_acc'][session] = float('%.3f' % (tsa * 100))
                procD['trlog']['new_max_acc'][session] = float('%.3f' % (ntsa * 100))
                procD['trlog']['new_all_max_acc'][session] = float('%.3f' % (natsa * 100))
                procD['trlog']['base_max_acc'][session] = float('%.3f' % (btsa * 100))
                procD['trlog']['prev_max_acc'][session] = float('%.3f' % (ptsa * 100))

                result_list.append(
                    'Session {}, test Acc {:.3f}\n'.format(session, procD['trlog']['max_acc'][session]))

                if args.use_coreset:
                    indexes = torch.arange(args.way) + args.base_class + (args.way)*(session-1)
                    args.coreset[session] = get_coreset(args, indexes, model, trainloader_nocoreset_noshuff, session, clsD, testloader.dataset.transform)

                    if args.check_implementation:
                        for key in args.coreset:
                            print('sess %d coreset' %session, key, args.coreset[key].shape)
                        print('sess %d trloader_nocore'%session, trainloader_nocoreset_noshuff.dataset.targets)
                        print('sess %d trloader_nocore' % session, clsD['seen_unsort_map'][trainloader_nocoreset_noshuff.dataset.targets])
                        # not work for mini, where ~.dataset.data has list of ~~/~~/ImageNoxx.jpg and it is converted via getitem
                        print('sess %d trloader_nocore' % session, trainloader_nocoreset_noshuff.dataset.data.shape)
                        print('sess %d trloader'%session, trainloader.dataset.targets)
                        print('sess %d trloader' % session, trainloader.dataset.data.shape)

                if args.plot_tsne:
                    # if session !=  args.sessions -1:
                    #    continue
                    base_tsne_idx = torch.arange(args.base_class)[torch.randperm(args.base_class)[:draw_base_cls_num]]
                    palette = np.array(sns.color_palette("hls", args.base_class + args.way * session))

                    data_base_, label_base_ = tot_datalist(args, base_testloader, model, False, clsD['seen_unsort_map'],
                                                           gpu=False)
                    data_base_, label_base_ = selec_datalist(args, data_base_, label_base_, base_tsne_idx,
                                                             draw_n_per_basecls)
                    # draw_tsne(data_, label_, n_components, perplexity, palette, base_tsne_idx, 'base test')

                    new_tsne_idx = torch.arange(args.way * session) + args.base_class
                    new_tsne_idx = new_tsne_idx[-20:]
                    data_new_, label_new_ = tot_datalist(args, new_all_testloader, model, False,
                                                         clsD['seen_unsort_map'], gpu=False)
                    data_new_, label_new_ = selec_datalist(args, data_new_, label_new_, new_tsne_idx,
                                                           draw_n_per_basecls)

                    data_ = torch.cat((data_base_, data_new_), dim=0)
                    label_ = torch.cat((label_base_, label_new_), dim=0)
                    combine_tsne_idx = torch.cat((base_tsne_idx, new_tsne_idx), dim=0)
                    # draw_tsne(data_, label_, n_components, perplexity, palette, combine_tsne_idx, 'new test')
                    palette2 = np.array(sns.color_palette("hls", args.way * session))
                    lll = label_new_ - args.base_class
                    iii = new_tsne_idx - args.base_class
                    # draw_tsne(data_new_, label_new_, n_components, perplexity, palette2, new_tsne_idx, 'new test')
                    draw_tsne(data_new_, lll, n_components, perplexity, palette2, iii, 'new test')


                # if session == 1:
                if args.angle_exp:
                    if session == args.sessions-1:
                        inter_angles_base, angle_intra_mean_base, angle_intra_std_base, angle_feat_fc_base, angle_feat_fc_base_std, \
                        inter_angles_inc, angle_intra_mean_inc, angle_intra_std_inc, angle_feat_fc_inc, angle_feat_fc_inc_std, \
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

                    # print(inter_angles_base, angle_intra_std_base, angle_feat_fc_base, inter_angles_inc, angle_intra_std_inc, \
                    # angle_feat_fc_inc, angle_base_feats_new_clf, angle_base_clfs_new_feat, base_inc_fc_angle, inc_inter_fc_angle)

        result_list.append('Base Session Best Epoch {}\n'.format(procD['trlog']['max_acc_epoch']))
        result_list.append(procD['trlog']['max_acc'])
        print('max_acc:', procD['trlog']['max_acc'])
        print('new_max_acc:', procD['trlog']['new_max_acc'])
        print('new_all_max_acc:', procD['trlog']['new_all_max_acc'])
        print('base_max_acc:', procD['trlog']['base_max_acc'])
        print('prev_max_acc:', procD['trlog']['prev_max_acc'])
        save_list_to_txt(os.path.join(args.final_save_path, 'results.txt'), result_list)
        #save_obj(args.final_save_path, procD, clsD, bookD)

        print('Base Session Best epoch:', procD['trlog']['max_acc_epoch'])

        if args.angle_exp:
            print('base exp result')
            print(init_inter_angles, init_intra_angle_mean, init_intra_angle_std, init_angle_feat_fc,
                  init_angle_feat_fc_std, \
                  init_angle_featmean_fc)
            print(te_init_inter_angles, te_init_intra_angle_mean, te_init_intra_angle_std, te_init_angle_feat_fc, \
                  te_init_angle_feat_fc_std, te_init_angle_featmean_fc)
            print(afterbase_inter_angles, afterbase_intra_angle_mean, afterbase_intra_angle_std, \
                  afterbase_angle_feat_fc, afterbase_angle_feat_fc_std, afterbase_angle_featmean_fc)
            print(te_afterbase_inter_angles, te_afterbase_intra_angle_mean, te_afterbase_intra_angle_std, \
                  te_afterbase_angle_feat_fc, te_afterbase_angle_feat_fc_std, te_afterbase_angle_featmean_fc)

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
