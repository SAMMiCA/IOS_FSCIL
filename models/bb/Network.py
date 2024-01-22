import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet18_encoder_cifar import CIFAR_ResNet18, CIFAR_ResNet18_1, CIFAR_ResNet18_2, CIFAR_ResNet18_3
from models.supcon_resnet_cifar import supcon_resnet18
from vissl.utils.checkpoint import replace_module_prefix
from utils_s import *

class MYNET(nn.Module):

    def __init__(self, args, fw_mode=None, SL=None, arch=None):
        super().__init__()

        #self.new_mode = args.new_mode
        #self.temperature = args.temperature
        self.m = args.m
        self.s = args.s
        self.fw_mode = fw_mode
        self.base_class = args.base_class
        self.way = args.way
        self.use_head = args.use_head
        self.use_encmlp = args.use_encmlp
        if args.use_head:
            self.head_dim = args.head_dim
            self.head_type = args.head_type
        if args.use_encmlp:
            self.encmlp_dim = args.encmlp_dim
            self.encmlp_layers = args.encmlp_layers

        #self.base_mode = args.base_mode
        # self.num_features = 512
        model_dict = {
            'resnet18': [resnet18, 512],
            'CIFAR_resnet18': [CIFAR_ResNet18, 512],
            'CIFAR_resnet18_1': [CIFAR_ResNet18_1, 512],
            'CIFAR_resnet18_2': [CIFAR_ResNet18_2, 512],
            'CIFAR_resnet18_3': [CIFAR_ResNet18_3, 512],
            'supcon_resnet18': [supcon_resnet18, 512],
            'resnet20': [resnet20, 64],
            'resnet34': [resnet34, 512],
            'resnet50': [resnet50, 2048],
            'resnet101': [resnet101, 2048],
        }

        # SL, arch option not used for original FSCIL.
        # SL, arch option used for linear_eval option.
        if SL is None:
            if arch is None:
                if args.dataset in ['cifar100']:
                    if args.use_cifar_resnet18:
                        if args.use_cifar_resnet18_opt1:
                            model_, self.num_features = model_dict['CIFAR_resnet18_1']
                        elif args.use_cifar_resnet18_opt2:
                            model_, self.num_features = model_dict['CIFAR_resnet18_2']
                        elif args.use_cifar_resnet18_opt3:
                            model_, self.num_features = model_dict['CIFAR_resnet18_3']
                        else:
                            model_, self.num_features = model_dict['CIFAR_resnet18']
                    elif args.use_supcon_resnet18:
                        model_, self.num_features = model_dict['supcon_resnet18']
                    else:
                        model_, self.num_features = model_dict['resnet20']
                    self.encoder = model_()
                if args.dataset in ['mini_imagenet']:
                    if not args.use_cifar_resnet18_mini:
                        model_, self.num_features = model_dict['resnet18']
                    else:
                        model_, self.num_features = model_dict['CIFAR_resnet18_2']
                    #self.encoder = model_(False, args)
                    self.encoder = model_()
                if args.dataset == 'cub200':
                    model_, self.num_features = model_dict['resnet18']
                    self.encoder = model_(True, args)
            else:
                raise NotImplementedError
        else:
            if arch is None:
                raise NotImplementedError
            else:
                if args.dataset in ['cifar100']:
                    model_, self.num_features = model_dict[arch]
                    # model_, self.num_features = model_dict['resnet18']
                    self.encoder = model_()
                if args.dataset in ['mini_imagenet']:
                    model_, self.num_features = model_dict[arch]
                    self.encoder = model_(SL, args)
                if args.dataset == 'cub200':
                    model_, self.num_features = model_dict[arch]
                    self.encoder = model_(SL, args)

        if not args.use_encmlp:
            self.fc = nn.Linear(self.num_features, args.num_classes, bias=False)
        else:
            self.fc = nn.Linear(self.encmlp_dim, args.num_classes, bias=False)

        if args.use_head:
            if self.head_type == 'linear':
                self.head = nn.Linear(self.num_features, self.head_dim)
            elif self.head_type == 'mlp':
                self.head = nn.Sequential(
                    nn.Linear(self.num_features, self.num_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_features, self.head_dim)
                )
            else:
                raise NotImplementedError(
                    'head not supported: {}'.format(self.head_type))
        if args.use_encmlp:
            self.encmlp = projection_MLP(self.num_features, self.encmlp_dim, self.encmlp_layers)


        #self.fc.weight.data = abs(self.fc.weight.data)
        #nn.init.orthogonal_(self.fc.weight)

        #with torch.no_grad(): ### edit on 221009 for cosface debug mini
        #    self.fc.weight *= 2.321
        nn.init.xavier_uniform_(self.fc.weight)

        if args.fw_mode == 'arcface':
            self.cos_m = math.cos(self.m)
            self.sin_m = math.sin(self.m)
            self.th = math.cos(math.pi - self.m)
            self.mm = math.sin(math.pi - self.m) * self.m

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def set_mode(self, fw_mode):
        self.fw_mode = fw_mode

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        if self.use_encmlp:
            x = self.encmlp(x)
        return x

    def calc_head(self, x, doenc=True):
        if doenc:
            x = self.encoder(x)
        x = F.normalize(self.head(x), dim=1)
        return x

    def forward(self, input, label=None, sess=None, doenc=True):
        if self.fw_mode == 'encoder':
            feat = self.encode(input)
            return feat
        elif self.fw_mode == 'fc_cos' or self.fw_mode == 'fc_dot':
            output = self.forward_fc(x=input, sess=sess, doenc=doenc)
            return output
        elif self.fw_mode == 'fc_cosface' or self.fw_mode == 'fc_arcface':
            logit, cos_logit = self.forward_fc(x=input, sess=sess, label=label, doenc=doenc)
            return logit, cos_logit
        elif self.fw_mode == 'head':
            output = self.calc_head(input, doenc=doenc)
            return output
        else:
            raise NotImplementedError

    def forward_fc(self, x, sess, label=None, doenc=True):
        if doenc:
            x = self.encode(x)
        n_cls = self.base_class if sess==0 else self.base_class + self.way*(sess)
        #fc = self.fc[:n_cls]
        if self.fw_mode == 'fc_cos':
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            #x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
            x = self.s * x
            x = x[:, :n_cls]
            return x
        elif self.fw_mode == 'fc_dot':
            x = self.fc(x)
            x = x[:, :n_cls]
            return x
        elif self.fw_mode == 'fc_cosface':
            cos = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            cos = cos[:, :n_cls]
            phi = cos - self.m
            # --------------------------- convert label to one-hot ---------------------------

            B = x.shape[0]
            one_hot = torch.arange(n_cls).expand(B, n_cls).cuda()
            label_ = label.unsqueeze(1).expand(B, n_cls)
            one_hot = torch.where(one_hot == label_, 1, 0)
            output = (one_hot * phi) + ((1.0 - one_hot) * cos)
        elif self.fw_mode == 'fc_arcface':
            cos = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            cos = cos[:, :n_cls]
            sine = torch.sqrt((1.0 - torch.pow(cos, 2)).clamp(0, 1))
            phi = cos * self.cos_m - sine * self.sin_m

            phi = torch.where(cos > self.th, phi, cos - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cos.size(), device='cuda')
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = (one_hot * phi) + (
                    (1.0 - one_hot) * cos)
        else:
            raise NotImplementedError

        output *= self.s
        return output, cos


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim

        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x

