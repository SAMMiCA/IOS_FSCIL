import random
import torch
import os
import time
import math
import numpy as np
import pprint as pprint
from sklearn import manifold
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from torchvision.utils import save_image
import copy
from omnixai_iu.data.image import Image as OmniImage
from PIL import Image
import cv2

from omnixai_iu.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM
from abc import abstractmethod
from omnixai_iu.utils.misc import AutodocABCMeta
from omnixai_iu.preprocessing.image import Resize
from omnixai_iu.utils.misc import is_torch_available
from omnixai_iu.explanations.image.pixel_importance import PixelImportance
from clip_iu.simple_tokenizer import SimpleTokenizer as _Tokenizer
from captum.attr import visualization

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss



def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()

def init_maps(args):
    #book = args.cls_book
    st = args.base_class
    inc = args.way
    tot = args.num_classes
    class_order = args.task_class_order

    tasks = []
    class_maps = []
    p = 0

    tasks.append(class_order[:st])
    class_map = np.full(tot, -1)
    for i, j in enumerate(tasks[-1]): class_map[j] = i
    class_maps.append(class_map)
    p += st

    while p < tot:
        tasks.append(class_order[p:p + inc])
        class_map = np.full(tot, -1)
        for i, j in enumerate(tasks[-1]): class_map[j] = i
        class_maps.append(class_map)
        p += inc
    #book['tasks'] = [torch.tensor(task).cuda() for task in tasks]
    tasks = [torch.tensor(task).cuda() for task in tasks]
    #book['class_maps'] = [torch.tensor(class_map).cuda() for class_map in class_maps]
    class_maps = [torch.tensor(class_map).cuda() for class_map in class_maps]
    return tasks, class_maps

def inc_maps(args, clsD, procD, session):
    #book = args.cls_book
    tasks = clsD['tasks']
    clsD_ = clsD
    num_classes = args.num_classes
    #session = procD['session']

    prev = sorted(set([k for task in tasks[:session] for k in task]))
    prev_unsort = [k for task in tasks[:session] for k in task]
    seen = sorted(set([k for task in tasks[:session + 1] for k in task]))
    seen_unsort = [k for task in tasks[:session + 1] for k in task]
    prev_map = np.full(num_classes, -1)
    seen_map = np.full(num_classes, -1)
    prev_unsort_map = np.full(num_classes, -1)
    seen_unsort_map = np.full(num_classes, -1)
    for i, j in enumerate(prev): prev_map[j] = i
    for i, j in enumerate(seen): seen_map[j] = i
    for i, j in enumerate(prev_unsort): prev_unsort_map[j] = i
    for i, j in enumerate(seen_unsort): seen_unsort_map[j] = i

    clsD_['prev'] = torch.tensor(prev, dtype=torch.long).cuda()
    clsD_['prev_unsort'] = torch.tensor(prev_unsort, dtype=torch.long).cuda()
    clsD_['seen'] = torch.tensor(seen, dtype=torch.long).cuda()
    clsD_['seen_unsort'] = torch.tensor(seen_unsort, dtype=torch.long).cuda()
    clsD_['prev_map'] = torch.tensor(prev_map).cuda()
    clsD_['seen_map'] = torch.tensor(seen_map).cuda()
    clsD_['prev_unsort_map'] = torch.tensor(prev_unsort_map).cuda()
    clsD_['seen_unsort_map'] = torch.tensor(seen_unsort_map).cuda()
    #return prev_map, seen_map
    return clsD_


def book_val(args):

    book_v = []
    num_classes = args.num_classes

    st = args.base_class
    inc = args.way
    tot = args.num_classes
    class_order = args.task_class_order

    tasks = []
    class_maps = []
    p = 0

    tasks.append(class_order[:st])
    class_map = np.full(tot, -1)
    for i, j in enumerate(tasks[-1]): class_map[j] = i
    class_maps.append(class_map)
    p += st

    while p < tot:
        tasks.append(class_order[p:p + inc])
        class_map = np.full(tot, -1)
        for i, j in enumerate(tasks[-1]): class_map[j] = i
        class_maps.append(class_map)
        p += inc
    tasks_ = [torch.tensor(task).cuda() for task in tasks]
    class_maps_ = [torch.tensor(class_map).cuda() for class_map in class_maps]

    #for session in range(args.start_session, args.sessions):
    for session in range(args.sessions):
        book_vs = {}
        book_vs['tasks'] = tasks_
        book_vs['class_maps'] = class_maps_

        tasks = book_vs['tasks']
        prev = sorted(set([k for task in tasks[:session] for k in task]))
        prev_unsort = [k for task in tasks[:session] for k in task]
        seen = sorted(set([k for task in tasks[:session + 1] for k in task]))
        seen_unsort = [k for task in tasks[:session + 1] for k in task]
        prev_map = np.full(num_classes, -1)
        seen_map = np.full(num_classes, -1)
        prev_unsort_map = np.full(num_classes, -1)
        seen_unsort_map = np.full(num_classes, -1)
        for i, j in enumerate(prev): prev_map[j] = i
        for i, j in enumerate(seen): seen_map[j] = i
        for i, j in enumerate(prev_unsort): prev_unsort_map[j] = i
        for i, j in enumerate(seen_unsort): seen_unsort_map[j] = i

        book_vs['prev'] = torch.tensor(prev, dtype=torch.long).cuda()
        book_vs['prev_unsort'] = torch.tensor(prev_unsort, dtype=torch.long).cuda()
        book_vs['seen'] = torch.tensor(seen, dtype=torch.long).cuda()
        book_vs['seen_unsort'] = torch.tensor(seen_unsort, dtype=torch.long).cuda()
        book_vs['prev_map'] = torch.tensor(prev_map).cuda()
        book_vs['seen_map'] = torch.tensor(seen_map).cuda()
        book_vs['prev_unsort_map'] = torch.tensor(prev_unsort_map).cuda()
        book_vs['seen_unsort_map'] = torch.tensor(seen_unsort_map).cuda()

        book_v.append(book_vs)

    return book_v



"""

book['prev'] = torch.tensor(prev, dtype=torch.long).cuda()
book['prev_unsort'] = torch.tensor(prev_unsort, dtype=torch.long).cuda()
book['seen'] = torch.tensor(seen, dtype=torch.long).cuda()
book['seen_unsort'] = torch.tensor(seen_unsort, dtype=torch.long).cuda()
book['prev_map'] = torch.tensor(prev_map).cuda()
book['seen_map'] = torch.tensor(seen_map).cuda()
book['prev_unsort_map'] = torch.tensor(prev_unsort_map).cuda()
book['seen_unsort_map'] = torch.tensor(seen_unsort_map).cuda()
"""


def learn_gauss(args, trainloader, model, clsD, procD, gaussD):
    # only model on gpu
    # Else given by cpu (torch)
    sess = procD['session']
    beta =  args.tukey_beta
    num_features = model.module.num_features
    #base_mean = torch.zeros(base_class, num_features)
    #base_cov = torch.zeros(base_class, num_features, num_features)
    # gaussD[i] = mean, cov where mean.shape: (num_features), cov.shape: (num_features, num_features)
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            if args.base_doubleaug is False:
                data, train_label = [_.cuda() for _ in batch]
                #target_cls = clsD['class_maps'][sess][train_label]
                target_cls = clsD['seen_unsort_map'][train_label]
            else:
                data = torch.cat((batch[0][0],batch[0][1]),dim=0).cuda()
                train_label = batch[1].cuda()
                train_label = train_label.repeat(2)
                #target_cls = clsD['class_maps'][sess][train_label]
                target_cls = clsD['seen_unsort_map'][train_label]

            label = target_cls
            #model.module.mode = 'encoder'
            embedding = model.module.encode(data)
            embedding = torch.pow(embedding,beta)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    classnames = np.unique(label_list)
    for i in classnames:
        #ind_cl = torch.where(i == seen_unsort_map_[label_list])[0]
        ind_cl = torch.where(i == label_list)[0]
        gaussD['mean'][i] = embedding_list[ind_cl].mean(dim=0)
        #mat = embedding_list[ind_cl] - embedding_list[ind_cl].mean(dim=0)  # 500,512
        mat = embedding_list[ind_cl] - gaussD['mean'][i]  # 500,512
        mat = mat.unsqueeze(dim=2)  # 500,512,1
        mat2 = mat.permute(0, 2, 1)  # 500,1,512
        cov_ = torch.bmm(mat, mat2)  # 500,512,512
        cov_ = torch.sum(cov_,dim=0)/(len(cov_)-1)
        gaussD['cov'][i] = cov_
    return gaussD

def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    # torch cpu
    dist = []
    for i in range(len(base_means)):
        dist.append(torch.norm(query - base_means[i]))
    index = torch.topk(torch.tensor(dist),k).indices
    slc_base_means = torch.index_select(base_means,dim=0,index=index)
    mean = torch.cat([slc_base_means, query.unsqueeze(0)])
    calibrated_mean = torch.mean(mean, dim=0)
    slc_base_covs = torch.index_select(base_cov,dim=0,index=index)
    calibrated_cov = torch.mean(slc_base_covs, dim=0) + alpha

    return calibrated_mean, calibrated_cov

def distribution_calibration2(query, base_means, base_cov, k, alpha=0.21):
    # torch cpu
    dist = []
    for i in range(len(base_means)):
        dist.append(torch.norm(query - base_means[i]))
    index = torch.topk(torch.tensor(dist),k).indices.cuda()
    slc_base_covs =torch.index_select(base_cov,dim=0,index=index)
    calibrated_cov = torch.mean(slc_base_covs, dim=0) + alpha
    return calibrated_cov


#def checkparser_dependencies(args):

def f(d,n):
    x = math.pow(n,-(2/(d-1)))
    y = math.gamma(1+1/(d-1))
    z = (math.gamma(d/2)/(2*math.sqrt(math.pi)*(d-1)*math.gamma((d-1)/2)))
    return x*y*(math.pow(z,-(1/(d-1))))


def tot_datalist(args, dataloader, model, doubleaug, map=None):
    # model, map is assumed to be in gpu

    data_ = []
    label_ = []
    with torch.no_grad():
        model.eval()
        #set_seed(0)
        for batch in dataloader:
            if doubleaug is False:
                data, label = [_.cuda() for _ in batch]
            else:
                data = batch[0][0].cuda()
                label = batch[1].cuda()
            #data = model(data).detach()
            #data = model.module.clip_img_encoder(data).detach()
            data = model.module.image_encoder(data)
            data = data.detach()
            data_.append(data)
            label_.append(label)

        data_ = torch.cat(data_, dim=0)
        label_ = torch.cat(label_, dim=0)
        if map is not None:
            label_cls = (map)[label_]
        else:
            label_cls = label_
        #data_ = np.array(data_)
        #label_cls = np.array(label_cls)
        return data_, label_cls

def tot_datalist_train(args, dataloader, model, doubleaug, map=None):
    # model, map is assumed to be in gpu

    data_ = []
    label_ = []
    model.train()
    model.module.set_mode('encoder')
    #set_seed(0)
    for batch in dataloader:
        if doubleaug is False:
            data, label = [_.cuda() for _ in batch]
        else:
            data = batch[0][0].cuda()
            label = batch[1].cuda()
        #data = model(data).detach()
        #data = model.module.clip_encoder(data)
        data = model.module.encode(data)
        data_.append(data)
        label_.append(label)
    data_ = torch.cat(data_, dim=0)
    label_ = torch.cat(label_, dim=0)
    if map is not None:
        label_cls = (map)[label_]
    else:
        label_cls = label_
    #data_ = np.array(data_)
    #label_cls = np.array(label_cls)
    return data_, label_cls


def selec_datalist(args, datas, labels, idx, n_per_cls):
    labellist = set(np.unique(labels.cpu()).tolist())
    idxlist = set(np.unique(idx.cpu()).tolist())
    assert idxlist.issubset(labellist)

    d_ = []
    l_ = []
    for i in idx:
        j_ = torch.where(i == labels)[0][:n_per_cls]
        d_.append(datas[j_])
        for _ in range(len(j_)):
            l_.append(i)
    res_data = torch.stack(d_)
    res_data = res_data.view(-1, datas.shape[1])
    res_label = torch.stack(l_)

    return res_data, res_label


def draw_tsne(data_, label_, n_components, perplexity,palette, idxs , title=None):

    tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
    x = tsne.fit_transform(data_)
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    #c=palette[label_.astype(np.int)])
                    c=palette[torch.tensor(label_,dtype=int)])
    plt.title(title)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    """
    for i in idxs:
        # Position of each label.
        xtext, ytext = np.median(x[label_ == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(int(i)), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    """
    plt.show()

def set_trainable_module(moduelist_, ts=[]):
    if not isinstance(ts, (list, range)):
        ts = [ts]
    for t, m in enumerate(moduelist_):
        requires_grad = (t in ts)
        for param in m.parameters():
            param.requires_grad = requires_grad

def set_trainable_param(paramlist_, ts=[]):
    if not isinstance(ts, (list, range)):
        ts = [ts]
    for t, m in enumerate(paramlist_):
        requires_grad = (t in ts)
        m.requires_grad = requires_grad
    return paramlist_


def save_obj(save_path, procD, clsD, bookD, gaussD=None):
    dict = {}
    dict['procD']=procD
    dict['clsD'] = clsD
    dict['bookD'] = bookD
    if gaussD is not None:
        dict['gaussD'] = gaussD
    fn = os.path.join(save_path, 'saved_dicts')
    with open(fn,'wb') as f:
        pickle.dump(dict,f,pickle.HIGHEST_PROTOCOL)
    print('save object saved')

def get_coreset(args, indexes, model, trainloader, session, clsD, transform):
    loader = copy.deepcopy(trainloader)
    loader.dataset.transform = transform
    embedding_list, label_list = tot_datalist(args, loader, model, doubleaug=False,
                                              map=clsD['seen_unsort_map'])

    label_list = label_list.cuda()
    mean_list = []

    #for class_index in clsD['tasks'][session]:
    for cls_idx in indexes:
        data_index = (label_list == cls_idx).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        mean_list.append(embedding_this)

    mean_list = torch.stack(mean_list, dim=0)

    coreset_list = []
    #for k, cls in enumerate(clsD['tasks'][session]):
    for k, cls in enumerate(indexes):
        # for class_index in range(len(model.module.angle_w[session].data)):
        cos_ = nn.CosineSimilarity(dim=1, eps=1e-6)
        top_close_feature = []
        top_distance = []
        data_index = (label_list == cls).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        mean_ = mean_list[k].unsqueeze(0)
        #mean_ = mean_.unsqueeze(0).repeat(embedding_this.shape[0],1)
        for i in range(len(embedding_this)):
            if i < args.shot:
                top_close_feature.append(embedding_this[i])
                dist_ = cos_(mean_, embedding_this[i].unsqueeze(0))[0]
                #dist_ = cos_(mean_, embedding_this[i])
                top_distance.append(dist_)
            else:
                dist_ = cos_(mean_, embedding_this[i].unsqueeze(0))[0]
                for j in range(len(top_distance)):
                    if dist_ > top_distance[j]: # cos so opposite
                        top_close_feature[j] = embedding_this[i]
                        top_distance[j] = dist_
        top_close_feature = torch.stack(top_close_feature, dim=0)
        coreset_list.append(top_close_feature)
    coreset_list = torch.stack(coreset_list, dim=0)

    return coreset_list






def cosine_distance(input1, input2):
    if len(input1.shape)>1 and len(input2.shape)>1:
        return F.linear(F.normalize(input1), F.normalize(input2))
    else:
        return F.linear(input1/torch.norm(input1), input2/torch.norm(input2))

def cos2angle(cosine):
    return torch.acos(cosine.clamp(-1,1)) * 180/math.pi

def angle_btw_base_new(feats1, feats2):
    cos = F.linear(F.normalize(feats1), F.normalize(feats2)).clamp(-1, 1)
    theta = torch.acos(cos) * 180 / math.pi
    mean_angle = torch.sum(theta) / (theta.shape[0] * theta.shape[1])
    return mean_angle


def get_angle_feats_vec(feats, vec):
    len_ = len(feats)
    vec = vec.repeat(len_, 1)
    cos = F.linear(F.normalize(feats), F.normalize(vec)).clamp(-1, 1)
    theta = torch.acos(cos) * 180 / math.pi

    mean_ang = torch.mean(torch.diagonal(theta, 0))
    std_ang = torch.std(torch.diagonal(theta, 0))

    return mean_ang, std_ang

def get_inter_angle(feats):
    cos = F.linear(F.normalize(feats),F.normalize(feats)).clamp(-1,1)
    theta = torch.acos(cos)*180/math.pi
    sum = torch.sum(theta)-torch.sum(torch.diagonal(theta,0))
    sum /= len(theta)*(len(theta)-1)
    return sum

def get_intra_angle(feats):
    mean_ = torch.mean(F.normalize(feats), dim=0)
    mean_ang, std_ang = get_angle_feats_vec(feats, mean_)
    return mean_ang, std_ang


def base_angle_exp(args, base_loader, doubleaug, procD, clsD, model, transform=None):
    if transform is not None:
        base_loader.dataset.transform = transform
    dd_ = tot_datalist(args, base_loader, model, doubleaug=doubleaug, module=True)
    cls = clsD['tasks'][0].cpu()
    d_ = []
    for i in cls:
        j_ = torch.where(i == dd_[1])[0]
        d_.append(dd_[0][j_])

    mean = []
    intra_mean = []
    intra_std = []
    mean_feats_fc_angle = []
    std_feats_fc_angle = []
    featmean_fc_angle = []

    for ii in range(len(d_)):
        #mean_ = torch.mean(d_[ii], dim=0)
        mean_ = torch.mean(F.normalize(d_[ii]), dim=0)
        mean.append(mean_)

        intra_mean_, intra_std_ = get_intra_angle(d_[ii]) # normalized inside

        intra_mean.append(intra_mean_)
        intra_std.append(intra_std_)

        #mean_feats_fc_angle_, std_feats_fc_angle_ = get_angle_feats_vec(d_[ii], model.fc.weight[ii].cpu())
        mean_feats_fc_angle_, std_feats_fc_angle_ = get_angle_feats_vec(d_[ii], model.module.textual_classifier.weight[ii].cpu())
        mean_feats_fc_angle.append(mean_feats_fc_angle_)
        std_feats_fc_angle.append(std_feats_fc_angle_)

        #featmean_fc_angle_ = F.linear(F.normalize(mean_, dim=0), F.normalize(model.fc.weight[ii].cpu(), dim=0)).clamp(-1, 1)
        featmean_fc_angle_ = F.linear(F.normalize(mean_, dim=0), F.normalize(model.module.textual_classifier.weight[ii].cpu(), dim=0)).clamp(
            -1, 1)
        featmean_fc_angle_ = torch.acos(featmean_fc_angle_) * 180 / math.pi
        featmean_fc_angle.append(featmean_fc_angle_)


    mean = torch.stack(mean)
    intra_mean = torch.stack(intra_mean)
    intra_std = torch.stack(intra_std)
    mean_feats_fc_angle = torch.stack(mean_feats_fc_angle)
    std_feats_fc_angle = torch.stack(std_feats_fc_angle)
    featmean_fc_angle = torch.stack(featmean_fc_angle)

    angle_intra_mean = torch.mean(intra_mean, dim=0)
    angle_intra_std = torch.mean(intra_std, dim=0)

    inter_angles = get_inter_angle(mean)

    angle_feat_fc = torch.mean(mean_feats_fc_angle, dim=0)
    angle_feat_fc_std = torch.mean(std_feats_fc_angle,dim=0)

    angle_featmean_fc = torch.mean(featmean_fc_angle, dim=0)

    return inter_angles, angle_intra_mean, angle_intra_std,  angle_feat_fc, angle_feat_fc_std, angle_featmean_fc

def inc_angle_exp(args, base_testloader, new_testloader, doubleaug, procD, clsD, model):

    dd_base = tot_datalist(args, base_testloader, model, False, module=True)
    dd_inc = tot_datalist(args, new_testloader, model, False, module=True)
    session = procD['session']
    start_class = args.base_class + args.way * (session - 1)

    cls_base = clsD['tasks'][0].cpu()
    cls_inc = clsD['tasks'][session].cpu()
    assert args.way == len(cls_inc)

    d_base = []
    d_inc = []
    for i in cls_base:
        j_ = torch.where(i == dd_base[1])[0]
        d_base.append(dd_base[0][j_])
    for i in cls_inc:
        j_ = torch.where(i == dd_inc[1])[0]
        d_inc.append(dd_inc[0][j_])

    mean_base = []
    intra_mean_base = []
    intra_std_base = []
    mean_feats_fc_angle_base = []
    std_feats_fc_angle_base = []

    angle_base_feats_new_clfs = []
    featmean_fc_angle_base = []
    featmean_fc_angle_inc = []

    for ii in range(len(d_base)):
        mean_base_ = torch.mean(F.normalize(d_base[ii]), dim=0)
        mean_base.append(mean_base_)
        intra_mean_base_, intra_std_base_ = get_intra_angle(d_base[ii])
        intra_mean_base.append(intra_mean_base_)
        intra_std_base.append(intra_std_base_)
        mean_feats_fc_angle_base_, std_feats_fc_angle_base_ = get_angle_feats_vec(d_base[ii], model.module.textual_classifier.weight[ii].cpu())
        mean_feats_fc_angle_base.append(mean_feats_fc_angle_base_)
        std_feats_fc_angle_base.append(std_feats_fc_angle_base_)

        angle_base_feats_new_clfs_ = angle_btw_base_new(d_base[ii], model.module.textual_classifier.weight[
                                                                   start_class:start_class + args.way].cpu())
        angle_base_feats_new_clfs.append(angle_base_feats_new_clfs_)

        featmean_fc_angle_ = F.linear(F.normalize(mean_base_,dim=0), F.normalize(model.module.textual_classifier.weight[ii].cpu(),dim=0)).clamp(-1, 1)
        featmean_fc_angle_ = torch.acos(featmean_fc_angle_) * 180 / math.pi
        featmean_fc_angle_base.append(featmean_fc_angle_)

    mean_inc = []
    intra_mean_inc = []
    intra_std_inc = []
    mean_feats_fc_angle_inc = []
    std_feats_fc_angle_inc = []
    angle_base_clfs_new_feats = []

    for ii in range(len(d_inc)):
        mean_inc_ = torch.mean(F.normalize(d_inc[ii]), dim=0)
        mean_inc.append(mean_inc_)
        intra_mean_inc_, intra_std_inc_ = get_intra_angle(d_inc[ii])
        intra_mean_inc.append(intra_mean_inc_)
        intra_std_inc.append(intra_std_inc_)
        mean_feats_fc_angle_inc_, std_feats_fc_angle_inc_ = get_angle_feats_vec(d_inc[ii],
                                       model.module.textual_classifier.weight[start_class + ii].cpu())
        mean_feats_fc_angle_inc.append(mean_feats_fc_angle_inc_)
        std_feats_fc_angle_inc.append(std_feats_fc_angle_inc_)
        angle_base_clfs_new_feats_ = angle_btw_base_new(d_inc[ii], model.module.textual_classifier.weight[:args.base_class].cpu())
        angle_base_clfs_new_feats.append(angle_base_clfs_new_feats_)

        featmean_fc_angle_ = F.linear(F.normalize(mean_inc_,dim=0), F.normalize(model.module.textual_classifier.weight[start_class+ii].cpu(),dim=0)).clamp(-1,1)
        featmean_fc_angle_ = torch.acos(featmean_fc_angle_) * 180 / math.pi
        featmean_fc_angle_inc.append(featmean_fc_angle_)

    mean_base = torch.stack(mean_base)
    intra_mean_base = torch.stack(intra_mean_base)
    intra_std_base = torch.stack(intra_std_base)
    mean_feats_fc_angle_base = torch.stack(mean_feats_fc_angle_base)
    std_feats_fc_angle_base = torch.stack(std_feats_fc_angle_base)
    mean_inc = torch.stack(mean_inc)
    intra_mean_inc = torch.stack(intra_mean_inc)
    intra_std_inc = torch.stack(intra_std_inc)
    mean_feats_fc_angle_inc = torch.stack(mean_feats_fc_angle_inc)
    std_feats_fc_angle_inc = torch.stack(std_feats_fc_angle_inc)
    angle_base_feats_new_clfs = torch.stack(angle_base_feats_new_clfs)
    angle_base_clfs_new_feats = torch.stack(angle_base_clfs_new_feats)
    featmean_fc_angle_base = torch.stack(featmean_fc_angle_base)
    featmean_fc_angle_inc = torch.stack(featmean_fc_angle_inc)

    angle_intra_mean_base = torch.mean(intra_mean_base, dim=0)
    angle_intra_std_base = torch.mean(intra_std_base, dim=0)
    inter_angles_base = get_inter_angle(mean_base)
    angle_feat_fc_base = torch.mean(mean_feats_fc_angle_base, dim=0)
    angle_feat_fc_base_std = torch.mean(std_feats_fc_angle_base, dim=0)

    angle_intra_mean_inc = torch.mean(intra_mean_inc, dim=0)
    angle_intra_std_inc = torch.mean(intra_std_inc, dim=0)
    inter_angles_inc = get_inter_angle(mean_inc)
    angle_feat_fc_inc = torch.mean(mean_feats_fc_angle_inc, dim=0)
    angle_feat_fc_inc_std = torch.mean(std_feats_fc_angle_inc, dim=0)
    angle_base_feats_new_clf = torch.mean(angle_base_feats_new_clfs)
    angle_base_clfs_new_feat = torch.mean(angle_base_clfs_new_feats)

    base_inc_fc_angle = angle_btw_base_new(model.module.textual_classifier.weight[:args.base_class].cpu(),
                       model.module.textual_classifier.weight[ start_class : start_class + args.way].cpu())
    inc_inter_fc_angle = get_inter_angle(model.module.textual_classifier.weight[start_class : start_class + args.way].cpu())

    angle_featmean_fc_base = torch.mean(featmean_fc_angle_base, dim=0)
    angle_featmean_fc_inc = torch.mean(featmean_fc_angle_inc, dim=0)

    return inter_angles_base, angle_intra_mean_base, angle_intra_std_base, angle_feat_fc_base, angle_feat_fc_base_std,\
           inter_angles_inc, angle_intra_mean_inc, angle_intra_std_inc, angle_feat_fc_inc, angle_feat_fc_inc_std, \
           angle_base_feats_new_clf, angle_base_clfs_new_feat, base_inc_fc_angle, inc_inter_fc_angle, \
           angle_featmean_fc_base, angle_featmean_fc_inc



def get_intra_avg_angle_from_loader(args, loader, doubleaug, procD, clsD, model, normformean=False):
    session = procD['session']
    dd = tot_datalist(args, loader, model, doubleaug=doubleaug)
    cls = clsD['tasks'][session].cpu()

    d = []
    for i in cls:
        j = torch.where(i==dd[1])[0]
        if normformean == False:
            mean = torch.mean(dd[0][j],dim=0)
        else:
            mean = torch.mean(F.normalize(dd[0][j]), dim=0)
        d.append(mean)
    feats = torch.stack(d)
    sum = get_inter_angle(feats)
    return sum


def count_acc_topk(x, y, k=5):
    _, maxk = torch.topk(x, k, dim=-1)
    total = y.size(0)
    test_labels = y.view(-1, 1)
    # top1=(test_labels == maxk[:,0:1]).sum().item()
    topk = (test_labels == maxk).sum().item()
    return float(topk / total)


def count_acc_taskIL(logits, label, args):
    basenum = args.base_class
    incrementnum = (args.num_classes - args.base_class) / args.way
    for i in range(len(label)):
        currentlabel = label[i]
        if currentlabel < basenum:
            logits[i, basenum:] = -1e9
        else:
            space = int((currentlabel - basenum) / args.way)
            low = basenum + space * args.way
            high = low + args.way
            logits[i, :low] = -1e9
            logits[i, high:] = -1e9

    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()




def save_gradcam(img_pth, model, fc, idx2label, savename, tgt_layer):
    # Load the test image
    img = OmniImage(Image.open(img_pth).convert('RGB'))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])

    explainer = GradCAM(
        model=model,
        target_layer=tgt_layer,
        #target_layer=model.ln_post,
        preprocess_function=preprocess
    )
    # Explain the top label
    # save_image(torch.tensor(img.reshape(3,224,224)).float(), 'original_img.png')

    explanations = explainer.explain_iu(img, fc=fc)
    xx = explanations._plotly_figure(index=0, class_names=idx2label)
    xx.write_image(savename)
    print(savename, idx2label[explanations.explanations[0]['target_label']])




def get_image_attn_mask(img, att_mat, avg=False, layer=-1, token=0):
    """
    https://github.com/ricardodeazambuja/CLIP/blob/attn_weights/notebooks/Interacting_with_CLIP.ipynb
    image.shape => [img_dim, img_dim]
    grid => img_dim//patch_size
    token_seq_length => 1+grid**2, where the "1+" comes from the addition of CLS token
    att_mat.shape => [n_layers, n_heads, token_seq_length, token_seq_length]
    """

    # Heavily based on:
    # from https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    att_mat = att_mat.cpu()  # [n_layers, n_heads, token_seq_length, token_seq_length]

    # Average the attention weights across all heads.
    att_mat = att_mat.mean(dim=1)  # [n_layers, token_seq_length, token_seq_length]
    aug_att_mat = att_mat.float()  # torch.matmul won't work with fp16

    #
    # CLIP uses a series of ResidualAttentionBlocks that look like this:
    # input x => LN => ATTN => ADD x => LN => MLP => ADD x => output y
    #
    # But the input x is NOT a simple image at all. It is the result of
    # a conv2d with a kernel and stride the same size so it "chops" the image into
    # something like patches with a lot of layers (the token embedding dimension).
    # A class (CLS) token (learned and frozen for inference) is concatenated to the start and to the
    # whole "sequence" we add the positional encoder (again, learned and frozen for inference)
    #
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        tmp_mat = aug_att_mat[n] + joint_attentions[n-1]
        tmp_mat /= tmp_mat.norm()
        # ignoring the effects of the MLP...
        joint_attentions[n] = torch.matmul(tmp_mat, joint_attentions[n-1])

    # Attention from the output token to the input space.
    if avg:
        v = joint_attentions.mean(dim=0)
    else:
        v = joint_attentions[layer]

    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    att = torch.arange(att_mat.size(1))
    att = att[att != att[token]]
    mask = v[token, att].reshape(grid_size, grid_size)  # token=0 is CLS (the only one used by CLIP at the end)
    mask = np.asarray(Image.fromarray(mask.numpy()).resize(img.size)).copy()
    mask -= mask.min()
    mask /= mask.max()

    return mask




def confmatrix(logits, label, filename):
    font = {'family': 'FreeSerif', 'size': 18}
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({'font.family': 'FreeSerif', 'font.size': 18})
    plt.rcParams["font.family"] = "FreeSerif"

    pred = torch.argmax(logits, dim=1)
    cm = confusion_matrix(label, pred, normalize='true')
    # print(cm)
    clss = len(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    if clss <= 100:
        plt.yticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
        plt.xticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    elif clss <= 200:
        plt.yticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
        plt.xticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
    else:
        plt.yticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)
        plt.xticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)

    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + '.pdf', bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    cbar = plt.colorbar(cax)  # This line includes the color bar
    cbar.ax.tick_params(labelsize=16)
    if clss <= 100:
        plt.yticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
        plt.xticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    elif clss <= 200:
        plt.yticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
        plt.xticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
    else:
        plt.yticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)
        plt.xticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + '_cbar.pdf', bbox_inches='tight')
    plt.close()

    return cm


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



def interpret(image, texts, model, device, start_layer=1, start_layer_text=-1, no_txt_encode=False):

    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts, calc_logit=True, no_txt_encode=no_txt_encode)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
        # calculate index of last layer
        start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1:
        # calculate index of last layer
        start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
            continue
        #grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance

def show_image_relevance(image_relevance, image, orig_image):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image);
    axs[0].axis('off');

    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = np.roll(image_relevance, -10, 0)
    #image_relevance = np.roll(image_relevance, 20, 1)
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs[1].imshow(vis);
    axs[1].axis('off');


def show_heatmap_on_text(text, text_encoding, R_text):
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    print(text_scores)
    _tokenizer = _Tokenizer()
    text_tokens = _tokenizer.encode(text)
    text_tokens_decoded = [_tokenizer.decode([a]) for a in text_tokens]
    vis_data_records = [visualization.VisualizationDataRecord(text_scores, 0, 0, 0, 0, 0, text_tokens_decoded, 1)]
    visualization.visualize_text(vis_data_records)