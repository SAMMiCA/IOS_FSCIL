import argparse

import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from vissl.utils.checkpoint import replace_module_prefix
from utils_s import *
from tqdm import tqdm


from clip_iu.clip import load
from src.utils import torch_load, torch_save
#import clip_iu.clip_iu as clip_iu
from clip_iu.clip import tokenize
from src.datasets.templates import get_templates
from src.datasets.registry import get_dataset
from src.modeling import ClassificationHead, ImageEncoder
from english_words import english_words_lower_set
from clip_iu import clip
from clip_iu.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict
from torch.optim.lr_scheduler import _LRScheduler
from clip_iu.model import ResidualAttentionBlock

import clip_iu.clip as clip

_tokenizer = _Tokenizer()

class MYNET(nn.Module):

    def __init__(self, args, fw_mode=None, textual_clf_weights=None):
        super().__init__()

        #self.new_mode = args.new_mode
        #self.temperature = args.temperature
        self.m = args.m
        self.s = args.s
        self.fw_mode = fw_mode
        self.base_class = args.base_class
        self.way = args.way
        self.num_classes = args.num_classes
        self.num_cls = args.num_classes

        self.base_freeze_backbone = args.base_freeze_backbone
        self.inc_freeze_backbone = args.inc_freeze_backbone

        self.prompt_mode = args.prompt_mode
        if self.prompt_mode == 'clspr':
            self.clspr_ver = args.clspr_ver


        _clip_model = CLIP_Model(args, keep_lang=True)
        self.clip_model = _clip_model.model

        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.model_type  = args.model_type

        if (self.prompt_mode == 'cocoop') or (self.prompt_mode =='zsl'):
            self.prompt_learner_cocoop = PromptLearner_cocoop(args.cfg, args.classnames, self.clip_model, args)
            #self.tokenized_prompts = self.prompt_learner_cocoop.tokenized_prompts
            _tokenized_prompts = self.prompt_learner_cocoop.tokenized_prompts
            self.register_buffer("tokenized_prompts",_tokenized_prompts)
        elif self.prompt_mode == 'clspr':
            self.prompt_learner_clspr = PromptLearner_clspr(args.cfg, args.classnames, self.clip_model, args)
            _tokenized_prompts = self.prompt_learner_clspr.tokenized_prompts
            self.register_buffer("tokenized_prompts", _tokenized_prompts)
            # cross-attention
            #self.cra = nn.MultiheadAttention(self.clip_model.visual.width, self.clip_model.visual.heads)
            # CLIP ViT-B/16 # self.clip_model.transformer.width -> 512
            self.cra = nn.MultiheadAttention(self.clip_model.transformer.width, args.n_heads)
            # multi-head self-attention
            #self.msa = nn.Sequential(
            #    *[ResidualAttentionBlock(self.clip_model.visual.width, self.clip_model.visual.heads, attn_mask=None) for _ in
            #      range(args.nlayer_msa)])
            self.msa = nn.Sequential(
                *[ResidualAttentionBlock(self.clip_model.transformer.width, args.n_heads, attn_mask=None) for _ in
                  range(args.nlayer_msa)])

            ctx_init = args.cfg.TRAINER.COCOOP.CTX_INIT  # default prompt e.g. ctx_init = 'a photo of a'
            classnames = [name.replace("_", " ") for name in args.classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [ctx_init + " " + name + "." for name in classnames]

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(tokenized_prompts)
            # embedding: (n_cls, 77, 512)
            # tokenized_prompts: (n_cls, 77)
            cls_pmt_init = self.text_encoder(embedding, tokenized_prompts)
            #self.register_buffer("cls_embed", cls_pmt_init)
            #self.cls_embed = cls_pmt_init.cuda() # SOS

            #self.cls_embed = nn.Parameter(cls_pmt_init)  #(n_cls, 512)
            self.register_buffer("cls_embed", nn.Parameter(cls_pmt_init))


        else:
            raise NotImplementedError



        #textual_classifier = get_classification_head(args, args.train_dataset)
        if self.clip_model is not None:
            self.train_preprocess = _clip_model.train_preprocess
            self.val_preprocess = _clip_model.val_preprocess
        self.textual_clf_weights = textual_clf_weights


        if args.model_type == 'ViT-L_14':
            self.num_features = 768
        elif args.model_type == 'ViT-B_16':
            self.num_features = 768
        elif args.model_type == 'ViT-B_32':
            self.num_features = 768
        elif args.model_type == 'RN50':
            self.num_features = 2048
        elif args.model_type == 'RN101':
            self.num_features = 2048
        elif args.model_type == 'RN50x4':
            self.num_features = 2048
        else:
            raise NotImplementedError



    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return torch_load(filename)

    def set_mode(self, fw_mode):
        self.fw_mode = fw_mode

    def update_text_clf(self, weights):
        self.textual_classifier = ClassificationHead(normalize=True, weights=weights)

    def get_text_classifiers(self, img, sess=None, train=False, encode=True):
        # img should be single image (e.g. 3-dim with H W ch)
        inputs = img.unsqueeze(0)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        if encode:
            image_features = self.image_encoder(inputs)

        else:
            image_features = inputs
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # (bsz, n_dim)

        if (self.prompt_mode == 'cocoop') or (self.prompt_mode =='zsl'):
            prompts = self.prompt_learner_cocoop(image_features) # (bsz, n_cls, n_tkn, n_dim) ex. (4,100,77,512)
            logits = []
            imf_i = inputs[0]
            pts_i = prompts[0]

            text_features = self.text_encoder(pts_i, tokenized_prompts) # (n_cls, n_dim)
            # tokenized_prompts used to find the location of eos within prompt
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features
        elif self.prompt_mode == 'clspr':
            logits = []
            if (self.prompt_mode=='clspr') and (self.clspr_ver=='mm_lin'):
                prompts = self.prompt_learner_clspr(image_features, sess, text_encoder=self.text_encoder)
            else:
                prompts = self.prompt_learner_clspr(image_features, sess)

            n_cls_cursess = self.base_class if sess == 0 else self.base_class + self.way * (sess)
            imf_i = image_features[0]
            pts_i = prompts[0]
            text_features = self.text_encoder(pts_i[:n_cls_cursess], tokenized_prompts[:n_cls_cursess])  # (n_cls_cursess n_dim)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features
        else:
            raise NotImplementedError

    def forward(self, inputs, sess=None, train=False, encode=True, labels=None):
        if train and (labels is not None):
            assert self.prompt_mode == 'clspr'

        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        if encode:
            image_features = self.image_encoder(inputs)

        else:
            image_features = inputs
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (bsz, n_dim)

        if (self.prompt_mode == 'cocoop') or (self.prompt_mode =='zsl'):
            prompts = self.prompt_learner_cocoop(image_features) # (bsz, n_cls, n_tkn, n_dim) ex. (4,100,77,512)
            logits = []
            for pts_i, imf_i in zip(prompts, image_features):
                text_features = self.text_encoder(pts_i, tokenized_prompts) # (n_cls, n_dim)
                # tokenized_prompts used to find the location of eos within prompt
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)
            logits = torch.stack(logits)
            n_cls_cursess = self.base_class if sess == 0 else self.base_class + self.way * (sess)
            logits = logits[:, :n_cls_cursess]
            #if self.prompt_learner.training:
            #    return F.cross_entropy(logits, label)
            return logits # (bsz, n_cls)
        elif self.prompt_mode == 'clspr':
            logits = []
            #bias = self.lin_net(image_features) # bias: (bsz, embed_dim)
            if train:
                img_q = image_features.unsqueeze(0)
                txt_k = self.cls_embed[labels].unsqueeze(0)
                txt_v = txt_k
                #img_attn = self.cra(img_q, txt_k, txt_v)[0]
                #img_attn = self.msa(img_attn)[0]
                img_attn = img_q.squeeze(0)
                if (self.prompt_mode=='clspr') and (self.clspr_ver=='mm_lin'):
                    prompts = self.prompt_learner_clspr(img_attn, sess, text_encoder=self.text_encoder) # (bsz, n_cls, 77, 512)
                else:
                    prompts = self.prompt_learner_clspr(img_attn, sess)  # (bsz, n_cls, 77, 512)

            else:
                """
                img_q = image_features.unsqueeze(0)
                img_attn_list = []
                for i in range(self.num_cls):
                    txt_k = self.cls_embed[i].repeat(inputs.shape[0],1).unsqueeze(0)
                    txt_v = txt_k
                    #img_attn = self.cra(img_q, txt_k, txt_v)[0]
                    #img_attn = self.msa(img_attn)[0]
                    img_attn = img_q.squeeze(0)
                    img_attn_list.append(img_attn)
                """
                if (self.prompt_mode=='clspr') and (self.clspr_ver=='mm_lin'):
                    #prompts = self.prompt_learner_clspr(img_attn_list, clswise=True, text_encoder=self.text_encoder)
                    prompts = self.prompt_learner_clspr(image_features, sess, text_encoder=self.text_encoder)
                else:
                    #prompts = self.prompt_learner_clspr(img_attn_list, clswise=True)
                    prompts = self.prompt_learner_clspr(image_features, sess)

            #for pts_i, imf_i, i in zip(prompts, img_attn, range(len(img_attn))):
            n_cls_cursess = self.base_class if sess == 0 else self.base_class + self.way * (sess)
            for pts_i, imf_i in zip(prompts, image_features):
                # prompts: (bsz, n_cls, n_tkn=77, emb_dim=512)
                #labels_i = labels[i]
                # THis was used, but it is wrong.
                # tokenized_prompts[labels_i.unsqueeze(0)]: (1,77)
                # Then only one eot loc is found, while we need eot loc for all classes.
                # So will be removed.
                #text_features = self.text_encoder(pts_i, tokenized_prompts[labels_i.unsqueeze(0)])  # (n_cls, n_dim)
                # pts_i: (n_cls, 77, 512), tokenized_prompts: (n_cls, 77)
                # tokenized~ used to find  loc of eot for each input(pts_i) for txt_encoder
                # .shape[0] of input for txt_encoder is like a batch, which is n_cls now.
                text_features = self.text_encoder(pts_i[:n_cls_cursess], tokenized_prompts[:n_cls_cursess])  # (n_cls_cursess n_dim)
                #text_features = self.text_encoder(pts_i, tokenized_prompts)  # (n_cls, n_dim)
                # tokenized_prompts used to find the location of eos within prompt
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = logit_scale * imf_i @  text_features.t()
                logits.append(l_i)
            logits = torch.stack(logits)
            #logits = logits[:, :n_cls_cursess]
            # if self.prompt_learner.training:
            #    return F.cross_entropy(logits, label)
            return logits  # (bsz, n_cls)
        else:
            raise NotImplementedError


class CLIP_Model(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f'Loading {args.model_type} pre-trained weights.')
        self.model, self.train_preprocess, self.val_preprocess = load(
            args.model_type, args.device, jit=False)

        self.keep_lang = keep_lang

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, texts=None):
        # usually use forward of self.model, which is already implemented
        # instead f forward here.
        if texts == None:
            return self.model.encode_image(images)
        else:
            return self.model.encode_image(images), self.model.encode_text(texts)


    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return torch_load(filename)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner_cocoop(nn.Module):
    def __init__(self, cfg, classnames, clip_model, args):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT # default prompt e.g. ctx_init = 'a photo of a'
        #dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        #cfg_imsize = cfg.INPUT.SIZE[0]
        #assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization5
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        self.prompt_mode = args.prompt_mode

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim) # ctx_dim ex: 512
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim) # n_ctx ex: 4 ( a photo of a )
        #print('bias:', bias)
        #print('prefix:',prefix)
        #print('suffix:', suffix)
        if self.prompt_mode == 'cocoop':
            ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)
        elif self.prompt_mode == 'zsl':
            ctx_shifted = ctx.repeat(bias.shape[0],1,1) # (batch, n_ctx, ctx_dim)
        else:
            raise NotImplementedError

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts  # maybe (iuyoon) (batch, n_cls, n_tkn, ctx_dim)

class PromptLearner_clspr(nn.Module):
    def __init__(self, cfg, classnames, clip_model, args):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT # default prompt e.g. ctx_init = 'a photo of a'
        #dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        self.text_encoder = TextEncoder(clip_model)
        #cfg_imsize = cfg.INPUT.SIZE[0]
        #assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"


        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        prompt_prefix = ctx_init


        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        #self.ctx = nn.Parameter(ctx_vectors)
        self.lin_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        """
        self.lin_net_key = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, args.key_dim)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(args.key_dim , args.key_dim))
        ]))
        """
        #"""
        self.lin_net_key = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, args.key_dim)),
        ]))
        #"""
        """
        self.lin_net_key = Simple_Res(vis_dim, args.key_dim)
        """

        self.merge_net = MergeNetwork(vis_dim, vis_dim//16, ctx_dim)
        self.prompt_mode = args.prompt_mode
        if self.prompt_mode == 'clspr':
            self.clspr_ver = args.clspr_ver

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts) # (n_cls, 77, 512)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS (n_cls, 1, 512)
        #self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # CLS, EOS
        #self.register_buffer("token_suffix", [embedding[count, i:, :] for count, i in enumerate(tokenized_prompts.argmax(dim=-1))])
        #self.token_suffix = [embedding[count, i-1:, :] for count, i in enumerate(tokenized_prompts.argmax(dim=-1))]
        n_tkn_suffix_min = torch.min(tokenized_prompts.argmax(dim=-1))
        #self.token_suffix = embedding[:, n_tkn_suffix_min-1:, :]
        self.register_buffer("token_suffix", embedding[:, n_tkn_suffix_min-1:, :])
        #  *** This is not the final version, since the length may vary by classes.
        # ** In the forward, have to slice to the right length via toke~.argmax(dim=-1)

        #self.clswise_pr_embed = embedding[:, 1:1+n_ctx+1, :]
        #self.clswise_pr_embed = [embedding[count, 1:i-1, :] for count, i in enumerate(tokenized_prompts.argmax(dim=-1))]
        #self.clswise_pr_embed = nn.ParameterList([nn.Parameter(self.clswise_pr_embed[i]) for i in range(len(self.clswise_pr_embed))])
        n_tkn_suffix_max = torch.max(tokenized_prompts.argmax(dim=-1))
        clswise_pr_embed = embedding[:, 1:n_tkn_suffix_max-1, :] #n_cls, n_tkn_suffix_max-2,512
        clswise_pr_embed_fix = copy.deepcopy(clswise_pr_embed)
        self.register_buffer("clswise_pr_embed_fix", clswise_pr_embed_fix)
        self.n_tkn_suffix_max = n_tkn_suffix_max
        self.n_tkn_suffix_min = n_tkn_suffix_min

        for sess in range(args.sessions):
            if sess == 0:
                _clswise_pr_embed = clswise_pr_embed[:args.base_class]
                _clswise_pr_embed = nn.Parameter(_clswise_pr_embed)
                setattr(self, f'clswise_pr_embed_{sess}', _clswise_pr_embed)
            else:
                start_idx = args.base_class + args.way * (sess-1)
                end_idx = args.base_class + args.way * (sess)
                _clswise_pr_embed = clswise_pr_embed[start_idx:end_idx]
                _clswise_pr_embed = nn.Parameter(_clswise_pr_embed)
                setattr(self, f'clswise_pr_embed_{sess}', _clswise_pr_embed)

        #self.clswise_pr_embed = nn.Parameter(self.clswise_pr_embed)
        _cp_clswise_pr_embed = copy.deepcopy(clswise_pr_embed)

        self.n_prpool_base = args.n_prpool_base
        self.n_prpool_inc = args.n_prpool_inc
        self.key_dim = args.key_dim
        self.n_prpool_total = args.n_prpool_base + args.n_prpool_inc * (args.sessions-1)
        """
        self.keys = copy.deepcopy(self.clswise_pr_embed)[:self.n_prpool_total]        
        with torch.no_grad():
            _embedding = copy.deepcopy(embedding) #(n_cls, 77, 512)
            self.keys = self.text_encoder(_embedding, tokenized_prompts)[:self.n_prpool_total]
        self.keys = nn.Parameter(self.keys)
        """
        #"""
        with torch.no_grad():
            #for i in range(n_cls):
            _embedding = copy.deepcopy(embedding) #(n_cls, 77, 512)
            #self.keys = self.text_encoder(_embedding, tokenized_prompts)[:self.n_prpool_total]
            #_keys = []
            _text_embeddings = self.text_encoder(_embedding, tokenized_prompts) #(n_cls, 512)
            for sess in range(args.sessions):
                if sess == 0:
                    _key_cls = _text_embeddings[:self.n_prpool_base]
                    _key_cls = nn.Parameter(_key_cls)
                    setattr(self, f'keys_{sess}', _key_cls)
                    #_keys.append(_key_cls)
                else:
                    #start_idx = self.n_prpool_base + self.n_prpool_inc * (sess-1)
                    start_idx = args.base_class + args.way * (sess-1)
                    end_idx = start_idx + self.n_prpool_inc
                    _key_cls = _text_embeddings[start_idx:end_idx]
                    _key_cls = nn.Parameter(_key_cls)
                    setattr(self, f'keys_{sess}', _key_cls)
                    #_keys.append(_key_cls)
            #self.keys = nn.ParameterList(_keys)
        # """

        #self.keys = nn.Parameter(torch.randn(self.n_prpool_total, args.key_dim)) # (n_cls, key_dim)
        #self.keys = nn.Linear(args.key_dim, n_cls)
        #self.key_prompts = self.keys
        """
        self.key_prompts = copy.deepcopy(self.clswise_pr_embed)[:self.n_prpool_total] # n_prpool_total, n_tkn_suffix_max-2, 512
        self.key_prompts = nn.Parameter(self.key_prompts)
        """
        # """

        #_key_prompts = []
        for sess in range(args.sessions):
            if sess == 0:
                _key_prompts_cls = _cp_clswise_pr_embed[:self.n_prpool_base] #n_prpool_base, n_tkn_suffix_max-2,512
                if not args.use_single_prompt:
                    _key_prompts_cls = nn.Parameter(_key_prompts_cls) #n_prpool_base, n_tkn_suffix_max-2,512
                else:
                    #_key_prompts_cls = _key_prompts_cls[:,-1,:].unsqueeze(dim=1).repeat(1, self.n_tkn_suffix_max-2, 1)
                    _key_prompts_cls = _key_prompts_cls[:, -1, :]
                    _key_prompts_cls = nn.Parameter(_key_prompts_cls) #n_prpool_base, 512
                setattr(self, f'key_prompts_{sess}', _key_prompts_cls)
                #_key_prompts.append(_key_prompts_cls)
            else:
                start_idx = args.base_class + args.way * (sess - 1)
                end_idx = start_idx + self.n_prpool_inc
                _key_prompts_cls = _cp_clswise_pr_embed[start_idx:end_idx]
                if not args.use_single_prompt:
                    _key_prompts_cls = nn.Parameter(_key_prompts_cls) # n_prpool_inc, n_tkn_suffix_max-2,512
                else:
                    #_key_prompts_cls = _key_prompts_cls[:,-1,:].unsqueeze(dim=1).repeat(1, self.n_tkn_suffix_max-2, 1)
                    _key_prompts_cls = _key_prompts_cls[:, -1, :]
                    _key_prompts_cls = nn.Parameter(_key_prompts_cls) #n_prpool_inc, 512
                setattr(self, f'key_prompts_{sess}', _key_prompts_cls)
                #_key_prompts.append(_key_prompts_cls)
        #self.key_prompts = nn.ParameterList(_key_prompts)
        # """

        #n_tkn_suffix_max-2: -2 stands for SOT and EOT.

        # Note that self.clswise_pr_embed are from min ~ max.
        self.key_tokenized_prompts = tokenized_prompts[:self.n_prpool_total]
        #self.key_prompts = nn.Parameter(torch.randn(self.n_prpool_total, self.token_prefix.shape[-1])) #(n_cls, 512)
        #  *** This is not the final version, since the length may vary by classes.
        # ** In the forward, have to slice to the right length via toke~.argmax(dim=-1)

        # may have differ in length. some classes have two-token embedding.

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor # (n_cls, n_tkn)
        self.name_lens = name_lens
        self.base_class = args.base_class
        self.way = args.way
        self.topk_pool = args.topk_pool
        self.sessions = args.sessions
        self.use_single_prompt = args.use_single_prompt

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        assert (prefix.shape[1] + ctx.shape[1] + suffix.shape[1]) == 77
        #if not ((prefix.shape[1] + ctx.shape[1] + suffix.shape[1])==77):
            #assert (prefix.shape[1] + ctx.shape[1] + suffix.shape[1])==76
            #suffix = suffix[:-1, :]

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (bsz, 1, dim)
                ctx,  # (bsz, n_ctx, dim) n_ctx may be 5 or 6 (4+ 1 or 2 for cls)
                suffix,  # (bsz, , *, dim)
            ],
            dim=1,
            #dim=0,
        )

        return prompts

    def forward(self, im_features, sess=None, text_encoder=None):
        # to the right number of tokens

        if text_encoder is not None:
            assert self.prompt_mode == 'clspr'
            assert self.clspr_ver == 'mm_lin'

        tkn_lens = self.tokenized_prompts.argmax(dim=-1)
        tkn_lens_min = torch.min(tkn_lens)
        tkn_lens_max = torch.max(tkn_lens)

        suffix = [self.token_suffix[i, tkn_lens[i]-tkn_lens_min:, :] for i in range(len(tkn_lens))]

        clswise_pr_embed = torch.cat([getattr(self, f'clswise_pr_embed_{k}') for k in range(self.sessions)],dim=0)
        _clswise_pr_embed = []
        clswise_pr_embed_fix = []
        #clswise_pr_embed
        for i in range(len(tkn_lens)):
            if not tkn_lens_max-tkn_lens[i] == 0:
                _clswise_pr_embed.append(clswise_pr_embed[i, :-(tkn_lens_max-tkn_lens[i]), :])
                clswise_pr_embed_fix.append(self.clswise_pr_embed_fix[i, :-(tkn_lens_max-tkn_lens[i]), :])
            else:
                # exception for tkn_lens[i]=tken_lens_max. :-0 is not we want.
                _clswise_pr_embed.append(clswise_pr_embed[i, :, :])
                clswise_pr_embed_fix.append(self.clswise_pr_embed_fix[i, :, :])
        prefix = self.token_prefix
        #suffix = [self.token_suffix[i] for i in range(len(self.token_suffix))]

        n_cls = len(clswise_pr_embed)
        bsz = im_features.shape[0]
        prompts = []
        if self.clspr_ver == 'none':
            # im_features.shape: (batch, ctx_dim)
            # ctx = self.ctx  # (n_ctx, ctx_dim)
            # clswise_pr_embed:  list len=n_cls, each 5or6, 512)
            for i in range(n_cls):
                pfx = prefix[i].repeat(bsz, 1, 1) # (batch, 1, 512)
                sfx = suffix[i].repeat(bsz, 1, 1)  # (batch, ~, 512)
                ctx = _clswise_pr_embed[i]  # (5or6, 512)
                ctx = ctx.unsqueeze(0)  # (1, 5or6, 512)
                ctx_s = ctx.repeat(bsz, 1, 1)
                temp_p = self.construct_prompts(ctx_s, pfx, sfx) # (bsz, 77, 512)
                prompts.append(temp_p)
        elif self.clspr_ver == 'img_lin':
            bias = self.lin_net(im_features)  # (bsz, ctx_dim) (ctx_dim e.g. 512)
            bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
            for i in range(n_cls):
                pfx = prefix[i].repeat(bsz, 1, 1)  # (batch, 1, 512)
                sfx = suffix[i].repeat(bsz, 1, 1)  # (batch, ~, 512)

                ctx = _clswise_pr_embed[i] #(5or6, 512)
                ctx = ctx.unsqueeze(0) #(1, 5or6, 512)
                ctx_s = ctx + bias #(batch, 5or6, 512)
                temp_p = self.construct_prompts(ctx_s, pfx, sfx) #(bsz, 77, 512)
                prompts.append(temp_p)
        elif self.clspr_ver == 'prpool':
            f_key = self.lin_net_key(im_features) #(bsz, key_dim)
            #n_cls_cursess = self.base_class if sess == 0 else self.base_class + self.way * (sess)
            n_prpool_cursess = self.n_prpool_base if sess==0 else self.n_prpool_base + self.n_prpool_inc * (sess)
            """
            keys = self.keys[:n_prpool_cursess, :] #(n_cls_cursess, key_dim)
            key_prompts = self.key_prompts
            """

            #""""
            #keys = torch.cat([self.keys[k] for k in range(sess+1)], dim=0) # (n_cls_cursess, key_dim)
            keys = torch.cat([getattr(self, f'keys_{k}') for k in range(sess + 1)], dim=0)  # (n_cls_cursess, key_dim)
            #key_prompts = torch.cat([self.key_prompts[k] for k in range(self.sessions)], dim=0) #(n_prpool_tot, key_dim)
            if not self.use_single_prompt:
                key_prompts = torch.cat([getattr(self, f'key_prompts_{k}') for k in range(self.sessions)],dim=0) # (n_prpool_tot, n_tkn_suffix_max-2, key_dim)
            else:
                key_prompts = torch.cat([getattr(self, f'key_prompts_{k}') for k in range(self.sessions)], dim=0)  # (n_prpool_tot,  key_dim)
                key_prompts = key_prompts.unsqueeze(dim=1).repeat(1, self.n_tkn_suffix_max-2, 1)

            #  because to be indexed (idx within n_prpool_cursess)
            #""""
            match_key = F.normalize(f_key,dim=1) @ F.normalize(keys,dim=1).t() # (bsz, n_cls_cursess) #
            topk_val, topk_idx = torch.topk(match_key ,self.topk_pool) # (bsz, topk_pool), (bsz, topk_pool)
            topk_val = F.softmax(topk_val, dim=1) #(bsz, topk_pool)
            #topk_prompts = self.key_prompts[topk_idx] #(bsz, topk_pool, 512)
            #weighted_pr = self.key_prompts[topk_idx] * topk_val.unsqueeze(2).repeat(1,1,self.key_prompts.shape[-1]) #(bsz, topk_pool, 512)

            #self.key_prompts # n_prpool_total, n_tkn_suffix_max-2, 512
            #self.key_prompts[topk_idx] # bsz, topk_pool, n_tkn_suffix_max-2, 512
            topk_val = topk_val.unsqueeze(2).unsqueeze(3) #(bsz, topk_pool, 1, 1)
            weighted_pr = key_prompts[topk_idx] * topk_val.repeat(1, 1, key_prompts.shape[-2], key_prompts.shape[-1])


            # (bsz, topk_pool, n_tkn_suffix_max-2, 512)
            bias = torch.sum(weighted_pr, dim=1) # (bsz, n_tkn_suffix_max-2, 512)
            #bias = torch.sum(weighted_pr, dim=1) #(bsz, 512)
            #bias = bias.unsqueeze(1) #(bsz, 1, 512=ctx_dim)
            for i in range(n_cls):
                pfx = prefix[i].repeat(bsz, 1, 1)  # (batch, 1, 512)
                sfx = suffix[i].repeat(bsz, 1, 1)  # (batch, ~, 512)

                ctx = _clswise_pr_embed[i] #(5or6, 512)
                ctx = ctx.unsqueeze(0) #(1, 5or6, 512)

                if not tkn_lens_max - tkn_lens[i] == 0:
                    _bias = bias[:, :-(tkn_lens_max -tkn_lens[i]) , :]
                else:
                    _bias = bias

                #ctx_s = ctx + bias #(batch, 5or6, 512), bias: bsz, 5or6, 512
                ctx_s = ctx + _bias #ctx shape (1, 5or6, 512), bias shape (bsz,
                temp_p = self.construct_prompts(ctx_s, pfx, sfx) #(bsz, 77, 512)
                prompts.append(temp_p)
            #match_key = self.keys(f_key) # (n_cls)

        else:
            raise NotImplementedError
        prompts = torch.stack(prompts)
        prompts = prompts.transpose(0,1) # (bsz, n_cls, 77, 512)
        return prompts



def word2textembed(word, args, model, template, device, v2=False):
    texts = []
    for t in template:
        texts.append(t(word))
    texts = tokenize(texts).to(device)  # tokenize   #prompts x sentence -> #prompts x 77 (prompt embedding dimension)
    embeddings, _ = model.encode_text(texts)  # embed with text encoder  #prompts x prompt_embed_dim -> #prompts x embed_dim
    if not v2:
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.mean(dim=0, keepdim=True) #1 x embed_dim
        embeddings /= embeddings.norm()
    else:
        embeddings = embeddings.mean(dim=0, keepdim=True)  # 1 x embed_dim
        embeddings = F.normalize(embeddings, dim=1)
    return embeddings


def temptokenize(args, template, word):
    texts = []
    for t in template:
        texts.append(t(word))
    texts = tokenize(texts).cuda()  # tokenize   #prompts x sentence -> #prompts x 77 (prompt embedding dimension)
    #texts = texts.mean(dim=0) #1 x embed_dim
    text = texts[0]      ################ Note that in this version only use first template.
    # this is originated that temptokenize is just developed to test the whether flyp code works.
    # Patch if needed
    return text

def lab_text_2weights(model, args, template, device, words):
    # origin from patch. I changed the function name but almost similar
    # Biggest differ point from v2 is the train<>eval (with no_grad existence) and *=logit_scale.exp() existence
    logit_scale = model.logit_scale

    model.eval()
    model.to(device)
    num_words = len(words)

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        #for i in range(args.num_classes):
        for i in range(num_words):
        #for classname in tqdm(dataset.classnames):
            classname = words[i]
            embeddings = word2textembed(classname, args, model, template, device).squeeze() # (1,ch_dim) -> (ch_dim)
            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights *= logit_scale.exp()
    return zeroshot_weights

def lab_text_2weights_v2(model, args, template, device, words):
    # origin from patch. I changed the function name but almost similar
    # Biggest differ point from v2 is the train<>eval (with no_grad existence) and *=logit_scale.exp() existence

    logit_scale = model.logit_scale

    model.train()
    #model.to(device)
    num_words = len(words)

    print('Building classification head.')
    zeroshot_weights = []
    #for i in range(args.num_classes):
    for i in range(num_words):
    #for classname in tqdm(dataset.classnames):
        classname = words[i]
        embeddings = word2textembed(classname, args, model, template, device, v2=True).squeeze() # (1,ch_dim) -> (ch_dim)
        zeroshot_weights.append(embeddings)

    #zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=0)
    return zeroshot_weights



def get_words_weights(args, clip_model, words):

    print(f'Did not find classification head for '
          f'{args.model_type} on {args.dataset} at {args.text_clf_weight_fn}, building one from scratch.')

    template = get_templates(args.dataset)
    zeroshot_weights = lab_text_2weights(clip_model, args, template, args.device, words)
    return zeroshot_weights


class MergeNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MergeNetwork, self).__init__()
        # Individual subnetwork for input vector A
        self.subnetwork_A = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Individual subnetwork for input vector B
        self.subnetwork_B = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        # Merge layer
        self.merge_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size),
            nn.ReLU()
        )
        self.merge_layer2 = nn.Sequential(
            nn.Linear(input_size * 2, 2*hidden_size),
            nn.ReLU()
        )
        # Output layer
        #self.output_layer = nn.Linear(output_size, output_size)
        self.output_layer = nn.Linear(2*hidden_size, output_size)

    def forward(self, input_A, input_B):
        merged_output = torch.cat((input_A, input_B), dim=1)
        merged_output = self.merge_layer2(merged_output)

        # Pass input A through subnetwork A
        #output_A = self.subnetwork_A(input_A)

        # Pass input B through subnetwork B
        #output_B = self.subnetwork_B(input_B)

        # Merge the outputs of subnetwork A and B
        #merged_output = torch.cat((output_A, output_B), dim=1)  # Concatenation

        # Pass the merged output through the merge layer
        #merged_output = self.merge_layer(merged_output)

        # Pass the merged output through additional layers if needed
        #merged_output = self.additional_layers(merged_output)
        # Generate the final output vector
        output = self.output_layer(merged_output)
        return output



class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        torch_save(self, filename)

    def load_weight(self, weight):
        print(f'Loading weight for finetune model')
        self.weight = weight

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return torch_load(filename)

class Simple_Res(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim

        self.num_layers = num_layers
        #self.layer = nn.Sequential(OrderedDict([
        #    ("linear1", nn.Linear(in_dim, hidden_dim)),
       #    ("relu", nn.ReLU(inplace=True)),
        #    ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        #]))
        #"""
        # Cause error when data number is not multiple of bsz
        # Maybe because of the Batchnorm
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
            #nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )
        #"""
    def forward(self, x):
        res = x
        x = self.layer1(x)
        x = self.layer2(x)
        x += res
        return x

class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, eyes=False):
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
        if eyes:
            nn.init.eye_(self.layer1[0].weight)
            nn.init.eye_(self.layer2[0].weight)
            nn.init.eye_(self.layer3[0].weight)

    def forward(self, x):
        if self.num_layers == 1:
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)

class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


