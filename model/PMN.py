import model.resnet as resnet
import torch
import math
import random
import numpy as np
from torch import nn
import torch.nn.functional as F


class similarity_func(nn.Module):
    def __init__(self):
        super(similarity_func, self).__init__()

    def forward(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)
        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

class masked_average_pooling(nn.Module):
    def __init__(self):
        super(masked_average_pooling, self).__init__()

    def forward(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

class proto_generation(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_average_pooling = masked_average_pooling()

    def forward(self, fea_list, mask_list):
        feature_fg_protype_list = []
        feature_bg_protype_list = []

        for k in range(len(fea_list)):
            feature_fg_protype = self.masked_average_pooling(fea_list[k], (mask_list[k] == 1).float())[None, :]
            feature_bg_protype = self.masked_average_pooling(fea_list[k], (mask_list[k] == 0).float())[None, :]
            feature_fg_protype_list.append(feature_fg_protype)
            feature_bg_protype_list.append(feature_bg_protype)

        # average K foreground prototypes and K background prototypes
        FP = torch.mean(torch.cat(feature_fg_protype_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_protype_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        return FP, BP


class SSP_func(nn.Module):
    def __init__(self, channal):
        super(SSP_func, self).__init__()
        self.channal = channal
        self.tau = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_q, out):
        bs = feature_q.shape[0]
        pred_1 = out.softmax(1).view(bs, 2, -1)
        pred_fg, pred_bg = pred_1[:, 1], pred_1[:, 0]

        fg_ls = []
        bg_ls = []

        for epi in range(bs):
            fg_thres = self.sigmoid(self.tau)
            bg_thres = 1 - fg_thres
            cur_feat = feature_q[epi].view(self.channal, -1)

            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]                # .mean(-1)
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]  # .mean(-1)

            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]                # .mean(-1)
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]  # .mean(-1)

            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)

            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

        # global proto
        fg_proto = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        bg_proto = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        return fg_proto, bg_proto


class Enhence(nn.Module):
    def __init__(self, dim):
        super(Enhence, self).__init__()

        self.similarity_func = similarity_func()
        self.ssp_func = SSP_func(dim)

    def forward(self, supp_fp, supp_bp, query_fea):

        # step1: 用support的原型得到query的第一次分割结果
        query_out = self.similarity_func(query_fea, supp_fp, supp_bp)

        # step2: 把query_out当做伪标签，利用ssp得到query的原型
        query_fp, query_bp = self.ssp_func(query_fea, query_out)

        # step3: 使用query的原型对query的特征进行引导
        activated_map = F.cosine_similarity(query_fea, query_fp, dim=1).unsqueeze(1)
        deactivated_map = F.cosine_similarity(query_fea, query_bp, dim=1).unsqueeze(1)
        activated_map = (activated_map - activated_map.min()) / (activated_map.max() - activated_map.min())
        deactivated_map = (deactivated_map - deactivated_map.min()) / (deactivated_map.max() - deactivated_map.min())
        query_fea = query_fea * activated_map + query_fea * (1 - deactivated_map)
        return query_fea, query_fp, query_bp

class GlobalContextBlock(nn.Module):
    def __init__(self, in_channel, patch_size, token_length, lora_dim=16):
        super(GlobalContextBlock, self).__init__()
        self.embed_dims = in_channel
        self.token_length = token_length
        self.lora_dim = lora_dim
        self.patch_size = patch_size

        # Learnable tokens
        self.learnable_tokens_a = nn.Parameter(torch.empty([self.token_length, self.lora_dim]))
        self.learnable_tokens_b = nn.Parameter(torch.empty([self.lora_dim, self.embed_dims * self.patch_size * self.patch_size]))

        nn.init.kaiming_uniform_(self.learnable_tokens_a)
        nn.init.kaiming_uniform_(self.learnable_tokens_b)

        self.lora_q = nn.Sequential(
            nn.Linear(self.embed_dims * self.patch_size * self.patch_size, self.embed_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dims, self.embed_dims * self.patch_size * self.patch_size),
        )

        self.lora_k = nn.Sequential(
            nn.Linear(self.embed_dims * self.patch_size * self.patch_size, self.embed_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dims, self.embed_dims * self.patch_size * self.patch_size),
        )

        self.lora_v = nn.Sequential(
            nn.Linear(self.embed_dims * self.patch_size * self.patch_size, self.embed_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dims, self.embed_dims * self.patch_size * self.patch_size),
        )

        # Unfold for patch extraction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward_delta_feat(self, feats, tokens):
        feats_q  = self.lora_q(feats)
        tokens_k = self.lora_k(tokens)
        tokens_v = self.lora_v(tokens)
        attn = torch.einsum("bnc,mc->bnm", feats_q, tokens_k)
        attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum("bnm,mc->bnc", attn, tokens_v)
        feats = delta_f * feats
        return feats

    def forward(self, x):

        # Global context
        batch_size, channels, height, width = x.size()

        # Patch extraction
        out = self.unfold(x)  # Use nn.Unfold
        out = out.view(batch_size, channels, self.patch_size * self.patch_size, -1)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(batch_size, -1, self.patch_size * self.patch_size * channels)

        learnable_tokens = self.learnable_tokens_a @ self.learnable_tokens_b

        # Feature enhancement
        delta_feat = self.forward_delta_feat(out, learnable_tokens)
        delta_feat = delta_feat.view(batch_size, -1, channels, self.patch_size, self.patch_size)
        delta_feat = delta_feat.permute(0, 2, 1, 3, 4).contiguous()

        # Reshape and combine with input
        delta_feat = delta_feat.view(batch_size, channels, -1, self.patch_size * self.patch_size)
        delta_feat = delta_feat.permute(0, 1, 3, 2).contiguous()
        delta_feat = delta_feat.view(batch_size, channels, height, width)
        return self.alpha * delta_feat + self.beta * x


class IFA_MatchingNet(nn.Module):
    def __init__(self, backbone, shot=1):
        super(IFA_MatchingNet, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3

        self.shot = shot
        self.similarity_func = similarity_func()
        self.freq_attn_0 = GlobalContextBlock(in_channel=128, patch_size=10, token_length=100)
        self.freq_attn_1 = GlobalContextBlock(in_channel=256, patch_size=10, token_length=100)
        self.proto_generation = proto_generation()
        self.enhence_2 = Enhence(512)
        self.enhence_3 = Enhence(1024)

    def forward(self, img_s_list, mask_s_list, img_q, mask_q):

        b, c, h, w = img_q.shape

        # feature maps of support images
        supp_feat_2_list = []
        supp_feat_3_list = []

        for k in range(len(img_s_list)):
            with torch.no_grad():
                s_0 = self.layer0(img_s_list[k])                       #  s_0: [128, 100, 100]
                s_1 = self.layer1(s_0)                                 #  s_1: [256, 100, 100]
            s_2 = self.layer2(s_1)                                     #  s_2: [512,  50,  50]
            s_3 = self.layer3(s_2)                                     #  s_3: [1024, 50,  50]

            supp_feat_2_list.append(s_2)
            supp_feat_3_list.append(s_3)

        supp_fp_2, supp_bp_2 = self.proto_generation(supp_feat_2_list, mask_s_list)
        supp_fp_3, supp_bp_3 = self.proto_generation(supp_feat_3_list, mask_s_list)

        with torch.no_grad():
            q_0 = self.layer0(img_q)
        q_0 = self.freq_attn_0(q_0)

        with torch.no_grad():
            q_1 = self.layer1(q_0)
        q_1 = self.freq_attn_1(q_1)

        q_2 = self.layer2(q_1)
        q_2, query_fp_2, query_bp_2 = self.enhence_2(supp_fp_2, supp_bp_2, q_2)

        q_3 = self.layer3(q_2)
        q_3, query_fp_3, query_bp_3 = self.enhence_3(supp_fp_3, supp_bp_3, q_3)

        query_out = self.similarity_func(q_3, query_fp_3, query_bp_3)
        query_out = F.interpolate(query_out, size=(h, w), mode="bilinear", align_corners=True)             # q_out:[B, 2, 400, 400]
        return query_out