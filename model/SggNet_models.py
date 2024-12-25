import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from model.MobileNetV2 import mobilenet_v2
from torch.nn import Parameter
import math
from timm.models.layers import DropPath, trunc_normal_


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.PReLU(out_channel))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)


class HSSM_linear_new(nn.Module):
    def __init__(self, dim1=320, num_heads=5, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=7):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.pool_ratio = pool_ratio
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AdaptiveAvgPool2d((pool_ratio,pool_ratio))
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.norm_pre = nn.LayerNorm(dim1)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, h, w = x.size()
        y = x.reshape(B, C, -1).permute(0, 2, 1)
        y = self.norm_pre(y)
        q = self.q(y).reshape(B, h*w, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        x_ = y.permute(0, 2, 1).reshape(B, C, h, w)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)

        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, h*w, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        out = out.permute(0, 2, 1).reshape(B, C, h, w)
        return x + out


# class HSSM_linear(nn.Module):
#     def __init__(self, dim1, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=7):
#         super().__init__()
#         assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."
#
#         self.dim1 = dim1
#         self.pool_ratio = pool_ratio
#         self.num_heads = num_heads
#         head_dim = dim1 // num_heads
#
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
#         self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
#         self.v = nn.Linear(dim1, dim1, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim1, dim1)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.pool = nn.AdaptiveAvgPool2d((pool_ratio,pool_ratio))
#         self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
#         self.norm = nn.LayerNorm(dim1)
#         self.norm_pre = nn.LayerNorm(dim1)
#         self.act = nn.GELU()
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         B, C, h, w = x.size()
#         x = x.reshape(B, C, -1).permute(0, 2, 1)
#         x = self.norm_pre(x)
#         q = self.q(x).reshape(B, h*w, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#         x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
#         x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
#         x_ = self.norm(x_)
#         x_ = self.act(x_)
#
#         k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
#         v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, h*w, C)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         x = x.permute(0, 2, 1).reshape(B, C, h, w)
#         return x


class HSSM(nn.Module):  # high-level sematic self-attention Moudule
    def __init__(self,dim_in, dw_sr, pw_cr):
        super(HSSM, self).__init__()
        assert dim_in % pw_cr == 0

        self.pw_cr = pw_cr
        self.scale = dim_in ** -0.5
        self.Dw_sr = convbnrelu(dim_in, dim_in*2, k=dw_sr, p=0, s=dw_sr, g=dim_in)
        self.Dw = convbnrelu(dim_in, dim_in, k=3, g=dim_in)
        self.Pw_cr_k = convbnrelu(dim_in, dim_in//pw_cr, k=1, s=1, p=0)
        self.Pw_cr_q = convbnrelu(dim_in, dim_in // pw_cr, k=1, s=1, p=0)
        self.Pw = convbnrelu(dim_in, dim_in, k=1, s=1,p=0)
        self.norm = nn.LayerNorm(dim_in)
        self.proj = nn.Linear(dim_in, dim_in)

    def forward(self, x):
        B, C, H, W = x.shape

        v, k = self.Dw_sr(x).chunk(2, dim=1)
        q = self.Dw(x)

        q = self.Pw_cr_q(q).reshape(B, C // self.pw_cr, -1).permute(0, 2, 1)
        k = self.Pw_cr_k(k).reshape(B, C // self.pw_cr, -1).permute(0, 2, 1)
        v = self.Pw(v).reshape(B, C, -1).permute(0, 2, 1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = self.proj(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return out


# Sematic-Guided Edge Awareness Moudule
class SEAM(nn.Module):
    def __init__(self, channel1=16, channel2=24, channel3=320):
        super(SEAM, self).__init__()
        self.channel_out = channel2

        self.smooth_1 = DSConv3x3(channel1, self.channel_out, stride=1, dilation=1)  # channel1-> channel2

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.smooth_2 = DSConv3x3(channel2, self.channel_out, stride=1, dilation=1)  # channel2-> channel2

        self.upsample_3 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.smooth_3 = DSConv3x3(channel3, self.channel_out, stride=1, dilation=1)  # 3channel-> channel2

        self.conv_all = DSConv3x3(self.channel_out, self.channel_out,stride=1,dilation=1)

        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1x1 = nn.Conv2d(self.channel_out, self.channel_out, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.channel_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, xs):
        x11 = self.smooth_1(x1)
        x22 = self.upsample_2(self.smooth_2(x2))
        x33 = self.upsample_3(self.smooth_3(xs))
        x1s = x11 * x33
        x2s = x22 * x33
        x = x1s + x2s

        x = self.conv_all(x)

        tmp = x - self.avg_pool(x)
        edge = self.sigmoid(self.bn1(self.conv_1x1(tmp)))
        out = edge * x + x
        return out, tmp


class SEAM_aba(nn.Module):
    def __init__(self, channel1=16, channel2=24, channel3=320):
        super(SEAM_aba, self).__init__()
        self.channel_out = channel2

        self.smooth_1 = DSConv3x3(channel1, self.channel_out, stride=1, dilation=1)  # channel1-> channel2

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.smooth_2 = DSConv3x3(channel2, self.channel_out, stride=1, dilation=1)  # channel2-> channel2

        self.conv_all = DSConv3x3(self.channel_out, self.channel_out,stride=1,dilation=1)

    def forward(self, x1, x2):
        x11 = self.smooth_1(x1)
        x22 = self.smooth_2(self.upsample_2(x2))
        x = x11 + x22

        x = self.conv_all(x)

        return x


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):  # num_state=384 num_node=16
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):  # x [16,384,16]
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class GRAM_linear(nn.Module):
    def __init__(self, num_in=96, plane_mid=64, mids=4):
        super(GRAM_linear, self).__init__()

        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out


# Graph-based Region Awareness Module
class GRAM(nn.Module):
    def __init__(self, inchannel8=32, inchannel16=96, dim_mid=64, mids=4):
        super(GRAM, self).__init__()
        self.num_s = int(dim_mid)
        self.num_n = (mids * mids)
        self.DSconv8 = DSConv3x3(inchannel8, self.num_n)
        self.DSconv16 = DSConv3x3(inchannel16, self.num_s)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = DSConv3x3(self.num_s, inchannel16)

    def forward(self, x8, x16):
        n, c, h, w = x16.size()
        x8_16 = F.interpolate(x8,(x16.size()[-2],x16.size()[-1]))  # Down-sampling

        x16_state_reshaped = self.DSconv16(x16).view(n, self.num_s, -1)
        x8_proj = self.DSconv8(x8_16).view(n, self.num_n, -1)
        x8_proj = torch.nn.functional.softmax(x8_proj, dim=1)

        x8_rproj = x8_proj

        x_n_state = torch.matmul(x16_state_reshaped, x8_proj.permute(0, 2, 1))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x8_rproj)
        x_state = x_state_reshaped.view(n, self.num_s, *x16.size()[2:])
        out = x16 + (self.conv_extend(x_state))

        return out


class SalHead(nn.Module):
    def __init__(self, in_channel, scale=1):
        super(SalHead, self).__init__()
        if scale > 1:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Identity()
        self.conv = nn.Sequential(
            nn.Dropout2d(p=0),
            nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.conv(self.up(x))

# class Decoder(nn.Module):
#     def __init__(self, channel_bone, channel_pre, Up_scale, head_scale=2):
#         super(Decoder, self).__init__()
#         self.Up = nn.Upsample(scale_factor=Up_scale, mode='bilinear', align_corners=True)
#         self.smooth_pre = DSConv3x3(channel_pre, channel_bone, stride=1)
#         # self.smooth_bo = DSConv3x3(channel_bone, channel_bone, stride=1)
#         self.smooth_all = DSConv3x3(channel_bone*2, channel_bone*2, stride=1)
#         self.head = SalHead(channel_bone*2, head_scale)
#
#     def forward(self, x_bone, x_pre):
#         xpre_smooth = self.smooth_pre(self.Up(x_pre))
#         # x_bone_smooth = self.smooth_bo(x_bone)
#         out = self.smooth_all(torch.cat([x_bone, xpre_smooth], dim=1))
#         return out, self.head(out)  # Supervision


class Decoder5(nn.Module):
    def __init__(self, channel_bone=96, channel_pre=320, chnnel_out=96, Up_scale=2, head_scale=8):
        super(Decoder5, self).__init__()
        self.smooth_pre = DSConv3x3(channel_pre, channel_bone, stride=1)
        self.smooth_cat = DSConv3x3(channel_bone * 2, channel_bone * 2, stride=1)
        self.smooth_up = DSConv3x3(channel_bone * 2, chnnel_out, stride=1)
        self.Up = nn.Upsample(scale_factor=Up_scale, mode='bilinear', align_corners=True)
        self.head = SalHead(chnnel_out, head_scale)

    def forward(self, x_bone, x_pre):
        pre = self.smooth_pre(x_pre)
        up =self.Up(self.smooth_cat(torch.cat([x_bone, pre],dim=1)))
        out = self.smooth_up(up)
        return out, self.head(out)  # Supervision


class Decoder34(nn.Module):
    def __init__(self, channel_pre=96, chnnel_out=48, Up_scale=4, head_scale=2):
        super(Decoder34, self).__init__()
        self.smooth_pre = DSConv3x3(channel_pre, channel_pre, stride=1)
        self.smooth_cat = DSConv3x3(channel_pre, channel_pre, stride=1)
        self.smooth_up = DSConv3x3(channel_pre, chnnel_out, stride=1)
        self.Up = nn.Upsample(scale_factor=Up_scale, mode='bilinear', align_corners=True)
        self.head = SalHead(chnnel_out, head_scale)

    def forward(self, x_pre):
        pre = self.smooth_pre(x_pre)
        up =self.Up(self.smooth_cat(pre))
        out = self.smooth_up(up)
        return out, self.head(out)  # Supervision


class Decoder12(nn.Module):
    def __init__(self, channel_bone=24, channel_pre=48, chnnel_out=24, Up_scale=2, head_scale=1):
        super(Decoder12, self).__init__()
        self.smooth_pre = DSConv3x3(channel_pre, channel_bone, stride=1)
        self.smooth_cat = DSConv3x3(chnnel_out*2, chnnel_out*2, stride=1)
        self.smooth_up = DSConv3x3(chnnel_out*2, chnnel_out, stride=1)
        self.Up = nn.Upsample(scale_factor=Up_scale, mode='bilinear', align_corners=True)
        self.head = SalHead(chnnel_out, head_scale)

    def forward(self, x_bone, x_pre):
        pre = self.smooth_pre(x_pre)
        up =self.Up(self.smooth_cat(torch.cat([x_bone, pre], dim=1)))
        out = self.smooth_up(up)
        return out, self.head(out)  # Supervision


class SggNet(nn.Module):
    def __init__(self, ckpt_path):
        super(SggNet, self).__init__()
        # Backbone model

        self.backbone = mobilenet_v2(False)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            self.backbone.load_state_dict(ckpt, strict=False)  # load
            print("loaded imagenet pretrained mobilenetv2")
        # input 288*288*3
        # conv1 144*144*16
        # conv2 72*72*24
        # conv3 36*36*32
        # conv4 18*18*96
        # conv5 9*9*320

        self.sigmoid = nn.Sigmoid()

        self.out45 = Decoder5()
        self.out34 = Decoder34()
        self.out12 = Decoder12()

        self.hssm = HSSM_linear_new(dim1=320, num_heads=5, pool_ratio=7)
        # self.hssm = HSSM(dim_in=320, dw_sr=2, pw_cr=10)
        self.Up_Fs_2x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.seam = SEAM()
        self.gram = GRAM_linear()
        # self.gaam = GRAM()

        self.edge_head = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                       convbnrelu(in_channel=24, out_channel=24, g=24),
                                       SalHead(24, 1))

    def forward(self, input):
        # generate backbone features
        conv1, conv2, conv3, conv4, conv5 = self.backbone(input)

        gaam_attn = self.gram(conv4, conv3)  # linear
        # gaam_attn = self.gram(conv3, conv4)   # cnn
        Fs = self.hssm(conv5)
        out45, super_45 = self.out45(gaam_attn, self.Up_Fs_2x(Fs))
        out34, super_34 = self.out34(out45)
        seam_attn, edge = self.seam(conv1, conv2, Fs)
        out12, super_12 = self.out12(seam_attn, out34)
        edge = self.edge_head(edge)

        return super_12, super_34, super_45, self.sigmoid(super_12), self.sigmoid(super_34), self.sigmoid(super_45), edge


if __name__ == '__main__':
    from thop import profile
    model = SggNet(None)
    x = torch.rand([1, 3, 288, 288])
    flops, params = profile(model, (x,))
    print(f"Flops: {flops / 1e9:.4f} GFlops")
    print(f"Params: {params / 1e6:.4f} MParams")