import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.DSC import DSConv_pro, EncoderConv
from torch import cat


class DSC_AttentionBaseline(nn.Module):

    def __init__(
            self, channels=256, q_size=224, n_heads=8, n_head_channels=32,
            attn_drop=0.0, proj_drop=0.0, stride=1, use_AiA=True, kernel_size=9,
            extend_scope=1.0, if_offset=True, device: torch.device = "cuda",
    ):

        super().__init__()


        self.n_head_channels = channels // n_heads
        assert n_heads*self.n_head_channels==channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size, q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        # self.nc = n_head_channels * n_heads
        self.nc = channels
        self.stride = stride

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0)

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.use_AiA = use_AiA
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.device = device
        self.dsc_proj1 = nn.Linear(self.kv_h * self.kv_w, self.nc)
        self.dsc_proj2 = nn.Linear(self.nc, self.kv_h * self.kv_w)
        self.conv_0 = EncoderConv(self.nc, self.nc)
        self.conv_x = DSConv_pro(
            self.nc,
            self.nc,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv_y = DSConv_pro(
            self.nc,
            self.nc,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv_1 = EncoderConv(3 * self.nc, self.nc)


    def forward(self, x):
        x = x
        B, C, H, W = x.size()

        q = self.proj_q(x)
        # B C H W -> B C 1 H*W
        x_sampled = x.reshape(B, C, 1, H * W)
        # self.proj_k.weight = torch.nn.Parameter(self.proj_k.weight.float())
        # self.proj_k.bias = torch.nn.Parameter(self.proj_k.bias.float())
        # self.proj_v.weight = torch.nn.Parameter(self.proj_v.weight.float())
        # self.proj_v.bias = torch.nn.Parameter(self.proj_v.bias.float())
        # 检查权重的数据类型
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)

        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, H * W)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, H * W)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B*h, H*W, H*W
        attn = attn.mul(self.scale)

        # 动态可变形卷积层
        if self.use_AiA:
            feature_map = attn
            # B*h Q K -> B*h K Q
            feature_map = torch.transpose(feature_map, 1, 2)
            feature_map = self.dsc_proj1(feature_map)
            # B*h K Q -> B*h Q K
            feature_map = torch.transpose(feature_map, 1, 2)
            # B*h Q K -> B*h C H W
            feature_map = feature_map.reshape(B * self.n_heads, self.nc, H, W)
            # Encoder
            feature_map1 = self.conv_0(feature_map)
            # x方向卷积
            feature_map2 = self.conv_x(feature_map)
            # y方向卷积
            feature_map3 = self.conv_y(feature_map)
            # cat
            feature_map = self.conv_1(cat([feature_map1, feature_map2, feature_map3], dim=1))
            # B*h C H W -> B*h C H*W
            feature_map = feature_map.reshape(B * self.n_heads, self.nc, H*W)
            # B*h Q K -> B*h K Q
            feature_map = torch.transpose(feature_map, 1, 2)
            feature_map = self.dsc_proj2(feature_map)
            # B*h K Q -> B*h Q K
            attn = torch.transpose(feature_map, 1, 2)
            attn = attn + feature_map


        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y

# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     A = np.random.rand(1, 256, 8, 8)
#     A = A.astype(dtype=np.float32)
#     A = torch.from_numpy(A)
#     conv0 = DSC_AttentionBaseline(
#         q_size=8,
#         n_heads=8,
#         )
#     if torch.cuda.is_available():
#         A = A.to(device)
#         conv0 = conv0.to(device)
#     out = conv0(A)
#     print(out.shape)
#     print(out)