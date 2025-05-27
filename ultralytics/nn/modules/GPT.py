import math
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.modules.DSC import DSConv_pro, EncoderConv
from torch import cat

class SnakeConv(nn.Module):

    def __init__(self, kernel_size=9, vert_anchors=8, horz_anchors=8, extend_scope=1.0, if_offset=True, device="cuda"):
        super(SnakeConv, self).__init__()
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.device = device
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.dsc_proj1 = nn.Linear(vert_anchors*horz_anchors, vert_anchors*horz_anchors)
        self.dsc_proj2 = nn.Linear(vert_anchors*horz_anchors, vert_anchors*horz_anchors)
        self.conv_0 = EncoderConv(vert_anchors*horz_anchors, vert_anchors*horz_anchors)
        self.conv_x = DSConv_pro(
            vert_anchors*horz_anchors,
            vert_anchors*horz_anchors,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            self.device,
        )
        self.conv_y = DSConv_pro(
            vert_anchors*horz_anchors,
            vert_anchors*horz_anchors,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            self.device,
        )
        self.conv_1 = EncoderConv(3 * vert_anchors*horz_anchors, vert_anchors*horz_anchors)

    def forward(self, x):
        b_s, h, nq, nk = x.size()
        x = x.reshape(b_s * h, nq, nk)
        x = torch.transpose(x, 1, 2)
        x = self.dsc_proj1(x)
        # B*h K Q -> B*h Q K
        x = torch.transpose(x, 1, 2)
        x = x.reshape(b_s * h, self.vert_anchors*self.horz_anchors, self.vert_anchors, self.horz_anchors)
        x1 = self.conv_0(x)
        x2 = self.conv_x(x)
        x3 = self.conv_y(x)
        x = self.conv_1(cat([x1, x2, x3], dim=1))
        x = x.reshape(b_s * h, nq, nk)
        # B*h Q K -> B*h K Q
        x = torch.transpose(x, 1, 2)
        x = self.dsc_proj2(x)
        # B*h K Q -> B*h Q K
        x = torch.transpose(x, 1, 2)
        x = x.reshape(b_s, h, nq, nk)
        return x


# Transformer
class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1, use_AiA=True, vert_anchors=8, horz_anchors=8):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

        self.use_AiA = use_AiA
        if self.use_AiA:
            self.sc = SnakeConv(9, vert_anchors, horz_anchors)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(
            0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(
            0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(
            0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        if self.use_AiA:
            att = att + self.sc(att)


        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(
            b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, use_AiA, vert_anchors, horz_anchors):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop, use_AiA, vert_anchors, horz_anchors)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=4, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, use_AiA=True):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(
            1,  vert_anchors * horz_anchors, self.n_embd))

        # transformer
        lst = [1 if i == 7 else 0 for i in range(8)]
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, use_AiA and lst[layer]==1, vert_anchors, horz_anchors)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d(
            (self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """

        rgb_fea = x  # rgb_fea (tensor): dim:(B, C, H, W)
        # ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        # print("-----GPT in shape-----", rgb_fea.size(), ir_fea.size())
        # assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        # ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        # ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        # token_embeddings = torch.cat(
        #     [rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = rgb_fea_flat
        # token_embeddings = token_embeddings.permute(
        #     0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)
        token_embeddings = token_embeddings.permute(
            0, 2, 1).contiguous()  # dim:(B, H*W, C)

        # transformer
        # sum positional embedding and token    dim:(B, 2*H*W, C)
        token_embeddings = self.drop(self.pos_emb + token_embeddings)
        token_embeddings = self.trans_blocks(token_embeddings)  # dim:(B, 2*H*W, C)

        # decoder head
        token_embeddings = self.ln_f(token_embeddings)  # dim:(B, 2*H*W, C)
        # x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        # x = x.view(bs, 1, self.vert_anchors, self.horz_anchors, self.n_embd)
        # x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)
        token_embeddings = token_embeddings.view(bs, self.vert_anchors, self.horz_anchors, self.n_embd)
        token_embeddings = token_embeddings.permute(0, 3, 1, 2)  # dim:(B, 2, C, H, W)
        # 这样截取的方式, 是否采用映射的方式更加合理？
        # rgb_fea_out = x[:, 0, :, :, :].contiguous().view(
        #     bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        # ir_fea_out = x[:, 1, :, :, :].contiguous().view(
        #     bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        # rgb_fea_out = F.interpolate(
        #     rgb_fea_out, size=([h, w]), mode='bilinear')
        token_embeddings = F.interpolate(
            token_embeddings, size=([h, w]), mode='bilinear')
        # ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')
        # print("-----GPT out shape-----", rgb_fea_out.size(), ir_fea_out.size())

        return x + token_embeddings
