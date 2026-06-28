from functools import partial
from re import T
from typing import MutableMapping
from unittest.mock import patch

# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from vision_transformer_irpe import PatchEmbed, Block
import math
from util.pos_embed import get_2d_sincos_pos_embed
from typing import Tuple, Union
import torch
import torch.nn as nn
import math


class SASGTTarget(nn.Module):
    """Fixed speckle-adaptive scale-space target for single-channel SAR images.

    The target generator has no trainable parameters.  It converts multiplicative
    intensity variations to the log domain, measures scale-normalized Gaussian
    derivatives, estimates local response reliability, and deterministically
    aggregates scales.  The two output channels are the aggregated gradient
    response and the expected (dominant) log scale.
    """

    def __init__(self, scales=(0.8, 1.6, 3.2, 6.4), temperature=1.0,
                 gamma=1.0, reliability_window=7, eps=1e-6):
        super().__init__()
        if not scales or any(s <= 0 for s in scales):
            raise ValueError("SASGT scales must be positive")
        if temperature <= 0:
            raise ValueError("SASGT temperature must be positive")
        if reliability_window < 3 or reliability_window % 2 == 0:
            raise ValueError("reliability_window must be an odd integer >= 3")

        self.scales = tuple(float(s) for s in scales)
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.reliability_window = int(reliability_window)
        self.eps = float(eps)

        for index, sigma in enumerate(self.scales):
            radius = max(2, int(math.ceil(3.0 * sigma)))
            coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
            gaussian = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
            gaussian = gaussian / gaussian.sum()
            derivative = -(coords / (sigma ** 2)) * gaussian
            derivative = derivative - derivative.mean()
            self.register_buffer(f"gaussian_{index}", gaussian)
            self.register_buffer(f"derivative_{index}", derivative)

        self.register_buffer(
            "log_scales", torch.log(torch.tensor(self.scales, dtype=torch.float32))
        )

    @staticmethod
    def _separable_filter(x, kernel_x, kernel_y):
        pad_x = kernel_x.numel() // 2
        pad_y = kernel_y.numel() // 2
        x = F.pad(x, (pad_x, pad_x, 0, 0), mode="reflect")
        x = F.conv2d(x, kernel_x.view(1, 1, 1, -1))
        x = F.pad(x, (0, 0, pad_y, pad_y), mode="reflect")
        return F.conv2d(x, kernel_y.view(1, 1, -1, 1))

    @staticmethod
    def _standardize(x, eps):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        var = x.var(dim=(-2, -1), keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + eps)

    @torch.no_grad()
    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("SASGT expects input shaped [B, 1, H, W]")

        # ToTensor produces non-negative intensity values. Clamp also protects
        # mixed-precision execution and custom datasets with tiny negative noise.
        log_x = torch.log(torch.clamp(x.float(), min=0.0) + self.eps)
        magnitudes = []
        scores = []
        pool_pad = self.reliability_window // 2

        for index, sigma in enumerate(self.scales):
            gaussian = getattr(self, f"gaussian_{index}")
            derivative = getattr(self, f"derivative_{index}")
            dx = self._separable_filter(log_x, derivative, gaussian)
            dy = self._separable_filter(log_x, gaussian, derivative)
            magnitude = (sigma ** self.gamma) * torch.sqrt(dx.square() + dy.square() + self.eps)

            local_mean = F.avg_pool2d(
                magnitude, self.reliability_window, stride=1, padding=pool_pad
            )
            local_deviation = F.avg_pool2d(
                (magnitude - local_mean).abs(), self.reliability_window,
                stride=1, padding=pool_pad
            )
            reliability = magnitude / (local_deviation + self.eps)
            magnitudes.append(magnitude)
            scores.append(reliability)

        magnitude_stack = torch.cat(magnitudes, dim=1)
        score_stack = torch.cat(scores, dim=1)
        weights = torch.softmax(score_stack / self.temperature, dim=1)

        adaptive_gradient = (weights * magnitude_stack).sum(dim=1, keepdim=True)
        dominant_scale = (
            weights * self.log_scales.view(1, -1, 1, 1)
        ).sum(dim=1, keepdim=True)

        adaptive_gradient = self._standardize(adaptive_gradient, self.eps)
        dominant_scale = self._standardize(dominant_scale, self.eps)
        return torch.cat((adaptive_gradient, dominant_scale), dim=1)

class LFST_Target(nn.Module):
    """
    根据论文 "Low-Frequency Structural Target (LFST)" 实现的目标生成器。
    流程: 原图 -> FFT2 -> FFTShift -> 乘低通掩膜 -> iFFTShift -> iFFT2 -> 取实部 -> 空间分块
    - 输入: x (B, 1, H, W)
    - 输出: patched_X_low (B, L, D_patch) 空间域低频目标
    """
    def __init__(self, img_size=224, patch_size=16, cutoff_freq=30):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.cutoff_freq = cutoff_freq # 低通滤波器的截止频率 (半径)
        
        # 计算 patch 数量和维度
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = patch_size * patch_size  # 因为是单通道 SAR 图像

        # 预先计算并注册径向低通掩膜 M_low (公式 9)
        mask = self._create_radial_mask(img_size, img_size, cutoff_freq)
        self.register_buffer("M_low", mask)

    def _create_radial_mask(self, h, w, radius):
        """生成中心的径向低通掩膜 M_low"""
        center_x, center_y = w // 2, h // 2
        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist_from_center = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= radius
        return mask.float() # (H, W)

    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, 1, H, W) SAR 图像
        return: (B, L, D_patch) 空间域的低频目标，用于 MIM 重建
        """
        B, C, H, W = x.shape
        assert C == 1, "LFST is designed for single-channel SAR images"

        # 1. 2D FFT (公式 8: F(X))
        # 注意：这里使用全尺寸 fft2 而不是 rfft2，为了方便做完美的中心化径向掩膜
        fft_x = torch.fft.fft2(x) 
        
        # 2. 零频中心化 (公式 8: S(F(X)))
        fft_shift = torch.fft.fftshift(fft_x, dim=(-2, -1)) 
        
        # 3. 施加低通掩膜 (公式 9: F_shift \odot M_low)
        # M_low 形状为 (H, W)，自动广播到 (B, 1, H, W)
        fft_filtered = fft_shift * self.M_low.unsqueeze(0).unsqueeze(0)
        
        # 4. 逆中心化与逆 FFT (后续描述: S^-1 和 F^-1)
        ifft_shift = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        x_low_complex = torch.fft.ifft2(ifft_shift)
        
        # 5. 取实部回到空间域 (后续描述: X_low = real(...))
        x_low = torch.real(x_low_complex) # (B, 1, H, W)
        x_min = x_low.amin(dim=(-2, -1), keepdim=True)
        x_max = x_low.amax(dim=(-2, -1), keepdim=True)
        x_low = (x_low - x_min) / (x_max - x_min + 1e-6)

        # 6. 将空间域的低通图像进行分块，以适应 MIM 的 target 格式
        # x_low: (B, 1, H, W) -> patches: (B, L, D_patch)
        P = self.patch_size
        patches = x_low.unfold(2, P, P).unfold(3, P, P) # (B, 1, H/P, W/P, P, P)
        patches = patches.reshape(B, 1, self.num_patches, P, P)
        patches = patches.permute(0, 2, 1, 3, 4) # (B, L, 1, P, P)
        patches = patches.reshape(B, self.num_patches, -1) # (B, L, P*P)

        return patches



# 20250812-----------------------------------------------------------------------
import torch.nn.init as init
import scipy.linalg

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        
        def sum(tensor, dim=None, keepdim=False):
            if dim is None:
                # sum up all dim
                return torch.sum(tensor)
            else:
                if isinstance(dim, int):
                    dim = [dim]
                dim = sorted(dim)
                for d in dim:
                    tensor = tensor.sum(dim=d, keepdim=True)
                if not keepdim:
                    for i, d in enumerate(dim):
                        tensor.squeeze_(d-i)
                return tensor

        def pixels(tensor):
            return int(tensor.size(2) * tensor.size(3))

        w_shape = self.w_shape
        if not self.LU:
            pixels = pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = sum(self.log_s) * pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet
        
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class InvBlock(nn.Module):
    """
    严格对应公式 (24)-(26): 空间域仿射耦合 (Affine Coupling)
    保留了你原有的 subnet_constructor (即 DenseBlock) 接口
    """
    def __init__(self, subnet_constructor, channel_in, channel_out, d=1, clamp=0.8):
        super(InvBlock, self).__init__()
        # 空间特征 H_spat 的通道数被拆分为两半
        self.split_len = channel_in // 2 
        self.clamp = clamp

        # 公式 24: t1(·)
        self.t1 = subnet_constructor(self.split_len, self.split_len, d)
        # 公式 25: s1(·) 和 t2(·)
        self.s1 = subnet_constructor(self.split_len, self.split_len, d)
        self.t2 = subnet_constructor(self.split_len, self.split_len, d)

    def forward(self, x):
        # 公式 24 前置: H_spat split into H_a and H_b
        H_a, H_b = x.chunk(2, dim=1) 

        # 公式 24: Z_a = H_a + t1(H_b)
        Z_a = H_a + self.t1(H_b)
        
        # 公式 25: Z_b = H_b ⊙ exp(s1(Z_a)) + t2(Z_a)
        s = self.clamp * (torch.sigmoid(self.s1(Z_a)) * 2 - 1)
        Z_b = H_b * torch.exp(s) + self.t2(Z_a)

        # 公式 26: Pspat = Concat(Z_a, Z_b)
        out = torch.cat((Z_a, Z_b), dim=1)
        return out

class Freprocess(nn.Module):
    """
    严格对应公式 (22)-(23): 频域特征处理
    """
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        
        # 公式 22: gM(·) 和 gθ(·) 
        self.gM = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), 
                                nn.LeakyReLU(0.1, inplace=False),
                                nn.Conv2d(channels, channels, 1, 1, 0))
        self.gtheta = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), 
                                    nn.LeakyReLU(0.1, inplace=False),
                                    nn.Conv2d(channels, channels, 1, 1, 0))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, Hfreq):
        _, _, H, W = Hfreq.shape
        # cuFFT cannot process every window size in FP16/BF16. Keep this
        # compact FFT branch in FP32 while the rest of the model uses autocast.
        with torch.autocast(device_type=Hfreq.device.type, enabled=False):
            Hfreq = Hfreq.float()
            fft_complex = torch.fft.rfft2(self.pre1(Hfreq) + 1e-8, norm='backward')
            magnitude = torch.abs(fft_complex)
            phase = torch.angle(fft_complex)
            mapped_magnitude = self.gM(magnitude)
            mapped_phase = self.gtheta(phase)
            complex_recon = torch.polar(mapped_magnitude, mapped_phase)
            out = torch.fft.irfft2(complex_recon, s=(H, W), norm='backward')
            return self.post(torch.real(out))

class SFAFM(nn.Module): # 类名保持 SFAFM 以兼容你外层调用的代码，其实质是论文中的 SFAFM
    """
    严格对应公式 (17)-(31): SFAFM (Spatial-Frequency Adaptive Fusion Module)
    """
    def __init__(self, embed_dim, reduction=4):
        super().__init__()
        channels = embed_dim // reduction  # 通道数 C'
        
        # ================== Step 1 ==================
        # 压缩层 Compress(·)
        self.Compress = nn.Conv2d(embed_dim, channels, kernel_size=1)
        self.alpha_raw = nn.Parameter(torch.tensor(0.5))
        
        # 空间/频率子空间映射: gspat(·) 和 gfreq(·)
        self.gspat = nn.Conv2d(channels, channels * 2, 3, 1, 1) # 输出 2*channels 给 InvBlock 拆分
        self.gfreq = nn.Conv2d(channels, channels, 3, 1, 1)

        # ================== Step 2 ==================
        # 空间/频率分支处理模块
        self.spa_process = nn.Sequential(
            InvBlock(DenseBlock, channel_in=2*channels, channel_out=2*channels),
            nn.Conv2d(2*channels, channels, 1, 1, 0) # 仿射耦合后变回 channels
        )
        self.fre_process = Freprocess(channels)
        
        # ================== Step 3 ==================
        # 空间注意力 Aspat(·) 和 通道注意力 Achan(·)
        self.Aspat = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        
        # 保留你原来的池化/对比度计算
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels 
        
        self.Achan = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels // 2, channels * 2, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        # 后处理 fpost(·)
        self.fpost = nn.Conv2d(channels * 2, channels, 3, 1, 1)

        # ================== Step 4 ==================
        # 扩展层 Expand(·)
        self.Expand = nn.Conv2d(channels, embed_dim, kernel_size=1)
        
        
    def forward(self, Xi):  
        """
        输入: Xi (形状为 [B, C, H, W]，已由 MAE 解包为密集 2D 特征图)
        """
        # 公式 17: Xcached = X_i
        Xcached = Xi 

        # ========== Step 1: Feature Compression & Separation ==========
        # 公式 18: Xcomp = Compress(X_i)
        Xcomp = self.Compress(Xi)

        # 公式 19: alpha = sigmoid(alpha_raw)
        alpha = torch.sigmoid(self.alpha_raw)

        # 公式 20, 21: Hspat 和 Hfreq
        Hspat = self.gspat(Xcomp) * alpha
        Hfreq = self.gfreq(Xcomp) * (1 - alpha)

        # ========== Step 2: Branch-Specific Processing ==========
        Pspat = self.spa_process(Hspat) 
        Pfreq = self.fre_process(Hfreq) 

        # ========== Step 3: Dual-Domain Representation Interaction & Fusion ==========
        # 公式 27: Mspat = Aspat(Pspat - Pfreq)
        Mspat = self.Aspat(Pspat - Pfreq)

        # 公式 28: Pcomp = Pspat ⊙ Mspat + Pfreq
        Pcomp = Pspat * Mspat + Pfreq

        # 公式 29 前置准备: Concat(Pcomp, Pspat)
        cat_f = torch.cat([Pcomp, Pspat], dim=1) 
        
        # 公式 29: Ffused = Achan(Concat) * Concat
        # (包含了你原本代码中的对比度和全局均值特征)
        cha_weights = self.Achan(self.contrast(cat_f) + self.avgpool(cat_f))
        Ffused = cat_f * cha_weights

        # 公式 30: Xfused = fpost(Ffused) + Pcomp
        Xfused = self.fpost(Ffused) + Pcomp

        # ========== Step 4: Feature Expansion & Final Residual ==========
        # 公式 31: Yi = Expand(Xfused) + Xcached
        Yi = self.Expand(Xfused) + Xcached
        
        return Yi

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 lfst_cutoff=30, lfst_loss_weight=0.3,
                 sasgt_scales=(0.8, 1.6, 3.2, 6.4), sasgt_temperature=1.0,
                 sasgt_gamma=1.0, sasgt_reliability_window=7):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # 在 __init__ 中的定义应该类似这样：
        self.blocks = nn.ModuleList()
        self.sfafm_blocks = nn.ModuleList()
        
        for i in range(depth):
            # 添加常规 ViT
            self.blocks.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer))
            
            # 论文中：SFAFM is inserted after every two ViT blocks
            if (i + 1) % 2 == 0:
                self.sfafm_blocks.append(SFAFM(embed_dim, reduction=4))
            else:
                self.sfafm_blocks.append(None) # 用 None 占位，保持与 self.blocks 索引完全一致
                
        self.norm = norm_layer(embed_dim)
        

        self.encoder_pred = nn.Linear(embed_dim, decoder_embed_dim, bias=True) # decoder to patch
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # 20250812--------------------------------------------------------------------------
        self.img_SFAFM_process = SFAFM(embed_dim, reduction=4)
        
    
        # 20260522--------------------------------------------------------------------------
        # ========== Head B: 低频结构目标 (LFST) head ==========
        # 此时目标已经是在空间域滤波后的图像，所以维度变回正常的 patch 大小
        self.P = patch_size
        # 每个 patch 的维度: patch_size * patch_size * 1 (单通道)
        self.lfst_out_dim = self.P * self.P * 1  

        # 修改预测头的输出维度
        self.decoder_pred_lfst = nn.Linear(decoder_embed_dim, self.lfst_out_dim, bias=True)
        
        # 引入我们在上一步写好的 LFST_Target (请确保它被定义在上方或导入)
        self.lfst_builder = LFST_Target(
            img_size=self.img_size,
            patch_size=self.patch_size,
            cutoff_freq=lfst_cutoff
        )

        # LFST 损失权重（你可以暴露为超参）
        self.lfst_loss_weight = float(lfst_loss_weight)
        # --------------------------------------------------------------------------  
        # Fixed spatial target generator (no trainable parameters).
        self.sasgt_builder = SASGTTarget(
            scales=sasgt_scales,
            temperature=sasgt_temperature,
            gamma=sasgt_gamma,
            reliability_window=sasgt_reliability_window,
        )

        # 修改预测头的输出维度！
        # Two SASGT channels are predicted for every pixel in a patch.
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size**2 * 2, bias=True)
        # --------------------------------------------------------------------------
        # MAE decoder specifics

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def _get_pixel_label_2d(self, input_img, output_masks, norm=True):
        input_img = input_img.permute(0, 2, 3, 1)
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            size = self.feat_stride[depth][-1]
            label = input_img.unfold(1, size, size).unfold(2, size, size)
            label = label.flatten(1, 2).flatten(2)
            label = label[output_mask]
            if norm:
                mean = label.mean(dim=-1, keepdim=True)
                var = label.var(dim=-1, keepdim=True)
                label = (label - mean) / (var + 1.0e-6) ** 0.5
            labels.append(label)
        return labels

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        支持多通道的 patchify
        imgs: (N, C, H, W) -> x: (N, L, patch_size**2 * C)
        """
        p = self.patch_embed.patch_size[0]
        C = imgs.shape[1] # ✅ 动态获取通道数，不要写死 1
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        # ✅ 使用动态通道数 C
        x = imgs.reshape(shape=(imgs.shape[0], C, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * C))
        return x

    def _sparse_1d_to_dense_2d(self, x, window_size):
        """
        极简版 1D 转 2D (完美适配你的 Window Masking 机制)
        因为序列已经是包含了 mask_token 的完整局部窗口，直接 Reshape 即可。
        - x 形状: [B*num_window, window_size^2 + 1, D]
        """
        B_prime, _, D = x.shape
        cls_token = x[:, :1, :]
        x_spatial = x[:, 1:, :] # 去掉 CLS，剩下的就是窗口内的所有 patch [B', window_size^2, D]
        
        # 直接变形为 2D 图像特征图
        x_2d = x_spatial.permute(0, 2, 1).reshape(B_prime, D, window_size, window_size)
        return x_2d, cls_token

    def _dense_2d_to_sparse_1d(self, x_2d, cls_token):
        """
        极简版 2D 转 1D
        SFAFM 处理完后，将特征图展平并拼回 CLS token。
        """
        B_prime, D, H_w, W_w = x_2d.shape
        x_1d = x_2d.view(B_prime, D, -1).permute(0, 2, 1) # [B', window_size^2, D]
        
        x_sparse = torch.cat([cls_token, x_1d], dim=1)
        return x_sparse
    
    def unpatchify(self, x):
        """
        支持多通道的 unpatchify
        x: (N, L, patch_size**2 * C)
        返回 imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1], "序列长度无法开方为整数的网格"
        
        # 动态推断通道数 C
        C = x.shape[-1] // (p ** 2)
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, C))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], C, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def sample_patch_index_single_window(self,x,patch_index, keep_ratio):
        N, H, W, D = x.shape
        x = x.view(N,H*W,D)


        noise = torch.rand(N,patch_index.shape[0], device=patch_index.device)  # noise in [0, 1]
        
        ids_shuffle = torch.argsort(noise,dim=1)  # ascend: small is keep, large is remove

        ids_keep = ids_shuffle[:,:keep_ratio]

        patch_keeps = patch_index[ids_keep]

        return patch_keeps

    def sample_patch_index(self,x,patch_index, keep_ratio):

        N, H, W, D = x.shape
        M,P = patch_index.shape
        patch_index = patch_index.unsqueeze(0).expand(N,M,P)


        noise = torch.rand(N,M,P, device=patch_index.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise,dim=-1)  # ascend: small is keep, large is remove


        ids_keep = ids_shuffle[:,:,:keep_ratio]

        patch_keeps = torch.gather(patch_index, -1, ids_keep)

        return patch_keeps

    def generate_window_patches(self,x,left,top, window_size, mask_ratio):
        N, H, W, D = x.shape
        window_number = left.shape[0]
        

        #  extract the windows based on the coordinates
        left = left.unsqueeze(-1).expand(window_number,window_size)
        top  = top.unsqueeze(-1).expand(window_number, window_size)


        row = torch.arange(0,window_size,device=x.device).unsqueeze(0).expand(window_number,window_size)+left
        column = torch.arange(0,window_size*W,W, device = x.device).unsqueeze(0).expand(window_number, window_size)+top*W
        

        in_window_mask_number = int(window_size*window_size*mask_ratio)  

        assert in_window_mask_number>=1
        in_window_patches =row.unsqueeze(1).expand(window_number,window_size,window_size)  + column.unsqueeze(-1).expand(left.shape[0],window_size,window_size)
        in_window_patches = in_window_patches.view(window_number,-1)


        # sample the masked patch ids
        ids_mask_in_window =self.sample_patch_index(x,in_window_patches,in_window_mask_number)


        patches_to_keep = in_window_patches.unsqueeze(0).expand(N, window_number,window_size* window_size)
        x = x.view(N,H*W,D).unsqueeze(0).repeat(window_number,1, 1,1).view(N*window_number,H*W,D)


        sorted_patch_to_keep,_ = torch.sort(patches_to_keep,dim=-1)
        sorted_patch_to_keep = sorted_patch_to_keep.view(N*window_number,-1)

        ids_mask_in_window = ids_mask_in_window.view(N*window_number, -1)

        # gather the masked patches
        x_masked = torch.gather(x, dim=1, index=sorted_patch_to_keep.unsqueeze(-1).repeat(1, 1, D)).clone()
        # indices for recontruction
        mask_indices = ((sorted_patch_to_keep.unsqueeze(-1)- ids_mask_in_window.unsqueeze(1))==0).sum(-1)==1

        # zero out the patches in mask
        x_masked[mask_indices]=self.mask_token
 
        return x_masked, sorted_patch_to_keep,mask_indices


    def forward_encoder(self, x, window_size, num_window, mask_ratio):
        # 1. embed patches (不要在这里加位置编码，移到提取窗口之后加，确保 mask token 也能拿到位置信息)
        x = self.patch_embed(x).type(torch.float32) # [B, L, C]

        N, _, C = x.shape
        H = W = self.img_size // self.patch_size
        x = x.view(N, H, W, C)
        assert window_size <= H and window_size <= W

        # sample window coordinates
        rand_top_locations  = torch.randperm(H - window_size + 1, device=x.device)[:num_window]
        rand_left_locations = torch.randperm(W - window_size + 1, device=x.device)[:num_window]
        
        # generate the sampled and mask patches from the small windows
        # 此时 x 的形状变为 [B*num_window, window_size^2, C]
        # 注意: ids_restore 里面存的是这些 patch 在原图中的绝对一维索引
        x, ids_restore, mask_indices = self.generate_window_patches(
            x, rand_left_locations, rand_top_locations, window_size, mask_ratio
        )

        # ================== 🚨 核心修复：在这里赋予全局绝对位置信息 ==================
        # 取出全局的 patch 位置编码 [1, H*W, C]
        global_pos = self.pos_embed[:, 1:, :].expand(N * num_window, -1, -1)
        # 根据 window 切出的索引，把对应位置的位置编码拿过来，并加到序列上
        window_pos = torch.gather(global_pos, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))
        x = x + window_pos 
        # =========================================================================

        # append the cls tokens at the beginning
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # 记得给 CLS token 加上它的位置编码
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        x = torch.cat((cls_tokens, x), dim=1)

        # ----- encoder (带有 SFAFM 穿插逻辑) -----
        for i, blk in enumerate(self.blocks):
            # 1. 过基础的 ViT Block
            x = blk(x)
            
            # 2. 检查当前层后面是否跟着 SFAFM
            sfafm_blk = self.sfafm_blocks[i]
            if sfafm_blk is not None:
                # 局部窗口变 2D -> SFAFM -> 变回 1D
                # 传入 window_size 即可，因为你的子网格大小就是 window_size * window_size
                x_2d, cls_token = self._sparse_1d_to_dense_2d(x, window_size)
                x_2d = sfafm_blk(x_2d) 
                x = self._dense_2d_to_sparse_1d(x_2d, cls_token)
        
        # ------ 最后一个 SF-Block (SFAFM) ------
        x_2d, cls_token = self._sparse_1d_to_dense_2d(x, window_size)
        x_2d = self.img_SFAFM_process(x_2d)
        x = self._dense_2d_to_sparse_1d(x_2d, cls_token)
        
        x = self.norm(x)

        # ----- shared decoder trunk -----
        x = self.encoder_pred(x)          # proj to decoder dim
        for blk in self.decoder_blocks:
            x = blk(x)
        dec_feat = self.decoder_norm(x)   # (B*num_window, L_win+1, Ddec)

        # ===== two heads =====
        pred_grad_like = self.decoder_pred(dec_feat)         
        pred_lfst      = self.decoder_pred_lfst(dec_feat)    

        # remove cls token for both heads
        pred_grad_like = pred_grad_like[:, 1:, :]            
        pred_lfst      = pred_lfst[:, 1:, :]                 

        return pred_grad_like, pred_lfst, mask_indices, ids_restore

    def forward_loss_lfst(self, imgs, pred_lfst, mask_indices, num_window, ids_restore):
        """
        imgs:       (N, 1, H, W)
        pred_lfst:  (N, L, P*P) —— 来自 decoder_pred_lfst
        mask_indices: (N, L)  0=keep, 1=remove
        """
        with torch.no_grad():
            target_lfst = self.lfst_builder(imgs)     # (N, L, P*P)

        N, P, H = target_lfst.shape                   # P=L, H=D_lfst
        target_lfst = target_lfst.unsqueeze(0).repeat(num_window, 1, 1, 1).view(-1, P, H)
        target_lfst = torch.gather(target_lfst, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, H))

        # 使用 MSE 损失 (L2) 预测空间域平滑后的像素
        loss = (pred_lfst - target_lfst) ** 2
        loss = loss.mean(dim=-1)                      # (N, L)

        # 只在被 mask 的 patch 上求平均
        loss = (loss * mask_indices).sum() / (mask_indices.sum() + 1e-6)
        return loss


    def forward_loss(self, imgs, pred, mask_indices, num_window, ids_restore):
        # 1. 直接获取 (B, 4, H, W) 的多尺度梯度目标
        with torch.no_grad():
            t_spat = self.sasgt_builder(imgs)
            
        # 2. 将其切分为 patch: (N, L, P*P*4)
        target = self.patchify(t_spat)

        N, P_len, H_dim = target.shape
        target = target.unsqueeze(0).repeat(num_window, 1, 1, 1).view(-1, P_len, H_dim)
        target = torch.gather(target, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, H_dim))

        # 使用 MSE Loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  
        loss = (loss * mask_indices).sum() / (mask_indices.sum() + 1e-6)  
        return loss

    def forward(self, imgs, window_size=7, num_window=4, mask_ratio=0.8):
        # 共享解码器 + 双 head 的输出
        pred_grad_like, pred_lfst, mask_indices, ids_restore = self.forward_encoder(
            imgs, window_size, num_window, mask_ratio
        )

        # SASGT spatial-target loss.
        loss_grad = self.forward_loss(imgs, pred_grad_like, mask_indices, num_window, ids_restore)
        
        # 低频结构损失 (LFST)
        loss_lfst = self.forward_loss_lfst(imgs, pred_lfst, mask_indices, num_window, ids_restore)

        # 总损失
        loss = loss_grad + self.lfst_loss_weight * loss_lfst

        # 返回两个预测，便于可视化/调试
        return loss, (pred_grad_like, pred_lfst), mask_indices


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge448_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=448,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge672_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=672,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge996_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=996,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge336_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=336,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_384_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=384,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_448_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=448,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch14_224_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=224,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch8_224_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=224,
        patch_size=8, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_base_patch8_224 = mae_vit_base_patch8_224_dec512d8b
mae_vit_base_patch14_224 = mae_vit_base_patch14_224_dec512d8b
mae_vit_base_patch16_384 = mae_vit_base_patch16_384_dec512d8b
mae_vit_base_patch16_448 = mae_vit_base_patch16_448_dec512d8b

mae_vit_huge336_patch14 = mae_vit_huge336_patch14_dec512d8b
mae_vit_huge448_patch14 = mae_vit_huge448_patch14_dec512d8b
mae_vit_huge672_patch14 = mae_vit_huge672_patch14_dec512d8b
mae_vit_huge996_patch14 =mae_vit_huge996_patch14_dec512d8b





#mae_vit_huge448_patch14 = mae_vit_huge448_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

def vit_tiny(**kwargs):
    model = MaskedAutoencoderViT(img_size=224,
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT(img_size=224,
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model

mae_vit_tiny = vit_tiny
mae_vit_small = vit_small


# if __name__ == '__main__':
#     from torchsummary import summary
#     # from torchinfo import summary

#     # 创建模型
#     model = mae_vit_base_patch16()

#     # 将模型移到 GPU 或 CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('device',device)
#     model.to(device)

#     # 假设输入图像大小为 (1, 224, 224)
#     summary(model, input_size=(1, 224, 224))
