"""Network definitions for the released models (FNO, F-FNO, U-Net, DilResNet).

Extracted from the training code used in the paper. Load a released checkpoint with `load_model`.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    base_channels: int = 64
    num_down: int = 4
    max_channels: int = 512
    norm_type: str = "group"
    num_groups: int = 8
    use_strided_conv_down: bool = True
    bilinear: bool = True
    act: str = "gelu"


@dataclass
class DilResNetConfig:
    width: int = 128
    depth: int = 8
    dilation_cycle: int = 4
    kernel_size: int = 3
    norm_type: str = "group"
    num_groups: int = 8
    act: str = "gelu"


# =========================
# Utility
# =========================
def _get_act(name: str):
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


# torch._dynamo optional
try:
    import torch._dynamo as dynamo
    _has_dynamo = True
except Exception:
    dynamo = None
    _has_dynamo = False


# =========================
# FNO
# =========================
class SpectralConv2d(nn.Module):
    """
    2D Fourier layer: rFFT2 -> learned modes (both +kx and -kx) -> iFFT2.

    Follows the original FNO paper (Li et al. 2021):
    - Unnormalized FFT (no norm="ortho"), matching the official implementation.
    - Two weight tensors: weight_pos for the top +kx block, weight_neg for the
      bottom -kx block of the rfft2 spectrum.
    - Weights stored as real (float32) pairs to stay compatible with torch.compile
      and avoid Inductor complex32 issues; combined to cfloat on the fly.
    """
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)

        self.weight_pos = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_x, modes_y, 2) * scale
        )
        self.weight_neg = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_x, modes_y, 2) * scale
        )

    if _has_dynamo:
        @dynamo.disable()
        def forward(self, x):
            return self._forward_impl(x)
    else:
        def forward(self, x):
            return self._forward_impl(x)

    def _forward_impl(self, x):
        """
        x: (B, C_in, H, W) real
        returns: (B, C_out, H, W) real
        """
        B, C, H, W = x.shape
        mx = min(self.modes_x, H)
        my = min(self.modes_y, W // 2 + 1)

        x32 = x.float()
        x_ft = torch.fft.rfft2(x32)  # unnormalized, matching original FNO paper

        w_pos = (self.weight_pos[..., 0] + 1j * self.weight_pos[..., 1]).to(x_ft.dtype)
        w_neg = (self.weight_neg[..., 0] + 1j * self.weight_neg[..., 1]).to(x_ft.dtype)

        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            device=x.device, dtype=x_ft.dtype
        )

        out_ft[:, :, :mx, :my] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :mx, :my],
            w_pos[:, :, :mx, :my]
        )

        out_ft[:, :, -mx:, :my] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -mx:, :my],
            w_neg[:, :, :mx, :my]
        )

        out32 = torch.fft.irfft2(out_ft, s=(H, W))  # unnormalized
        return out32.to(x.dtype) if x.dtype != torch.float32 else out32


class FNOBlock2d(nn.Module):
    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_y: int,
        act: str = "gelu",
        variant: str = "basic",
        spectral_width: Optional[int] = None,
    ):
        super().__init__()
        self.variant = variant.lower()
        self.width = width
        self.spectral_width = spectral_width if spectral_width is not None else width

        self.in_proj = None
        self.out_proj = None
        if self.spectral_width != width:
            self.in_proj = nn.Conv2d(width, self.spectral_width, kernel_size=1)
            self.out_proj = nn.Conv2d(self.spectral_width, width, kernel_size=1)

        self.spec = SpectralConv2d(self.spectral_width, self.spectral_width, modes_x, modes_y)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.act = _get_act(act)

        if "mlp" in self.variant:
            self.mlp = nn.Sequential(
                nn.Conv2d(width, width * 2, kernel_size=1),
                _get_act(act),
                nn.Conv2d(width * 2, width, kernel_size=1),
            )
        else:
            self.mlp = None

    def forward(self, x):
        residual = x

        x_spec = x
        if self.in_proj is not None:
            x_spec = self.in_proj(x_spec)
        x_spec = self.spec(x_spec)
        if self.out_proj is not None:
            x_spec = self.out_proj(x_spec)

        x = x_spec + self.w(x)

        if "skip" in self.variant:
            x = x + residual

        x = self.act(x)

        if self.mlp is not None:
            mlp_out = self.mlp(x)
            if "skip" in self.variant:
                x = x + mlp_out
            else:
                x = mlp_out

        return x


class FNO2D(nn.Module):
    """
    FNO2D with residual spectral blocks and pointwise convs.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int = 20,
        modes_y: int = 20,
        width: int = 96,
        depth: int = 8,
        act: str = "gelu",
        pad_ratio: float = 0.0,
        variant: str = "basic",
        spectral_width: Optional[int] = 32,
        share_spectral_weights: bool = False,  # kept for config compatibility
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.depth = depth
        self.pad_ratio = pad_ratio
        self.variant = variant

        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            FNOBlock2d(
                width=width,
                modes_x=modes_x,
                modes_y=modes_y,
                act=act,
                variant=variant,
                spectral_width=spectral_width,
            )
            for _ in range(depth)
        ])
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_channels, kernel_size=1)
        self.act = _get_act(act)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.fc0(x)

        pad_h = int(H * self.pad_ratio)
        pad_w = int(W * self.pad_ratio)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        for blk in self.blocks:
            x = blk(x)

        if pad_h > 0 or pad_w > 0:
            x = x[..., :H, :W]

        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================
# F-FNO (Factorized Fourier Neural Operator)
# Tran et al., ICLR 2023  arXiv:2111.13802
# =========================
class FactorizedSpectralConv2d(nn.Module):
    """Factorized 2D spectral convolution layer (F-FNO).

    Replaces the single joint 2D spectral conv of FNO with two independent
    1-D spectral convolutions applied along each spatial axis:

        F(x) = IRFFT_x( W_x · RFFT_x(x)[:, :modes_x] )
             + IRFFT_y( W_y · RFFT_y(x)[:, :, :modes_y] )

    where RFFT_x / RFFT_y denote 1-D real FFT along H / W respectively.

    Parameter count comparison (in_ch = out_ch = C, modes_x = modes_y = M):
        FNO  SpectralConv2d : 2 × C² × M²  (complex)   [× 2 for ±kx blocks]
        FFNO FactSpectralConv: 2 × C² × M   (complex)   [x + y directions]

    Weights stored as real (float32) pairs [..., 2] for torch.compile / Inductor
    compatibility (avoids cfloat parameter issues).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / (in_channels * out_channels)

        # w_x : channel mixing in the x (height) frequency domain
        #   shape: (in_ch, out_ch, modes_x, 2)  — last dim is (real, imag)
        self.w_x = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_x, 2) * scale
        )
        # w_y : channel mixing in the y (width) frequency domain
        self.w_y = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_y, 2) * scale
        )

    if _has_dynamo:
        @dynamo.disable()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._forward_impl(x)
    else:
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C_in, H, W) real
        returns : (B, C_out, H, W) real
        """
        B, C, H, W = x.shape
        mx = min(self.modes_x, H // 2 + 1)
        my = min(self.modes_y, W // 2 + 1)

        x32 = x.float()

        # Build complex weights from stored real pairs
        wx = (self.w_x[..., 0] + 1j * self.w_x[..., 1]).to(torch.complex64)  # (in, out, modes_x)
        wy = (self.w_y[..., 0] + 1j * self.w_y[..., 1]).to(torch.complex64)  # (in, out, modes_y)

        # ── 1-D spectral conv along H (x-axis) ───────────────────────────────
        # rfft along dim=-2: (B, C, H, W) → (B, C, H//2+1, W)
        x_ft_h = torch.fft.rfft(x32, dim=-2)

        out_h = torch.zeros(
            B, self.out_channels, H // 2 + 1, W,
            device=x.device, dtype=x_ft_h.dtype,
        )
        # einsum: b=batch, i=in_ch, x=modes_x, w=W(spatial) → o=out_ch
        out_h[:, :, :mx, :] = torch.einsum(
            "bixw, iox -> boxw",
            x_ft_h[:, :, :mx, :],
            wx[:, :, :mx],
        )
        # irfft along dim=-2: (B, out_ch, H//2+1, W) → (B, out_ch, H, W)
        x_h = torch.fft.irfft(out_h, n=H, dim=-2)

        # ── 1-D spectral conv along W (y-axis) ───────────────────────────────
        # rfft along dim=-1: (B, C, H, W) → (B, C, H, W//2+1)
        x_ft_w = torch.fft.rfft(x32, dim=-1)

        out_w = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            device=x.device, dtype=x_ft_w.dtype,
        )
        # einsum: b=batch, i=in_ch, h=H(spatial), y=modes_y → o=out_ch
        out_w[:, :, :, :my] = torch.einsum(
            "bihy, ioy -> bohy",
            x_ft_w[:, :, :, :my],
            wy[:, :, :my],
        )
        # irfft along dim=-1: (B, out_ch, H, W//2+1) → (B, out_ch, H, W)
        x_w = torch.fft.irfft(out_w, n=W, dim=-1)

        out = x_h + x_w
        return out.to(x.dtype) if x.dtype != torch.float32 else out


class FFNOBlock2d(nn.Module):
    """Single F-FNO layer.

    Structure (from Tran et al. 2023, Section 3):
        h  = act( FactSpectralConv(x) + Conv1x1(x) )   # same as FNO basic block
        x' = h + x                                       # block-level residual ← NEW

    The block-level residual is the second key contribution of F-FNO
    (absent in standard FNO).  It is always enabled.
    """

    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_y: int,
        act: str = "gelu",
    ):
        super().__init__()
        self.spec   = FactorizedSpectralConv2d(width, width, modes_x, modes_y)
        self.bypass = nn.Conv2d(width, width, kernel_size=1)  # local pointwise path
        self.act    = _get_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # spectral + local mixing, then block residual
        return self.act(self.spec(x) + self.bypass(x)) + x


class FFNO2D(nn.Module):
    """Factorized Fourier Neural Operator 2D.

    Tran et al., "Factorized Fourier Neural Operators", ICLR 2023.
    arXiv: 2111.13802

    Architectural changes vs. FNO2D
    ────────────────────────────────
    1. FactorizedSpectralConv2d instead of SpectralConv2d
       - Two 1-D spectral convolutions (along H, along W) summed
       - Learnable weights: (in_ch, out_ch, modes_x) + (in_ch, out_ch, modes_y)
       - Reduces spectral params from O(modes²) to O(modes) per block

    2. Block-level residual in every FFNOBlock2d
       - output = act(spec(x) + local(x)) + x
       - Enables training deeper networks without gradient vanishing

    Lifting / projection layers are identical to FNO2D for a fair comparison.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int = 20,
        modes_y: int = 20,
        width: int = 96,
        depth: int = 8,
        act: str = "gelu",
    ):
        super().__init__()
        self.fc0 = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([
            FFNOBlock2d(width, modes_x, modes_y, act)
            for _ in range(depth)
        ])
        self.fc1 = nn.Conv2d(width, width, kernel_size=1)
        self.fc2 = nn.Conv2d(width, out_channels, kernel_size=1)
        self.act = _get_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.act(self.fc1(x))
        return self.fc2(x)


# =========================
# UNet (ResBlock-based, improved baseline)
# =========================
def _make_norm(norm_type: str, num_channels: int, num_groups: int = 8) -> nn.Module:
    """Returns a normalization layer.

    GroupNorm is preferred for EM-field surrogate models:
    - batch-size independent (works well with small batches)
    - no train/eval mismatch unlike BatchNorm
    The num_groups is silently reduced to the largest divisor <= num_groups
    so the layer always constructs successfully.
    """
    if norm_type == "group":
        g = min(num_groups, num_channels)
        while num_channels % g != 0:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm_type: {norm_type!r}. Choose 'group', 'batch', or 'none'.")


class ResBlock(nn.Module):
    """ResNet-style block: Conv→Norm→Act→Conv→Norm → Add(shortcut) → Act.

    Compared with the original UNet DoubleConv:
    - Residual (shortcut) connection stabilises training on deep encoders.
    - GroupNorm avoids BatchNorm's batch-size dependency and train/eval mismatch.
    - bias=False when a norm layer follows the conv (norm's affine absorbs bias).
    - 1×1 projection shortcut when in_ch != out_ch.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "group",
        num_groups: int = 8,
        act: str = "gelu",
    ):
        super().__init__()
        use_bias = (norm_type == "none")
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=use_bias)
        self.norm1 = _make_norm(norm_type, out_ch, num_groups)
        self.act1  = _get_act(act)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=use_bias)
        self.norm2 = _make_norm(norm_type, out_ch, num_groups)
        self.act2  = _get_act(act)

        # 1×1 projection shortcut when channel dims differ
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                _make_norm(norm_type, out_ch, num_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + self.shortcut(x))


class DownBlock(nn.Module):
    """Downsampling + ResBlock.

    use_strided_conv=True  → learnable strided Conv2d (common in surrogate papers)
    use_strided_conv=False → MaxPool2d (original UNet)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        use_strided_conv: bool = True,
        norm_type: str = "group",
        num_groups: int = 8,
        act: str = "gelu",
    ):
        super().__init__()
        if use_strided_conv:
            self.downsample = nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, bias=False)
        else:
            self.downsample = nn.MaxPool2d(2)
        self.resblock = ResBlock(in_ch, out_ch, norm_type, num_groups, act)

    def forward(self, x):
        return self.resblock(self.downsample(x))


class UpBlock(nn.Module):
    """Upsampling + skip concatenation + ResBlock.

    bilinear=True  → bilinear Upsample (no extra params)
    bilinear=False → ConvTranspose2d (learnable, sharper)
    """
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        bilinear: bool = True,
        norm_type: str = "group",
        num_groups: int = 8,
        act: str = "gelu",
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.resblock = ResBlock(in_ch + skip_ch, out_ch, norm_type, num_groups, act)

    def forward(self, x, skip):
        x = self.up(x)
        # centre-crop pad to match skip resolution (handles odd input sizes)
        dY = skip.size(-2) - x.size(-2)
        dX = skip.size(-1) - x.size(-1)
        if dY != 0 or dX != 0:
            x = F.pad(x, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])
        return self.resblock(torch.cat([skip, x], dim=1))


class UNet2D(nn.Module):
    """Encoder–decoder UNet with residual blocks and configurable normalization.

    Channel schedule: [C, 2C, 4C, …] capped at max_channels.
    Example with base_channels=64, num_down=4, max_channels=512:
        encoder: in→64→128→256→512→512  (bottleneck)
        decoder: 512+512→512→512+256→256→256+128→128→128+64→64→out
    """
    def __init__(self, in_channels: int, out_channels: int, cfg: UNetConfig):
        super().__init__()
        C  = cfg.base_channels
        mC = cfg.max_channels
        NT = cfg.norm_type
        G  = cfg.num_groups

        # channel sizes per encoder level (index 0 = after inc, index N = bottleneck)
        ch = [min(C * (2 ** i), mC) for i in range(cfg.num_down + 1)]

        self.inc   = ResBlock(in_channels, ch[0], NT, G, cfg.act)
        self.downs = nn.ModuleList([
            DownBlock(ch[i], ch[i + 1], cfg.use_strided_conv_down, NT, G, cfg.act)
            for i in range(cfg.num_down)
        ])
        self.ups = nn.ModuleList([
            UpBlock(ch[cfg.num_down - i], ch[cfg.num_down - i - 1],
                    ch[cfg.num_down - i - 1],
                    cfg.bilinear, NT, G, cfg.act)
            for i in range(cfg.num_down)
        ])
        self.outc = nn.Conv2d(ch[0], out_channels, kernel_size=1)

    def forward(self, x):
        # encoder – collect skip connections
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        # decoder
        x = skips[-1]
        for i, up in enumerate(self.ups):
            x = up(x, skips[-2 - i])

        return self.outc(x)


# =========================
# Dilated ResNet (CNN baseline)
# =========================
class DilResBlock(nn.Module):
    """Dilated residual block: Conv(dil)→Norm→Act→Conv(dil)→Norm → Add(identity) → Act.

    Uses dilated convolutions to increase the receptive field without downsampling,
    which preserves spatial resolution — important for dense field prediction.
    """
    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        kernel_size: int = 3,
        norm_type: str = "group",
        num_groups: int = 8,
        act: str = "gelu",
    ):
        super().__init__()
        use_bias = (norm_type == "none")
        pad = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation, bias=use_bias)
        self.norm1 = _make_norm(norm_type, channels, num_groups)
        self.act1  = _get_act(act)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation, bias=use_bias)
        self.norm2 = _make_norm(norm_type, channels, num_groups)
        self.act2  = _get_act(act)

    def forward(self, x):
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + x)


class DilResNet2D(nn.Module):
    """Dilated ResNet — pure-CNN surrogate baseline.

    Stacks residual blocks with cycling dilation rates
    [1, 2, 4, 8, 1, 2, 4, 8, …] (period = dilation_cycle) so that deeper
    layers see an exponentially growing receptive field while the spatial
    resolution stays constant (no pooling).

    Commonly used as the CNN baseline in neural-operator benchmarks
    (PDEBench, PDEArena, etc.).
    """
    def __init__(self, in_channels: int, out_channels: int, cfg: DilResNetConfig):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, cfg.width, kernel_size=1)

        dilations = [2 ** (i % cfg.dilation_cycle) for i in range(cfg.depth)]
        self.blocks = nn.ModuleList([
            DilResBlock(cfg.width, d, cfg.kernel_size,
                        cfg.norm_type, cfg.num_groups, cfg.act)
            for d in dilations
        ])

        self.output_proj = nn.Sequential(
            nn.Conv2d(cfg.width, cfg.width, kernel_size=1),
            _get_act(cfg.act),
            nn.Conv2d(cfg.width, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.output_proj(x)



def build_network(architecture: str, hp: dict, in_channels: int = 3, out_channels: int = 2) -> nn.Module:
    """Instantiate an architecture ("fno", "ffno", "unet", "dilresnet") with the hyperparameters stored in a checkpoint."""
    a = architecture.lower()
    if a == "fno":
        return FNO2D(in_channels=in_channels, out_channels=out_channels, modes_x=hp["modes_x"], modes_y=hp["modes_y"],
                     width=hp["width"], depth=hp["depth"], act=hp["act"], pad_ratio=hp["pad_ratio"],
                     variant=hp["variant"], spectral_width=hp["spectral_width"],
                     share_spectral_weights=hp["share_spectral_weights"])
    if a == "ffno":
        return FFNO2D(in_channels=in_channels, out_channels=out_channels, modes_x=hp["modes_x"], modes_y=hp["modes_y"],
                      width=hp["width"], depth=hp["depth"], act=hp["act"])
    if a == "unet":
        return UNet2D(in_channels, out_channels, UNetConfig(**hp))
    if a == "dilresnet":
        return DilResNet2D(in_channels, out_channels, DilResNetConfig(**hp))
    raise ValueError(f"unknown architecture: {architecture}")


def load_model(path: str, device: str = "cpu"):
    """Load a released checkpoint. Returns (model, checkpoint dict)."""
    ck = torch.load(path, map_location=device)
    model = build_network(ck["architecture"], ck["hyperparameters"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
