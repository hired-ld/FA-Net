import torch.nn as nn
import torch
import numpy as np
from typing import Any,Optional, Sequence, Union, Tuple
from collections.abc import Iterable

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool

def issequenceiterable(obj: Any) -> bool:
    """
    Determine if the object is an iterable sequence and is not a string.
    """
    try:
        if hasattr(obj, "ndim") and obj.ndim == 0:
            return False  # a 0-d tensor is not iterable
    except Exception:
        return False
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

def ensure_tuple_rep(tup: Any, dim: int) -> Tuple[Any, ...]:
    """
    Returns a copy of `tup` with `dim` values by either shortened or duplicated input.

    Raises:
        ValueError: When ``tup`` is a sequence and ``tup`` length is not ``dim``.

    """
    if isinstance(tup, torch.Tensor):
        tup = tup.detach().cpu().numpy()
    if isinstance(tup, np.ndarray):
        tup = tup.tolist()
    if not issequenceiterable(tup):
        return (tup,) * dim
    if len(tup) == dim:
        return tuple(tup)

    raise ValueError(f"Sequence must have length {dim}, got {len(tup)}.")

class TwoConv(nn.Sequential):
    """two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()

        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)

class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        is_pad: bool = True,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the encoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.
            is_pad: whether to pad upsampling features to fit features from encoder. Defaults to True.

        """
        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)
        self.is_pad = is_pad

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)
        if x_e is not None:
            if self.is_pad:
                # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
                dimensions = len(x.shape) - 2
                sp = [0] * (dimensions * 2)
                for i in range(dimensions):
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x

class UpCat_IA(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        is_pad: bool = True,
        atten: str = "ag"
    ):

        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            scale_factor=2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )

        self.convs = TwoConv(spatial_dims, out_chns*2, out_chns, act, norm, bias, dropout)
        self.is_pad = is_pad

        print('interactive attention')
        self.ag = InteractiveAttention(spatial_dims, out_chns, out_chns,out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)
        if x_e is not None:
            if self.is_pad:
                # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
                dimensions = len(x.shape) - 2
                sp = [0] * (dimensions * 2)
                for i in range(dimensions):
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate")

            x_e = self.ag(x_e,x_0)

            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)

        else:
            x = self.convs(x_0)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        dimensions: Optional[int] = None,
    ):
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions

        self.conv_0 = TwoConv(spatial_dims, in_channels, 32, act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, 32, 64, act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, 64, 128, act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, 128, 256, act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, 256, out_channels, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor):

        x_1 = self.conv_0(x)
        x_2 = self.down_1(x_1)
        x_3 = self.down_2(x_2)
        x_4 = self.down_3(x_3)
        x_5 = self.down_4(x_4)

        return [x_1,x_2,x_3,x_4,x_5]

class FusionModule(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
    ):
        super().__init__()
        self.conv = TwoConv(spatial_dims, in_channels, out_channels, act, norm, bias, dropout)

    def forward(self, x1: torch.Tensor,x2: torch.Tensor,x3: Optional[torch.Tensor]):
        if x3 is not None:
            x = self.conv(torch.cat([x1, x2,x3], dim=1))
        else:
            x = self.conv(torch.cat([x1, x2], dim=1))
        return x


class InteractiveAttention(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels_1: int = 1,
        in_channels_2: int = 1,
        out_channels: int = 2,
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
    ):
        super().__init__()

        if in_channels_2!=out_channels:
            raise ValueError(f"注意力机制中输入通道不对，请检查.")

        self.conv_0 = Convolution(spatial_dims, out_channels, out_channels, kernel_size=1,act=act, norm=norm, dropout=dropout, bias=bias, padding=0)
        self.conv_1 = Convolution(spatial_dims, in_channels_1, in_channels_2, kernel_size=1, act=act, norm=norm, dropout=dropout,
                             bias=bias, padding=0)
        self.sigmoid_0=nn.Sigmoid()
        self.sigmoid_1 = nn.Sigmoid()
    def forward(self, x1: torch.Tensor,x2: torch.Tensor):
        u1=self.conv_0(x1)
        u2=self.conv_1(x2)
        u = x2*self.sigmoid_0(u1)+x1*self.sigmoid_1(u2)
        out = u
        return out


class FA_Net(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        atten_g: str = 'true',
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        """
        Based on Basic UNet from MONAI PACKAGE.
        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        if in_channels != 4:
            raise ValueError(f"输入通道不对，请检查.")

        self.featrue_extract_1 = Encoder(spatial_dims,1,256, act, norm, bias, dropout) # 危及器官与靶区
        self.featrue_extract_2 = Encoder(spatial_dims, 2, 256, act, norm, bias, dropout)  # CT图像
        self.featrue_extract_3 = Encoder(spatial_dims, 1, 256, act, norm, bias, dropout)  # 开野信息

        self.fusion_1=FusionModule(spatial_dims,32*3,64, act, norm, bias, dropout)
        self.fusion_2 = FusionModule(spatial_dims, 64*3, 128, act, norm, bias, dropout)
        self.fusion_3 = FusionModule(spatial_dims, 128*3, 256, act, norm, bias, dropout)
        self.fusion_4 = FusionModule(spatial_dims, 256*3, 512, act, norm, bias, dropout)
        self.fusion_5 = FusionModule(spatial_dims, 256*3, 512, act, norm, bias, dropout)


        self.upcat_4 = UpCat_IA(spatial_dims, 512, 512, 512, act, norm, bias, dropout, upsample,halves=False,atten="ia")
        self.upcat_3 = UpCat_IA(spatial_dims, 512, 256, 256, act, norm, bias, dropout, upsample,atten="ia")
        self.upcat_2 = UpCat_IA(spatial_dims, 256, 128, 128, act, norm, bias, dropout, upsample,atten="ia")
        self.upcat_1 = UpCat_IA(spatial_dims, 128, 64, 64, act, norm, bias, dropout, upsample, atten="ia")

        self.final_conv = Conv["conv", spatial_dims](64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `spatial_dims`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        # print(x.shape)
        e_1 = self.featrue_extract_1(x[:,0,...].unsqueeze(1))
        e_2 = self.featrue_extract_2(x[:,1:3,...])
        e_3 = self.featrue_extract_3(x[:,3,...].unsqueeze(1))


        f_1 = self.fusion_1(e_1[0],e_2[0],e_3[0])
        f_2 = self.fusion_2(e_1[1], e_2[1], e_3[1])
        f_3 = self.fusion_3(e_1[2], e_2[2], e_3[2])
        f_4 = self.fusion_4(e_1[3], e_2[3], e_3[3])
        f_5 = self.fusion_5(e_1[4], e_2[4], e_3[4])

        u4 = self.upcat_4(f_5, f_4)
        u3 = self.upcat_3(u4, f_3)
        u2 = self.upcat_2(u3, f_2)
        u1 = self.upcat_1(u2, f_1)

        logits = self.final_conv(u1)
        return logits



class AENet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        mode: str = 'train',
        features: Sequence[int] = (64, 128, 256, 512, 512, 64),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        dimensions: Optional[int] = None,
    ):
        """
        Based on Basic UNet from MONAI PACKAGE.
        """
        super().__init__()
        if dimensions is not None:
            spatial_dims = dimensions
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")
        self.mode=mode


        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        self.up_4 = torch.nn.Sequential(
            UpSample(spatial_dims, fea[4], fea[3], mode=upsample, pre_conv="default", interp_mode="linear",  align_corners=True),
            TwoConv(spatial_dims, fea[3],  fea[3], act, norm, bias, dropout)
        )
        self.up_3 = torch.nn.Sequential(
            UpSample(spatial_dims, fea[3], fea[2], mode=upsample, pre_conv="default", interp_mode="linear",  align_corners=True),
            TwoConv(spatial_dims, fea[2],  fea[2], act, norm, bias, dropout)
        )
        self.up_2 = torch.nn.Sequential(
            UpSample(spatial_dims, fea[2], fea[1], mode=upsample, pre_conv="default", interp_mode="linear",  align_corners=True),
            TwoConv(spatial_dims, fea[1],  fea[1], act, norm, bias, dropout)
        )
        self.up_1 = torch.nn.Sequential(
            UpSample(spatial_dims, fea[1], fea[0], mode=upsample, pre_conv="default", interp_mode="linear",  align_corners=True),
            TwoConv(spatial_dims, fea[0],  fea[0], act, norm, bias, dropout)
        )


        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `spatial_dims`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.up_4(x4)
        u3 = self.up_3(u4)
        u2 = self.up_2(u3)
        u1 = self.up_1(u2)

        logits = self.final_conv(u1)

        if self.mode == 'train':
            return logits
        else:
            return [x0,x1,x2,x3,x4]
