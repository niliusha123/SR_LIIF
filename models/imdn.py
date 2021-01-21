# https://github.com/Zheng222/IMDN
import torch.nn as nn
from . import block as B
import torch
from argparse import Namespace


# AI in RTC Image Super-Resolution Algorithm Performance Comparison Challenge (Winner solution)
class IMDN(nn.Module):
    def __init__(self, args):
        super(IMDN, self).__init__()
        self.args = args

        fea_conv = [B.conv_layer(args.n_colors, args.n_feats, kernel_size=3)]
        rb_blocks = [B.IMDModule_speed(in_channels=args.n_feats) for _ in range(args.num_modules)]
        LR_conv = B.conv_layer(args.n_feats, args.n_feats, kernel_size=1)

        if args.no_upsampling:
            self.out_dim = args.n_feats
        else:
            self.out_dim = args.out_nc
            upsample_block = B.pixelshuffle_block
            # define tail module
            upsampler = upsample_block(args.n_feats, args.out_nc, upscale_factor=args.upscale)

        if args.no_upsampling:
            self.model = B.sequential(*fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)))
        else:
            self.model = B.sequential(*fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
                                  *upsampler)

    def forward(self, input):
        output = self.model(input)
        return output


def make_imdn(args_use, n_colors=3, n_feats=64, num_modules=5, out_nc=3, upscale=2):
    args = Namespace()
    args.n_colors = 3
    args.num_modules = num_modules
    args.n_feats = n_feats

    args.upscale = upscale
    args.no_upsampling = args_use.no_upsampling   # True

    args.out_nc = out_nc
    return IMDN(args)

