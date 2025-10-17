import torch
import torch.nn as nn

from encoder.stgcn.layers import *

class STGCN(nn.Module):
    def __init__(self,
                 n_hist,
                 Ks,
                 Kt,
                 blocks,
                 kernels,
                 dropout=0.0,
                 act_func='glu'):
        """
        :param n_hist: steps of history
        :param Ks: spatial conv kernel size
        :param Kt: temporal conv kernel size
        :param blocks: [input channels, hidden channels, output channels]
        :param kernels: graph kernel list for each block
        :param dropout: probability of dropout
        :param act_func: activation function
        """
        super(STGCN, self).__init__()

        self.n_hist = n_hist
        self.Ks = Ks
        self.Kt = Kt
        self.blocks = blocks
        self.kernel = kernels
        self.dropout = dropout
        self.act_func = act_func

        self.st_blocks = nn.ModuleList()

        Ko = n_hist
        for i, ch in enumerate(blocks):
            block = STConvBlock(Ks, Kt, ch, self.kernel, dropout=dropout, act_func=act_func)
            self.st_blocks.append(block)
            Ko -= 2 * (Kt - 1)

        if Ko <= 1:
            raise ValueError(f"[ERROR] Kernel size Ko must be greater than 1, but received {Ko}")

        c_last_out = self.blocks[-1][-1]

        self.output_layer = OutputLayer(Ko, c_last_out, n=c_last_out, act_func='glu')

    def forward(self, x):
        """
        :param x: tensor of shape [B, T, N, C]
        """
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, T, N]

        out = x
        for block in self.st_blocks:
            out = block(out)

        y = self.output_layer(out)

        y = y.permute(0, 2, 3, 1).contiguous()
        return y