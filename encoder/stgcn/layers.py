import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_norm(x, eps=1e-6):
    """
    :param x: tensor of shape (N, T, N, C)
    """
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.var(dim=(2, 3), keepdim=True, unbiased=False).sqrt()

    return (x - mean) / (std + eps)


class GConv(nn.Module):
    """
    Spectral-based graph convolution
    """
    def __init__(self, Ks, c_in, c_out, kernel):
        super(GConv, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out

        self.theta = nn.Parameter(torch.Tensor(Ks * c_in, c_out))
        nn.init.xavier_uniform_(self.theta)

        self.kernel = kernel.to(torch.float32)
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x):
        """
        :param x: tensor of shape (B, N, C_in)
        :return: tensor of shape (B, N, C_out)
        """
        batch_size, n, _ = x.shape

        x_tmp = x.permute(0, 2, 1).reshape(batch_size * self.c_in, n)                   # [B, C_in, N] -> [B * C_in, N]
        x_mul = torch.matmul(x_tmp, self.kernel)                                        # [B * C_in, Ks * N]
        x_mul = x_mul.reshape(batch_size, self.c_in, self.Ks, n)                        # [B, C_in, Ks, N]
        x_ker = x_mul.permute(0, 3, 1, 2).reshape(batch_size * n, self.c_in * self.Ks)  # [B * N, C_in * Ks]
        x_gconv = torch.matmul(x_ker, self.theta) + self.bias                           # [B, N, C_out]
        x_gconv = x_gconv.reshape(batch_size, n, self.c_out)                            # [B, C_out, N]

        return x_gconv


class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, act_func='glu'):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func

        self.conv = nn.Conv2d(in_channels=c_in,
                              out_channels=2 * c_out if act_func == 'glu' else c_out,
                              kernel_size=(Kt, 1),
                              stride=(1, 1),
                              padding=(0, 0),
                              bias=True)

    def forward(self, x):
        self.conv = self.conv.to(x.device)
        if self.c_in < self.c_out:
            pad_c = self.c_out - self.c_in
            zeros_c = torch.zeros(x.size(0), pad_c, x.size(2), x.size(3), dtype=x.dtype, device=x.device)
            x_input = torch.cat([x, zeros_c], dim=1)
        elif self.c_in > self.c_out:
            w_input = nn.Conv2d(in_channels=self.c_in,
                                out_channels=self.c_out,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=(0, 0),
                                bias=False).to(x.device)
            x_input = w_input(x)
        else:
            x_input = x
        x_conv = self.conv(x)
        x_input = x_input[:, :, self.Kt - 1:, :]

        if self.act_func == 'glu':
            A = x_conv[:, :self.c_out, :, :]
            B = x_conv[:, self.c_out:, :, :]
            return (A + x_input) * torch.sigmoid(B)
        else:
            return F.relu(x_conv + x_input)


class SpatioConvLayer(nn.Module):
    """
    Spatial convolution layer
    """
    def __init__(self, Ks, c_in, c_out, kernel):
        """
        :param Ks: kernel size of spatial convolution
        :param c_in: input channels
        :param c_out: output channels
        :param kernel: [n_route, Ks * n_route] graph kernel
        """
        super(SpatioConvLayer, self).__init__()

        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.gconv = GConv(Ks, c_in, c_out, kernel)

        if c_in != c_out:
            self.bottleneck = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        else:
            self.bottleneck = None

    def forward(self, x):
        """
        :param x: tensor of shape [B, C_in, T, N]
        """
        B, C_in, T, N = x.shape

        # bottleneck
        if self.bottleneck is not None and C_in > self.c_out:
            x_input = self.bottleneck(x)
        elif self.bottleneck is not None and C_in < self.c_out:
            pad_channels = self.c_out - self.c_in
            zeros = torch.zeros((B, pad_channels, T, N), device=x.device, dtype=x.dtype)
            x_input = torch.cat((x, zeros), dim=1)
            x_input = x_input
        else:
            x_input = x

        # gconv for each time
        out_list = []
        for t in range(T):
            x_t = x[:, :, t, :].permute(0, 2, 1)    # [B, N, C_in]
            out_t = self.gconv(x_t)                 # [B, N, C_out]
            out_t = out_t.permute(0, 2, 1)          # [B, C_out, N]
            out_list.append(out_t)

        x_gc = torch.stack(out_list, dim=2)
        return F.relu(x_gc + x_input)


class STConvBlock(nn.Module):
    """
    Spatio-temporal convolution block:
        - temporal conv (GLU)
        - spatial gconv
        - temporal conv
        - layer norm
        - dropout
    """
    def __init__(self,
                 Ks,
                 Kt,
                 channels,
                 kernel,
                 dropout=0.0,
                 act_func='glu'):
        super(STConvBlock, self).__init__()
        c_si, c_t, c_oo = channels  # input channels, hidden channels, output channels

        # block in
        self.tconv1 = TemporalConvLayer(Kt, c_si, c_t, act_func=act_func)
        self.sconv = SpatioConvLayer(Ks, c_t, c_t, kernel)

        # block out
        self.tconv2 = TemporalConvLayer(Kt, c_t, c_oo, act_func=act_func)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: tensor of shape [B, C_in, T, N]
        """
        # block in
        x_s = self.tconv1(x)
        x_t = self.sconv(x_s)

        # block out
        x_o = self.tconv2(x_t)

        # layer norm
        x_ln = layer_norm(x_o)
        x_ln = self.dropout(x_ln)

        return x_ln


class FullyConvLayer(nn.Module):
    """
    Fully convolution layer
    """
    def __init__(self, n, channel):
        """
        [B, 1, N, C] -> [B, 1, N, 1]
        """
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        """
        :param x: tensor of shape [B, 1, N, C]
        :return: tensor of shape [B, 1, N, 1]
        """
        out = self.conv(x)
        return out


class OutputLayer(nn.Module):
    """
    Output layer
    """
    def __init__(self, T, channel, n, act_func='glu'):
        super(OutputLayer, self).__init__()
        self.T = T
        self.channel = channel
        self.n = n
        self.act_func = act_func

        # the first temporal conv
        self.tconv_in = TemporalConvLayer(T, channel, channel, act_func=self.act_func)

        # the second temporal conv
        self.tconv_out = TemporalConvLayer(1, channel, channel, act_func='sigmoid')

        # fully connected
        self.fc = FullyConvLayer(n, channel)

    def forward(self, x):
        """
        :param x: tensor of shape [B, channels, T, N]
        :return: tensor of shape [B, 1, 1, N]
        """
        x_i = self.tconv_in(x)          # [B, C, T - T + 1, N] => [B, C, 1, N]
        x_ln = layer_norm(x_i)
        x_o = self.tconv_out(x_ln)      # [B, C, 1, N] => [B, C, 1, N]
        x_fc = self.fc(x_o)             # [B, 1, 1, N]
        return x_fc                     # [B, 1, 1, N]
