import torch.nn.functional as F
from torch import nn
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd.variable import Variable
import torchgeometry as tgm

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size,
                 dropout, factor = 3, series=True):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.K = 2
        self.factor = factor
        self.series = series

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        inputs_ = inputs.view(inputs.shape[0], 1, -1)
        if self.series: # batchSize * K * channel * length
            mask_L = int(inputs_.shape[2] / self.factor)
            shape = (inputs_.shape[0], self.K, inputs_.shape[1], mask_L)
            self.mask = Variable(torch.zeros(shape).cuda(), requires_grad=True)
            # mask rescale
            m1 = self.mask.view(inputs_.shape[0] * self.K, 1, inputs_.shape[1], -1)
            m_scale = F.interpolate(m1, size=(inputs_.shape[1], inputs_.shape[2]), mode="bilinear")
            m_scale = m_scale.view(inputs_.shape[0], self.K, inputs_.shape[1], inputs_.shape[2])

            inputs_hat = perturbation(inputs_, 'noise')
            data_X = torch.zeros((inputs_.shape[0], self.K, inputs_.shape[1], inputs_.shape[2]))
            data_X[:, 0, ...], data_X[:, 1, ...] = inputs_, inputs_hat
            data_X = data_X.cuda()
            m1 = torch.sigmoid(m_scale)
            sum_masks = m1.sum(1, keepdim=True)
            m1 = m1 / sum_masks
            mixed_data = m1 * data_X
            inputs_ = mixed_data.sum(1)

        y1 = self.tcn(inputs_)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)

def perturbation(X, method, std=0.2, mean=0.):
    img_shape = X.shape
    if method == 'noise':
        noise = torch.randn(img_shape) * std + mean
        noise = noise.cuda()
        X = X + noise
    elif method == 'blur':
        X = torch.unsqueeze(X, 1)
        X = tgm.image.gaussian_blur(X, (1, 3), (0.01, 0.3))
        X = torch.squeeze(X, 1)
    return X

class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
