import sys
import torch
import torch.nn as nn
from utils import *
from bayes_layer import BayesianConv2D
from bayes_layer import BayesianLinear

class Sequential(nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input, sample=False):
        for module in self._modules.values():
            if isinstance(module, BayesianConv2D) or isinstance(module, BayesianLinear):
                input = module(input, sample=sample)
            else:
                input = module(input)
        return input
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, ratio = 0.25):
    """3x3 convolution with padding"""
    return BayesianConv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, ratio = ratio)

def conv1x1(in_planes, out_planes, stride=1, ratio = 0.25):
    """1x1 convolution"""
    return BayesianConv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, ratio = ratio)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, ratio = 0.25, sample = False):
        super(BasicBlock, self).__init__()
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.sample = sample
        self.conv1 = conv3x3(inplanes, planes, stride, ratio = ratio)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, ratio = ratio)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, sample = False):
        identity = x
        out = self.conv1(x, sample)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, sample)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x, sample)

        out += identity
        out = self.relu(out)

        return out


class BayesianConvNetwork(nn.Module):
    def __init__(self, inputsize, taskcla, ratio = 0.25):
        super(BayesianConvNetwork, self).__init__()
        self.ratio = ratio
        self.taskcla = taskcla
        
        self.inplanes = 32
        self.blocks = 2
        
        self.conv1 = BayesianConv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 32, self.blocks)
        self.layer2 = self._make_layer(BasicBlock, 64, self.blocks, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, self.blocks, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, self.blocks, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last=torch.nn.ModuleList()
        
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        
        downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, ratio = self.ratio),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, ratio = self.ratio))
        self.inplanes = planes * block.expansion
        layers.append(block(self.inplanes, planes, ratio = self.ratio))

        return Sequential(*layers)
    
    def forward(self, x, sample = False):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x,sample))))
        x = self.layer1(x, sample = sample)
        x = self.layer2(x, sample = sample)
        x = self.layer3(x, sample = sample)
        x = self.layer4(x, sample = sample)

        x = self.avgpool(x)
        
        x = x.view(x.shape[0],-1)
        y = []
        for t,i in self.taskcla:
            y.append(self.last[t](x))
        
        return y
    