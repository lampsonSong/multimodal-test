# created by lampson.song @ 20210514
# to create efficientnetv2s separately

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def load_pretrained(model, pretrained):
    pretrained_state_dict = torch.load(pretrained)
    pretrained_keys = []

    for k,v in pretrained_state_dict.items():
        pretrained_keys.append(k)

    new_model = OrderedDict()

    for k,v in model.state_dict().items():
        if k in pretrained_keys and v.shape == pretrained_state_dict[k].shape:
            new_model[k] = pretrained_state_dict[k]
        else:
            new_model[k] = v

    return new_model

class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x = x * torch.sigmoid(x)
            return x
        else:
            return x * torch.sigmoid(x)

def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type == 'catavgmax':
        return 2
    else:
        return 1

def select_adaptive_pool2d(x, pool_type='avg', output_size=1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool2d(x, output_size)
    else:
        assert False, 'Invalid pool type %s' % pool_type
    return x

class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='fast', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or '' # convert other falsy values to empty string for consistent TS typing
        self.flatten = flatten
        if pool_type == '':
            self.pool = nn.Identity() # pass through
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        else:
            assert False, 'Invalid pool type : %s' % pool_type

    def is_identity(self):
        return self.pool_type == ''

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x

    def feat_mul(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                + 'pool_type' + self.pool_type \
                + ', flatten=' + str(self.flatten) + ')'

def _make_divisible(v, divisor, min_value=None):
    """ _make_divisible(v, divisor, min_value=None)
    Th
    It This function is taken from the original tf repo
    It It ensures that all layers have a channel number that is divisible by 8
    ht It can be seen here:
    :p https://github.com/tensorflow/models/blob/research/slim/nets/mobilenet/mobilenet.py
    :p :param v:
    :p :param divisor:
    :r :param min_value
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, hidden_dim, se_ratio=0.25, reduced_base_chs=None,
            act_layer=nn.ReLU, gate_fn=torch.sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.conv_reduce = nn.Conv2d(hidden_dim, reduced_chs, 1, bias=True)
        self.act1 = SiLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, hidden_dim, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2,3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)

class EdgeResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio, use_se):
        super(EdgeResidual, self).__init__()
        assert stride in [1,2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == outp

        # fused
        self.conv_exp = nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = SiLU(inplace=True)
        # pw-linear
        self.conv_pwl = nn.Conv2d(hidden_dim, outp, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(outp)

    def forward(self, x):
        shortcut = x

        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)

        if self.identity:
            x += shortcut

        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio, use_se):
        super(InvertedResidual, self).__init__()
        assert stride in [1,2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == outp

        #pw
        self.conv_pw = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = SiLU(inplace=True)
        #dw
        self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = SiLU(inplace=True)
        self.se = SqueezeExcite(inp, hidden_dim)
        # pw-linear
        self.conv_pwl = nn.Conv2d(hidden_dim, outp, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(outp)

    def forward(self, x):
        shortcut = x

        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.identity:
            x += shortcut

        return x

class EfficientNet_v2s(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(EfficientNet_v2s, self).__init__()

        # in_c, out_c, kernel_size, stride, padding
        conv_stem_cfg = [3, 24, (3,3), (2,2), (1,1)]
        # eps, momentum, affine(bool), track_running_stats(bool)
        bn_cfg = [1e-5, 0.1, True, True]

        # first conv_stem
        self.conv_stem = nn.Conv2d(conv_stem_cfg[0], conv_stem_cfg[1], kernel_size=conv_stem_cfg[2], stride=conv_stem_cfg[3], padding=conv_stem_cfg[4], bias=False)
        self.bn1 = nn.BatchNorm2d(24, eps=bn_cfg[0], momentum=bn_cfg[1], affine=bn_cfg[2], track_running_stats=bn_cfg[3])
        self.act1 = SiLU(inplace=True)

        # times, in_channels, number, stride, use_se
        self.blocks_cfg = [
                [1, 24, 2, 1, 0],
                [4, 48, 4, 2, 0],
                [4, 64, 4, 2, 0],
                [4, 128, 6, 2, 1],
                [6, 160, 9, 1, 1],
                [6, 272, 15, 2, 1],
                ]
        in_c = _make_divisible(24 * width_mult, 8)

        body_layers = []
        for t, c, n, s, use_se in self.blocks_cfg:
            layers = []
            out_c = _make_divisible(c * width_mult, 8)
            for i in range(n):
                if not use_se:
                    layers.append(EdgeResidual(in_c, out_c, s if i == 0 else 1, t, use_se))
                else:
                    layers.append(InvertedResidual(in_c, out_c, s if i == 0 else 1, t, use_se))
                in_c = out_c

            body_layers.append(nn.Sequential(*layers))

        self.blocks = nn.Sequential(*body_layers)

        self.conv_head = nn.Conv2d(272, 1792, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(1792, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.act2 = SiLU(inplace=True)
        self.global_pool = SelectAdaptivePool2d(pool_type='avg', flatten=True)
        self.classifier = nn.Linear(1792, num_classes, bias=True)

        self.latlayer12 = nn.Conv2d(1792, 160, kernel_size=1, stride=1, padding=0)

        self.finalfc1792 = nn.Linear(1792, num_classes)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        up_feat = F.interpolate(x, (H,W))

        return torch.add(up_feat, y)

    def forward(self, x):
        endpoints = self.extract_endpoints(x)

        feat1 = endpoints['reduction_5']

        bs = feat1.size()[0]
        feat1 = torch.nn.functional.adaptive_avg_pool2d(feat1, (1,1))
        feat1 = feat1.view(bs, -1)
        pred_feat1 = self.finalfc1792(feat1)

        return pred_feat1

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1,2,3,4,5]

        Args:
            inputs (tensor): Input tensor.

        """
        endpoints = dict()

        # Stem
        x = self.act1(self.bn1(self.conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        x = self.act2(self.bn2(self.conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x

        return endpoints


def get_efficientnet_v2s(num_classes=1000, pretrained=None):
    # pretrained path : https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_v2s_ra2_288-a6477665.pth
    model = EfficientNet_v2s(num_classes=num_classes)

    if pretrained:
        model.load_state_dict(load_pretrained(model, pretained))

    return model

if __name__ == '__main__':
    model = EfficientNet_v2s(num_classes=3)

    x = torch.zeros(1,3,288,288)
    pretrained = "./efficientnet_models/efficientnet_v2s_ra2_288-a6477665.pth"

    model.load_state_dict(load_pretrained(model, pretrained))

    y = model(x)

    print(y)
