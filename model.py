from typing import Union, Optional, List, Tuple, Dict, Iterable
import cv2

import numpy as np
import torch.nn as nn
from torchvision import models
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import center_of_mass
from fnmatch import fnmatch
# from nest import register

from prm.modules import PeakResponseMapping


import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def finetune(
    model: nn.Module, 
    base_lr: float, 
    groups: Dict[str, float], 
    ignore_the_rest: bool = False,
    raw_query: bool = False) -> List[Dict[str, Union[float, Iterable]]]:
    """Fintune.
    """

    print('finetune------->> ', base_lr, groups, ignore_the_rest, raw_query)

    parameters = [dict(params=[], names=[], query=query if raw_query else '*'+query+'*', lr=lr*base_lr) for query, lr in groups.items()]
    rest_parameters = dict(params=[], names=[], lr=base_lr)
    for k, v in model.named_parameters():
        for group in parameters:
            if fnmatch(k, group['query']):
                group['params'].append(v)
                group['names'].append(k)
            else:
                rest_parameters['params'].append(v)
                rest_parameters['names'].append(k)
    if not ignore_the_rest:
        parameters.append(rest_parameters)
    for group in parameters:
        group['params'] = iter(group['params'])
    return parameters


class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes):
        super(FC_ResNet, self).__init__()

        # feature encoding
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classifier
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




def fc_resnet50(num_classes: int = 20, pretrained: bool = True) -> nn.Module:
    """FC ResNet50.
    """
    model = FC_ResNet(models.resnet50(pretrained), num_classes)
    return model


def peak_response_mapping(
    backbone: nn.Module,
    enable_peak_stimulation: bool = True,
    enable_peak_backprop: bool = True,
    win_size: int = 3,
    sub_pixel_locating_factor: int = 1,
    filter_type: Union[str, int, float] = 'median') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping(
        backbone, 
        enable_peak_stimulation = enable_peak_stimulation,
        enable_peak_backprop = enable_peak_backprop, 
        win_size = win_size, 
        sub_pixel_locating_factor = sub_pixel_locating_factor, 
        filter_type = filter_type)
    return model