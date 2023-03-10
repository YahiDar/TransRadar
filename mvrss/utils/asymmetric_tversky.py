import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from mvrss.utils import MVRSS_HOME
from mvrss.loaders.dataloaders import Rescale, Flip, HFlip, VFlip
from torch.nn.functional import one_hot

# Helper function to enable loss function to be flexibly used for 
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]

    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


    
    
class AsymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07, global_weight = 1.):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.global_weight = global_weight
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = one_hot(torch.clamp(torch.argmax(y_pred,dim=1),min=0,max=1)).permute(0,-1,1,2)
        y_true = one_hot(torch.clamp(y_true,min=0,max=1)).permute(0,-1,1,2)
        if y_true.shape[1] == 1:
            y_true = torch.cat((y_true,torch.zeros(y_true.shape).cuda()), dim=1)
        if y_pred.shape[1] == 1:
           y_pred = torch.cat((y_pred,torch.zeros(y_pred.shape).cuda()),dim=1)

        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0]) 
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma) 

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
        loss = loss*self.global_weight
        return loss
    