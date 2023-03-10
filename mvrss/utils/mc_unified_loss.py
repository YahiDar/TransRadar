import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from mvrss.utils import MVRSS_HOME
from mvrss.losses.soft_dice import SoftDiceLoss
from mvrss.losses.coherence import CoherenceLoss
from mvrss.loaders.dataloaders import Rescale, Flip, HFlip, VFlip
from torch.nn.functional import one_hot

def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]

    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')



class MCSymmetricFocalLoss(nn.Module):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.25
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self,weights, delta=0.7, gamma=2.,s = 8, epsilon=1e-07):
        super(MCSymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.s = s
        self.epsilon = epsilon
        self.weights = weights

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        
        gamma_0 = self.gamma + self.s(1- negative)
	# Calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:,0,:,:], self.gamma) * cross_entropy[:,0,:,:]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = (cross_entropy[:,1,:,:]*self.weights[0] + cross_entropy[:,2,:,:]*self.weights[0] + cross_entropy[:,3,:,:]*self.weights[2]) * cross_entropy[:,1,:,:]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


    
    

class SymmetricFocalTverskyLoss(nn.Module):
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
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = one_hot(torch.clamp(torch.argmax(y_pred,dim=1),min=0,max=1)).permute(0,-1,1,2)
        y_true = one_hot(torch.clamp(torch.argmax(y_true,dim=1),min=0,max=1)).permute(0,-1,1,2)

        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())
        
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0]) * torch.pow(1-dice_class[:,0], -self.gamma)
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma) 
      
        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
        return loss


class MCSymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, weights, weight=0.2, delta=0.6, gamma=0.5, global_weight = 1.):
        super(MCSymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.weights = weights
        self.global_weight = global_weight
    def forward(self, y_pred, y_true):
      # Obtain Symmetric Focal Tversky loss

      symmetric_ftl = SymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)

      # Obtain Symmetric Focal loss
      symmetric_fl = MCSymmetricFocalLoss(weights = self.weights, delta=self.delta, gamma=self.gamma)(y_pred, y_true)
      
      # Return weighted sum of Symmetrical Focal loss and Symmetric Focal Tversky loss
      if self.weight is not None:
        return ((self.weight * symmetric_ftl) + ((1-self.weight) * symmetric_fl))* self.global_weight
      else:
        return (symmetric_ftl + symmetric_fl)*self.global_weight
