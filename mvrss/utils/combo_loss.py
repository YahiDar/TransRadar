import json
import numpy as np
import torch
import torch.nn as nn
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


class ComboLoss(nn.Module):
    
    def __init__(self, alpha = 0.5, beta = None, epsilon=1e-07, delta = 0.7, global_weight = 1.):
        super(ComboLoss, self).__init__()
        self.epsilon = epsilon
        self.global_weight = global_weight
        self.delta = delta
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        # print(y_pred.shape)
        y_pred = one_hot(torch.clamp(torch.argmax(y_pred,dim=1),min=0,max=1)).permute(0,-1,1,2)
        y_true = one_hot(torch.clamp(y_true,min=0,max=1)).permute(0,-1,1,2)
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())
        
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)
        dice = torch.mean(dice_class)


        cross_entropy = -y_true * torch.log(y_pred)
        if self.beta is not None:
            beta_weight = np.array([self.beta, 1-self.beta])
            cross_entropy = beta_weight*cross_entropy
        cross_entropy = torch.mean(torch.sum(cross_entropy, axis=1))
        if self.alpha is not None:
            combo_loss = (self.alpha*cross_entropy) - ((1-self.alpha)*dice)
        else:
            combo_loss = cross_entropy - dice
        return (combo_loss)*self.global_weight
