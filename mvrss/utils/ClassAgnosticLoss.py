import torch
import torch.nn as nn
from torch.nn.functional import one_hot



class OCLoss(nn.Module):
    def __init__(self, delta=0.6, epsilon=1e-07):
        super(OCLoss, self).__init__()
        self.delta = delta
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        
        back_g = (1 - y_pred[:,0,:,:]) * cross_entropy[:,0,:,:]
        back_g =  (1 - self.delta) * back_g
        fore_g = (1 - y_pred[:,1,:,:]) * cross_entropy[:,1,:,:]
        fore_g = self.delta * fore_g
        loss = torch.mean(torch.sum(torch.stack([back_g, fore_g], axis=-1), axis=-1))

        return loss
    
    
class CLLoss(nn.Module):
    def __init__(self, epsilon=1e-07):
        super(CLLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
            
        tp = torch.sum(y_true * y_pred, axis=[2, 3])
        fn = torch.sum(y_true * (1-y_pred), axis=[2, 3])
        fp = torch.sum((1-y_true) * y_pred, axis=[2, 3])
        intersection = (tp + self.epsilon)/(tp + fn + fp + self.epsilon)

        backg_intersection = (1-intersection[:,0]) 
        foreg_intersection = (1-intersection[:,1]) 
      
        # Average class scores
        loss = torch.mean(torch.stack([backg_intersection,foreg_intersection], axis=-1))
        return loss

    
    
    
class CALoss(nn.Module):
    def __init__(self,  delta=0.6, global_weight = 1., device = 'cuda'):
        super(CALoss, self).__init__()
        self.delta = delta
        self.global_weight = global_weight
        self.device = device

    def forward(self, y_pred, y_true):
        y_pred = one_hot(torch.clamp(torch.argmax(y_pred,dim=1),min=0,max=1)).permute(0,-1,1,2)
        y_true = one_hot(torch.clamp(y_true,min=0,max=1)).permute(0,-1,1,2)

        if y_true.shape[1] == 1:
            y_true = torch.cat((y_true,torch.zeros(y_true.shape).to(self.device)), dim=1)
        if y_pred.shape[1] == 1:
            y_pred = torch.cat((y_pred,torch.zeros(y_pred.shape).to(self.device)),dim=1)

        ocloss = OCLoss(delta=self.delta)(y_pred, y_true)
        cllos = CLLoss()(y_pred, y_true)

        return ((cllos + ocloss) * 0.5)*self.global_weight
    

