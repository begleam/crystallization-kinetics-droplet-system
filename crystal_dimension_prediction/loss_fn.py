import torch
import torch.nn as nn

class MultiTaskMSELoss(nn.Module):
    def __init__(self):
        super(MultiTaskMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='sum')
    def forward(self, preds0, preds1, preds2, preds3, preds4, labels0, labels1, labels2, labels3, labels4):
        w_loss = self.criterion(preds0, labels0)
        l_loss = self.criterion(preds1, labels1)
        theta_loss = self.criterion(preds2, labels2)
        phi_loss = self.criterion(preds3, labels3)
        gamma_loss = self.criterion(preds4, labels4)
        total_loss = w_loss + l_loss + theta_loss + phi_loss + gamma_loss

        return total_loss, w_loss.item(), l_loss.item(), theta_loss.item(), phi_loss.item(), gamma_loss.item()

def get_mae(preds, labels):
    return torch.sum(torch.abs(preds - labels))