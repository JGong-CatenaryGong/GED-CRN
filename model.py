import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
        
class GED_CNN1(nn.Module):
    def __init__(self, box_size, batch, with_res):
        super(GED_CNN1, self).__init__()
        self.box_size = box_size
        self.batch = batch

        self.with_res = True

        self.convblock1 = nn.Sequential(
            nn.Conv3d(2, 4, 5, 1, 2),
            nn.LeakyReLU(),
        )

        self.middleblock = nn.Sequential(
            nn.Conv3d(4, 8, 5, 1, 2, bias = True),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(8, 4, 5, 1, 2, bias = True),
            nn.LeakyReLU(),
        )

        self.deconvblock1 = nn.Sequential(
            nn.ConvTranspose3d(4, 1, 5, 1, 2),
            nn.ReLU(),
        )

        self.bn = nn.BatchNorm3d(3)

    def forward(self, x, pot):
        pot = pot.to(torch.float32)
        x = x.to(torch.float32)
        if self.with_res:
            x1 = self.convblock1(torch.concatenate((x, pot), dim = 1))
            x2 = self.middleblock(x1)
            x3 = self.deconvblock1(x2 + x1)
        else:
            x1 = self.convblock1(torch.concatenate((x, pot), dim = 1))
            x2 = self.middleblock(x1)
            x3 = self.deconvblock1(x2)
        return x3
    
class GradientDifferenceLoss(nn.Module):
    def __init__(self):
        super(GradientDifferenceLoss, self).__init__()

    def forward(self, t1, t2):
        grad_t1_x = torch.abs(t1[:, :, 1:, :, :] - t1[:, :, :-1, :, :])
        grad_t2_x = torch.abs(t2[:, :, 1:, :, :] - t2[:, :, :-1, :, :])

        grad_t1_y = torch.abs(t1[:, :, :, 1:, :] - t1[:, :, :, :-1, :])
        grad_t2_y = torch.abs(t2[:, :, :, 1:, :] - t2[:, :, :, :-1, :])

        grad_t1_z = torch.abs(t1[:, :, :, :, 1:] - t1[:, :, :, :, :-1])
        grad_t2_z = torch.abs(t2[:, :, :, :, 1:] - t2[:, :, :, :, :-1])

        grad_diff_x = torch.abs(grad_t1_x - grad_t2_x)
        grad_diff_y = torch.abs(grad_t1_y - grad_t2_y)
        grad_diff_z = torch.abs(grad_t1_z - grad_t2_z)

        loss = torch.mean(grad_diff_x) + torch.mean(grad_diff_y) + torch.mean(grad_diff_z)

        return loss
    