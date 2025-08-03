import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

#######################################################
#             AITL Classes & Functions                #
#######################################################

class FX(nn.Module):
    def __init__(self, dropout_rate, input_dim, h_dim, z_dim):
        super(FX, self).__init__()
        self.EnE = torch.nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate))
    def forward(self, x):
        output = self.EnE(x)
        #print(output.shape)    ####torch.Size([32, 512]), torch.Size([8, 512])
        return output


class FX_MLP(nn.Module):
    def __init__(self, dropout_rate, input_dim, h_dim, z_dim):
        super(FX_MLP, self).__init__()
        self.EnE = torch.nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU()
        )
    def forward(self, x):
        output = self.EnE(x)
        return output


class MTL(nn.Module):
    def __init__(self, dropout_rate, h_dim, z_dim):
        super(MTL, self).__init__()
        print("{} and {}".format(h_dim,z_dim))
        self.Sh = nn.Linear(h_dim, z_dim)
        self.bn1 = nn.BatchNorm1d(z_dim)
        self.Drop = nn.Dropout(p=dropout_rate)
        self.Source = torch.nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(z_dim, 1))
        self.Target = torch.nn.Sequential(
            nn.Linear(z_dim, 1),
            nn.Sigmoid())

    def forward(self, S, T):
        if S is None:
            ZT = F.relu(self.Drop(self.bn1(self.Sh((T)))))
            yhat_S = None
            yhat_T = self.Target(ZT)
        elif T is None:
            ZS = F.relu(self.Drop(self.bn1(self.Sh((S)))))
            yhat_S = self.Source(ZS)
            yhat_T = None
        else:
            ZS = F.relu(self.Drop(self.bn1(self.Sh((S)))))
            ZT = F.relu(self.Drop(self.bn1(self.Sh((T)))))
            yhat_S = self.Source(ZS)
            yhat_T = self.Target(ZT)
        return yhat_S, yhat_T


class MTLP(nn.Module):    ### drug response predictor ###
    def __init__(self, dropout_rate, h_dim, z_dim):
        super(MTLP, self).__init__()
        # print("{} and {}".format(h_dim, z_dim))
        self.Sh = nn.Linear(h_dim, z_dim)
        self.bn1 = nn.BatchNorm1d(z_dim)
        self.Drop = nn.Dropout(p=dropout_rate)
        self.Predictor = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(z_dim, 1),
            nn.Sigmoid())

    def forward(self, X):
        ZX = F.relu(self.Drop(self.bn1(self.Sh((X)))))
        yhat = self.Predictor(ZX)
        return yhat
    
    
class MTLP_EMB(nn.Module):    ### drug response predictor ###
    def __init__(self, dropout_rate, h_dim, z_dim):
        super(MTLP_EMB, self).__init__()
        print("{} and {}".format(h_dim, z_dim))
        self.Sh = nn.Linear(h_dim, z_dim)
        self.bn1 = nn.BatchNorm1d(z_dim)
        self.Drop = nn.Dropout(p=dropout_rate)
        self.Predictor = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(z_dim, z_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(z_dim, 1),
            nn.Sigmoid())

    def forward(self, X):
        ZX = F.relu(self.Drop(self.bn1(self.Sh((X)))))
        yhat = self.Predictor(ZX)
        return yhat
    

class GradReverse(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -1)


def grad_reverse(x):
    return GradReverse.apply(x)

class Discriminator(nn.Module):
    def __init__(self, dropout_rate, h_dim, z_dim):
        super(Discriminator, self).__init__()
        self.D1 = nn.Linear(h_dim, 1)
        self.D1 = torch.nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(h_dim, 1))
        self.Drop1 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = grad_reverse(x)
        yhat = self.Drop1(self.D1(x))
        return torch.sigmoid(yhat)

class GP(nn.Module):  ##combine feature extractor and predictor for shap explainer
    def __init__(self, gen_model, map_model):
        super(GP,self).__init__()
        self.gen_model=gen_model
        self.map_model=map_model

    def forward(self, x):
        F_x=self.gen_model(x)
        yhat_x=self.map_model(F_x)
        return yhat_x


