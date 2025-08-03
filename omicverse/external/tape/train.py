import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import simdatset, AutoEncoder, device
from .utils import showloss

def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def training_stage(model, train_loader, optimizer, epochs=128):
    
    model.train()
    model.state = 'train'
    loss = []
    recon_loss = []
    
    for i in tqdm(range(epochs)):
        for k, (data, label) in enumerate(train_loader):
            # reproducibility(seed=0)
            optimizer.zero_grad()
            x_recon, cell_prop, sigm = model(data)
            batch_loss = F.l1_loss(cell_prop, label) + F.l1_loss(x_recon, data) 
            batch_loss.backward()
            optimizer.step()
            loss.append(F.l1_loss(cell_prop, label).cpu().detach().numpy())
            recon_loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

    return model, loss, recon_loss

def adaptive_stage(model, data, optimizerD, optimizerE, step=10, max_iter=5):
    data = torch.from_numpy(data).float().to(device)
    loss = []
    model.eval()
    model.state = 'test'
    _, ori_pred, ori_sigm = model(data)
    ori_sigm = ori_sigm.detach()
    ori_pred = ori_pred.detach()
    model.state = 'train'
    
    for k in range(max_iter):
        model.train()
        for i in range(step):
            reproducibility(seed=0)
            optimizerD.zero_grad()
            x_recon, _, sigm = model(data)
            batch_loss = F.l1_loss(x_recon, data)+F.l1_loss(sigm,ori_sigm)
            batch_loss.backward()
            optimizerD.step()
            loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

        for i in range(step):
            reproducibility(seed=0)
            optimizerE.zero_grad()
            x_recon, pred, _ = model(data)
            batch_loss = F.l1_loss(ori_pred, pred)+F.l1_loss(x_recon, data)
            batch_loss.backward()
            optimizerE.step()
            loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

    model.eval()
    model.state = 'test'
    _, pred, sigm = model(data)
    return sigm.cpu().detach().numpy(), loss, pred.detach().cpu().numpy()

def train_model(train_x, train_y,
                model_name=None,
                batch_size=128, epochs=128):
    
    train_loader = DataLoader(simdatset(train_x, train_y), batch_size=batch_size, shuffle=True)
    model = AutoEncoder(train_x.shape[1], train_y.shape[1]).to(device)
    # reproducibility(seed=0)
    optimizer = Adam(model.parameters(), lr=1e-4)
    print('Start training')
    model, loss, reconloss = training_stage(model, train_loader, optimizer, epochs=epochs)
    print('Training is done')
    print('prediction loss is:')
    showloss(loss)
    print('reconstruction loss is:')
    showloss(reconloss)
    if model_name is not None:
        print('Model is saved')
        torch.save(model, model_name+".pth")
    return model

def predict(test_x, genename, celltypes, samplename,
            model_name=None, model=None,
            adaptive=True, mode='overall'):
    
    if model is not None and model_name is None:
        print('Model is saved without defined name')
        torch.save(model, 'model.pth')
    if adaptive is True:
        if mode == 'high-resolution':
            TestSigmList = np.zeros((test_x.shape[0], len(celltypes), len(genename)))
            TestPred = np.zeros((test_x.shape[0], len(celltypes)))
            print('Start adaptive training at high-resolution')
            for i in tqdm(range(len(test_x))):
                x = test_x[i,:].reshape(1,-1)
                if model_name is not None and model is None:
                    model = torch.load(model_name + ".pth")
                elif model is not None and model_name is None:
                    model = torch.load("model.pth")
                decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
                encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
                optimizerD = torch.optim.Adam(decoder_parameters, lr=1e-4)
                optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
                test_sigm, loss, test_pred = adaptive_stage(model, x, optimizerD, optimizerE, step=300, max_iter=3)
                TestSigmList[i, :, :] = test_sigm
                TestPred[i,:] = test_pred
            TestPred = pd.DataFrame(TestPred,columns=celltypes,index=samplename)
            CellTypeSigm = {}
            for i in range(len(celltypes)):
                cellname = celltypes[i]
                sigm = TestSigmList[:,i,:]
                sigm = pd.DataFrame(sigm,columns=genename,index=samplename)
                CellTypeSigm[cellname] = sigm
            print('Adaptive stage is done')

            return CellTypeSigm, TestPred

        elif mode == 'overall':
            if model_name is not None and model is None:
                model = torch.load(model_name + ".pth")
            elif model is not None and model_name is None:
                model = torch.load("model.pth")
            decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
            encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
            optimizerD = torch.optim.Adam(decoder_parameters, lr=1e-4)
            optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
            print('Start adaptive training for all the samples')
            test_sigm, loss, test_pred = adaptive_stage(model, test_x, optimizerD, optimizerE, step=300, max_iter=3)
            print('Adaptive stage is done')
            test_sigm = pd.DataFrame(test_sigm,columns=genename,index=celltypes)
            test_pred = pd.DataFrame(test_pred,columns=celltypes,index=samplename)

            return test_sigm, test_pred

    else:
        if model_name is not None and model is None:
            model = torch.load(model_name+".pth")
        elif model is not None and model_name is None:
            model = model
        print('Predict cell fractions without adaptive training')
        model.eval()
        model.state = 'test'
        data = torch.from_numpy(test_x).float().to(device)
        _, pred, _ = model(data)
        pred = pred.cpu().detach().numpy()
        pred = pd.DataFrame(pred, columns=celltypes, index=samplename)
        print('Prediction is done')
        return pred



