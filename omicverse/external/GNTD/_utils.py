import torch
import numpy as np

# Define performance metrics
def MSE(expr, expr_hat):
    return np.mean(np.square(expr_hat - expr))

def MAE(expr, expr_hat):
    return np.mean(np.abs(expr_hat - expr))

def RMSE(expr, expr_hat):
    return np.sqrt(np.mean(np.square(expr_hat - expr)))

def MAPE(expr, expr_hat, threshold=0.1):
    v = np.clip(np.abs(expr), threshold, None)
    diff = np.abs((expr_hat - expr) / v)
    return 100.0 * np.mean(diff)

def R2(expr, expr_hat):
    rss = np.sum(np.square(expr_hat - expr))
    tss = np.sum(np.square(expr - np.mean(expr)))
    return 1 - rss/tss

# Generate graph Laplacian for graph
def generate_graph_Laplacian(A, normalized = True):
    
    if np.allclose(A, np.eye(A.shape[0])):
        
        L = A
        
    else:
        if normalized:

            d = np.sum(A, axis=0)
            nz_index = np.where(d != 0)
            d[nz_index] = d[nz_index] ** (-0.5)
            d = np.expand_dims(d, axis=1)
            A = d.T * A * d

        L = np.diag(np.sum(A, axis=0)) - A
    
    L = torch.from_numpy(L.astype(np.float32))
    
    return L