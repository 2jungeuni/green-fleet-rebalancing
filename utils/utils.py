import os
import numpy as np
import scipy.sparse as sp
import xml.etree.ElementTree as ET
from scipy.sparse.linalg import eigs

import torch

from config import cfg

def load_adj():
    path = "./data"
    adj = sp.load_npz(os.path.join(path, "adj.npz"))
    adj = adj.tocsc()

    return adj.toarray()

def z_score(x, mean, std):
    return (x - mean) / std

def z_inverse(y, mean, std):
    return y * std + mean

def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return torch.mean(torch.abs(v_ - v))

def scaled_laplacian(W):
    # d -> diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)

    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])

    # lambda_max \approx 2.0, the largest eigenvalues of l.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))

def cheb_poly_approx(L, Ks, n):
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))

        return np.concatenate(L_list, axis=1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')

def compute_loss_and_pred(model,
                          x,
                          n_hist,
                          loss_fn,
                          device=None):
    if device is not None:
        x = x.to(device)

    y_hat = model(x[:, :n_hist, :, :])

    copy_l = loss_fn(
        x[:, n_hist - 1: n_hist, :, :],
        x[:, n_hist: n_hist + 1, :, :]
    )

    train_l = loss_fn(
        y_hat,
        x[:, n_hist: n_hist + 1, :, :]
    )

    single_pred = y_hat[:, 0, :, 0]

    return train_l, copy_l, single_pred

def map_normalization(x, y):
    x_ = (x - cfg.x_min) / (cfg.x_max - cfg.x_min)
    y_ = (y - cfg.y_min) / (cfg.y_max - cfg.y_min)
    return x_, y_

def update_route_file(config_path: str,
                      n_cav: int,
                      output_path: str = None):
    # 1) Load a config path
    tree = ET.parse(config_path)
    root = tree.getroot()
    new_route = f"osm.rou_{n_cav}.xml"

    # 2) find <route-files> under <input>
    for elem in root.findall(".//route-files"):
        elem.set("value", new_route)

    # 3) Save chagnes
    save_path = output_path or config_path
    tree.write(save_path, encoding="utf-8", xml_declaration=True)
    print(f"Updated <route-files> to '{new_route}' in {save_path}")
