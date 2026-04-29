from time import time
import os

import numpy as np
import torch
from sklearn.decomposition import PCA

from dataload import dataLoad, info_log, preprocess
from utils import get_adj, sc2Dpr


info_log.print('\n> Loading Packages')


def encode_group_labels(labels):
    if labels is None:
        return None

    y = np.asarray(labels)
    if y.ndim > 1:
        y = np.squeeze(y)
    if y.size == 0:
        return None

    if np.issubdtype(y.dtype, np.integer):
        return np.unique(y, return_inverse=True)[1].astype(np.int64)
    return np.unique(y.astype(str), return_inverse=True)[1].astype(np.int64)


def resolve_pca_components(args, X):
    requested = int(args.n_input)
    max_components = int(min(X.shape[0], X.shape[1]))
    if max_components < 1:
        raise ValueError("Input expression matrix must contain at least one cell and one gene.")

    resolved = min(max(1, requested), max_components)
    if resolved != requested:
        info_log.print(
            f"Adjusting PCA components from {requested} to {resolved} for matrix shape {tuple(X.shape)}."
        )
    return resolved


def getscData(args):
    param = dict()
    param['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param['dataloader_kwargs'] = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    param['tik'] = time()
    torch.manual_seed(args.seed)
    info_log.print(f"Using device: {param['device']}")
    info_log.print(args)

    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    info_log.print('\n> Loading data ...')
    X_sc_raw = dataLoad.sc_handler(args)

    info_log.print('\n> Preprocessing data ...')
    X_sc = preprocess.sc_handler(X_sc_raw, args)

    info_log.print('\n> Setting up data for Inference ...')

    X = X_sc.X.astype(np.float32)
    Y = encode_group_labels(np.array(X_sc.obs["Group"])) if "Group" in X_sc.obs else None

    if Y is not None and getattr(args, "n_clusters", None) is None:
        args.n_clusters = int(np.unique(Y).shape[0])

    args.n_input = resolve_pca_components(args, X)
    pca = PCA(n_components=args.n_input)
    feat = pca.fit_transform(X)
    adj, adj_n = get_adj(feat, k=args.knn)

    return sc2Dpr(X_hvg=X, X_pca=feat, adj=adj, adj_norm=adj_n, label=Y)
