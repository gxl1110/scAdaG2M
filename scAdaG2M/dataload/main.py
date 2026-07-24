from dataload import info_log
info_log.print('\n> Loading Packages')
from time import time
# Local modules
from dataload import dataLoad
from dataload import preprocess
import scipy.sparse as sp
import os
from utils import *
from sklearn.decomposition import PCA
from metric import cluster_accuracy
from sklearn.decomposition import TruncatedSVD
import sys
import pandas as pd
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataload"))

def getscData(args):

    # Set up the program
    param = dict()
    param['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param['dataloader_kwargs'] = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    param['tik'] = time()
    torch.manual_seed(args.seed)
    info_log.print(f"Using device: {param['device']}")
    info_log.print(args)
    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Load and preprocess data
    info_log.print('\n> Loading data ...')
    X_sc_raw = dataLoad.sc_handler(args)

    info_log.print('\n> Preprocessing data ...')
    X_sc = preprocess.sc_handler(X_sc_raw, args)

    info_log.print('\n> Setting up data for Training ...')

    X = X_sc.X
    Y = np.array(X_sc.obs["Group"]) if "Group" in X_sc.obs else None

    if Y is not None and getattr(args, "n_clusters", None) is None:
        args.n_clusters = int(max(Y) - min(Y) + 1)

    X = X.astype(np.float32)


    pca = PCA(n_components=args.n_input)
    feat = pca.fit_transform(X)


    adj, adj_n = get_adj(feat, k=args.knn)

    data = sc2Dpr(X_hvg=X ,X_pca=feat, adj = adj, adj_norm=adj_n, label=Y)

    return data
