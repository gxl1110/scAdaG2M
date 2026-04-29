import numpy as np
import scanpy as sc
from dataload import info_log
import pandas as pd
import h5py
import scipy as sp

def sc_handler(adata, args, copy=True, size_factors=True, normalize_input=True, logtrans_input=True):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error


    info_log.print('--------> Preprocessing sc data ...')

    num_gene_select = args.preprocess_top_gene_select

    # Step 1: Truncate low-quality cells and genes
    info_log.print('----------------> Truncating genes and cells ...')

    X_raw_n_obs = adata.n_obs
    X_raw_n_vars = adata.n_vars

    adata.var_names_make_unique() # 处理个别数据集基因不唯一

    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)

    # adata = filter2(adata) # 多执行的

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        adata.X = adata.X.astype(np.float32) # 个别数据集需要处理
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    # Step 2: Select highly variable genes
    if num_gene_select != -1 and num_gene_select < adata.n_vars:
        info_log.print('----------------> Selecting highly variable genes with Scanpy ...')
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=num_gene_select,
            flavor='seurat',
            subset=True
        )

        info_log.print(f"--------> Selected {num_gene_select} highly variable genes.")

    info_log.print('----------------> Log-transforming and normalizing data ...')
    # sc.pp.normalize_total(adata) # Adam / Quake_Smart-seq2_Diaphragm 数据集需要执行该代码
    # sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    info_log.print(f"--------> Preprocessed sc data has {adata.n_obs} cells and {adata.n_vars} genes, "
                   f"Removing {X_raw_n_obs - adata.n_obs} cells and {X_raw_n_vars - adata.n_vars} genes")

    return adata
