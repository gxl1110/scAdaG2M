import numpy as np
import pandas as pd
# import scipy.sparse as sp
import os
from dataload import info_log
import h5py
import scanpy as sc
import scipy as sp


def sc_handler(args):
    # Get the directory path of the dataset
    dir_path = os.path.join(args.load_dataset_dir, args.load_dataset_name)

    # Check if the user wants to use the benchmark dataset
    if args.load_h5:
        info_log.print('--------> Loading from h5 file ...')
        return load_h5(
            os.path.join(dir_path, args.load_h5)
            # is_cell_by_gene=True
        )

    elif args.load_h5_2:
        info_log.print('--------> Loading sc raw expression ...')
        return load_h5_v2(
            os.path.join(dir_path, args.load_h5_2)
        )
    # If none of the above, load the raw sc expression data
    else:
        info_log.print('--------> Loading sc raw expression ...')
        return load_dense(
            os.path.join(dir_path, args.load_sc_dataset), os.path.join(dir_path, args.load_cell_type_labels),
            is_cell_by_gene=True
        )



def load_h5(filename, sparsify=False, skip_exprs=False):
    X, Y = prepro(filename, sparsify=sparsify, skip_exprs=skip_exprs)

    adata = sc.AnnData(X)
    adata.obs['Group'] = Y

    return adata

def load_h5_v2(filename):
    info_log.print('--------> Loading h5 X Y ...')
    data_mat = h5py.File(filename, "r")
    X = np.array(data_mat['X'])
    Y = np.array(data_mat['Y'])

    cell_label = np.unique(Y, return_inverse=True)[1]

    adata = sc.AnnData(X)
    adata.obs["Group"] = cell_label

    return adata


def load_dense(file_path, labelfile,  is_cell_by_gene=False, has_gene_name=True, has_cell_name=True, dtype=float, kwargs=None):
    """
    Load dense expression matrix and convert to sc.AnnData format.

    Args:
        file_path (str): Path to the expression matrix file.
        is_cell_by_gene (bool): Whether the matrix is cell x gene format.
        has_gene_name (bool): Whether the matrix contains gene names.
        has_cell_name (bool): Whether the matrix contains cell names.
        dtype (type): Data type to convert expression matrix to.
        kwargs (dict): Optional arguments for pd.read_csv.

    Returns:
        sc.AnnData: AnnData object containing expression matrix and metadata.
    """
    info_log.print('----------------> Reading matrix (dense) ...')

    kwargs = {'index_col': 0, 'sep': None} if kwargs is None else kwargs

    # 读取原始表达矩阵
    X = pd.read_csv(file_path, engine="python", **kwargs)
    expr = X.to_numpy().astype(dtype)

    # 获取细胞名和基因名
    rows = np.arange(X.shape[0])
    columns = np.arange(X.shape[1])
    cell = rows if is_cell_by_gene else columns
    gene = columns if is_cell_by_gene else rows

    if has_cell_name:
        cell = X.index.to_numpy() if is_cell_by_gene else X.columns.to_numpy()
    if has_gene_name:
        gene = X.columns.to_numpy() if is_cell_by_gene else X.index.to_numpy()

    # 转换为 cell × gene
    expr_matrix = expr if is_cell_by_gene else expr.T

    # 构建 AnnData 对象
    adata = sc.AnnData(
        X=expr_matrix,
        obs=pd.DataFrame(index=cell),
        var=pd.DataFrame(index=gene)
    )

    info_log.print(f"----------------> Matrix has {adata.n_obs} cells and {adata.n_vars} genes")

    y = pd.read_csv(labelfile, index_col=0, sep=None)
    y.index = y.index.astype(str)

    # 确保标签顺序与 AnnData.obs 顺序一致（必须有完整匹配）
    adata.obs_names = adata.obs_names.astype(str)

    # 检查 y.index 和 adata.obs_names 是否一致
    if not y.index.equals(adata.obs_names):
        print("警告：y.index 和 adata.obs_names 的顺序不一致！")
        # 可以打印出不一致的部分
        diff = set(y.index) - set(adata.obs_names)
        print(f"y.index 中存在但 adata.obs_names 中不存在的索引：{diff}")
        diff = set(adata.obs_names) - set(y.index)
        print(f"adata.obs_names 中存在但 y.index 中不存在的索引：{diff}")
        label_df = y.loc[adata.obs_names]
        ct_labels = label_df.iloc[:, 0].astype(str).to_numpy()
    else:
        print("y.index 和 adata.obs_names 的顺序一致！")
        ct_labels = y.iloc[:, 0].astype(str).to_numpy()
    # label_df = y.loc[adata.obs_names]

    # ct_labels = label_df.iloc[:, 0].astype(str).to_numpy()
    cell_label = np.unique(ct_labels, return_inverse=True)[1]
    adata.obs["Group"] = cell_label

    # n_clusters = int(max(cell_label) - min(cell_label) + 1)


    info_log.print(f"----------------> {len(cell_label)} ground-truth cell type labels loaded")

    return adata


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)



def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]))
        var = pd.DataFrame(dict_from_group(f["var"]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


def prepro(filename,sparsify=False, skip_exprs=False):
    data_path = filename
    mat, obs, var, uns = read_data(data_path, sparsify=sparsify, skip_exprs=skip_exprs)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label