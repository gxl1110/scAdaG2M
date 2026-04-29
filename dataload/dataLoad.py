import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp

from dataload import info_log


def resolve_path(base_dir: Path, filename):
    if filename in (None, ""):
        return None

    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def ensure_existing_file(path: Path, context: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"{context} path is required but missing.")
    if not path.exists():
        raise FileNotFoundError(f"{context} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{context} is not a file: {path}")
    return path


def infer_h5_path(dataset_dir: Path, dataset_name: str):
    preferred = dataset_dir / f"{dataset_name}.h5"
    if preferred.exists():
        return preferred

    h5_files = sorted(path for path in dataset_dir.iterdir() if path.is_file() and path.suffix.lower() == ".h5")
    if len(h5_files) == 1:
        return h5_files[0]
    if len(h5_files) > 1:
        available = ", ".join(path.name for path in h5_files)
        raise FileNotFoundError(
            f"Multiple .h5 files found under {dataset_dir}: {available}. "
            "Specify one with --load_h5 or --load_h5_2."
        )
    return None


def resolve_input_route(args):
    dataset_dir = Path(args.load_dataset_dir).expanduser() / args.load_dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if args.load_h5:
        return "load_h5", ensure_existing_file(resolve_path(dataset_dir, args.load_h5), "Legacy h5 dataset")

    if args.load_h5_2:
        return "load_h5_2", ensure_existing_file(resolve_path(dataset_dir, args.load_h5_2), "h5 dataset")

    if args.load_sc_dataset:
        expr_path = ensure_existing_file(resolve_path(dataset_dir, args.load_sc_dataset), "Expression matrix")
        label_path = resolve_path(dataset_dir, args.load_cell_type_labels)
        if label_path is not None:
            label_path = ensure_existing_file(label_path, "Label file")
        return "load_dense", expr_path, label_path

    inferred_h5 = infer_h5_path(dataset_dir, args.load_dataset_name)
    if inferred_h5 is not None:
        info_log.print(f"--------> Auto-detected h5 dataset: {inferred_h5.name}")
        return "load_h5_2", inferred_h5

    raise FileNotFoundError(
        f"Could not infer an input file under {dataset_dir}. "
        "Set --load_h5, --load_h5_2, or --load_sc_dataset explicitly."
    )


def sc_handler(args):
    route = resolve_input_route(args)
    if route[0] == "load_h5":
        info_log.print('--------> Loading from h5 file ...')
        return load_h5(str(route[1]))

    if route[0] == "load_h5_2":
        info_log.print('--------> Loading sc raw expression ...')
        return load_h5_v2(str(route[1]))

    info_log.print('--------> Loading sc raw expression ...')
    return load_dense(
        str(route[1]),
        None if route[2] is None else str(route[2]),
        is_cell_by_gene=True,
    )


def load_h5(filename, sparsify=False, skip_exprs=False):
    X, Y = prepro(filename, sparsify=sparsify, skip_exprs=skip_exprs)
    adata = sc.AnnData(X)
    adata.obs['Group'] = Y
    return adata


def load_h5_v2(filename):
    info_log.print('--------> Loading h5 X Y ...')
    with h5py.File(filename, "r") as data_mat:
        if 'X' not in data_mat:
            raise KeyError(f"h5 dataset is missing required key 'X': {filename}")
        X = np.array(data_mat['X'])
        Y = np.array(data_mat['Y']) if 'Y' in data_mat else None

    adata = sc.AnnData(X)
    if Y is not None:
        adata.obs["Group"] = np.unique(Y, return_inverse=True)[1]
    else:
        info_log.print('--------> No Y key found. Continuing without labels.')

    return adata


def load_dense(file_path, labelfile, is_cell_by_gene=False, has_gene_name=True, has_cell_name=True, dtype=float, kwargs=None):
    info_log.print('----------------> Reading matrix (dense) ...')
    if not file_path:
        raise ValueError("A dense expression matrix path must be provided for load_dense.")
    kwargs = {'index_col': 0, 'sep': None} if kwargs is None else kwargs

    X = pd.read_csv(file_path, engine="python", **kwargs)
    expr = X.to_numpy().astype(dtype)

    rows = np.arange(X.shape[0])
    columns = np.arange(X.shape[1])
    cell = rows if is_cell_by_gene else columns
    gene = columns if is_cell_by_gene else rows

    if has_cell_name:
        cell = X.index.to_numpy() if is_cell_by_gene else X.columns.to_numpy()
    if has_gene_name:
        gene = X.columns.to_numpy() if is_cell_by_gene else X.index.to_numpy()

    expr_matrix = expr if is_cell_by_gene else expr.T
    adata = sc.AnnData(
        X=expr_matrix,
        obs=pd.DataFrame(index=cell),
        var=pd.DataFrame(index=gene),
    )

    info_log.print(f"----------------> Matrix has {adata.n_obs} cells and {adata.n_vars} genes")

    if not labelfile:
        info_log.print("----------------> No label file provided. Continuing without labels.")
        return adata

    y = pd.read_csv(labelfile, index_col=0, sep=None, engine="python")
    if y.empty:
        raise ValueError(f"Label file is empty: {labelfile}")

    y.index = y.index.astype(str)
    adata.obs_names = adata.obs_names.astype(str)

    label_df = y.reindex(adata.obs_names)
    missing_mask = label_df.iloc[:, 0].isna()
    if missing_mask.any():
        missing_examples = ", ".join(label_df.index[missing_mask].tolist()[:5])
        raise ValueError(
            f"Label file does not cover all cells in the expression matrix. "
            f"Missing {int(missing_mask.sum())} cells. Examples: {missing_examples}"
        )

    ct_labels = label_df.iloc[:, 0].astype(str).to_numpy()

    adata.obs["Group"] = np.unique(ct_labels, return_inverse=True)[1]
    info_log.print(f"----------------> {len(ct_labels)} ground-truth cell type labels loaded")
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


def read_data(filename, sparsify=False, skip_exprs=False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]))
        var = pd.DataFrame(dict_from_group(f["var"]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix(
                    (exprs_handle["data"][...], exprs_handle["indices"][...], exprs_handle["indptr"][...]),
                    shape=exprs_handle["shape"][...],
                )
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


def prepro(filename, sparsify=False, skip_exprs=False):
    mat, obs, var, uns = read_data(filename, sparsify=sparsify, skip_exprs=skip_exprs)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label
