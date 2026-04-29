import random
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph
import torch

from models import IGAE, StudentAE

try:
    from deeprobust.graph.data import Dataset
except ImportError:
    class Dataset:
        pass


EPS = 1e-12


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_idx: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_idx}")
    return torch.device("cpu")


def soft_assign(z: torch.Tensor, centers: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    dist = torch.cdist(z, centers, p=2).pow(2)
    q = 1.0 / (1.0 + dist / alpha)
    q = q.pow((alpha + 1.0) / 2.0)
    q = q / (q.sum(dim=1, keepdim=True) + EPS)
    return q


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    dim = int(max(y_true.max(), y_pred.max()) + 1)
    weight = np.zeros((dim, dim), dtype=np.int64)
    for i in range(y_pred.shape[0]):
        weight[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(weight.max() - weight)
    mapping = np.zeros(dim, dtype=np.int64)
    mapping[row] = col
    mapped_pred = mapping[y_pred]
    acc = (mapped_pred == y_true).mean()
    return float(acc), mapped_pred


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc, mapped_pred = cluster_accuracy(y_true, y_pred)
    return {
        "acc": float(acc),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
    }


class sc2Dpr(Dataset):
    def __init__(self, X_hvg, X_pca, adj, adj_norm, label, multi_splits=False, **kwargs):
        self.adj = adj
        self.adj_norm = adj_norm
        self.X_hvg = X_hvg
        self.X_pca = X_pca
        self.labels = label


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()
    normalized_D = degree_power(A, -0.5)
    return np.asarray(normalized_D.dot(A).dot(normalized_D), dtype=np.float32)


def get_adj(count, k=10, mode="connectivity"):
    n_samples = int(count.shape[0])
    if n_samples < 1:
        raise ValueError("count must contain at least one sample to build a graph.")

    if n_samples == 1:
        adj = np.ones((1, 1), dtype=np.float32)
        return adj, norm_adj(adj)

    requested_k = int(k)
    resolved_k = min(max(1, requested_k), n_samples - 1)
    if resolved_k != requested_k:
        print(f"Adjusting knn from {requested_k} to {resolved_k} for {n_samples} cells.")

    A = kneighbors_graph(count, resolved_k, mode=mode, metric="cosine", include_self=True)
    print("A", A.shape)
    adj = A.toarray().astype(np.float32, copy=False)
    adj_n = norm_adj(adj)
    return adj, adj_n


def build_teacher_encoder(args, input_dim: int):
    teacher_name = args.teacher_model.upper()
    if teacher_name != "IGAE":
        raise ValueError(f"Only IGAE teacher is supported, but got: {args.teacher_model}")
    return IGAE(
        gae_n_enc_1=args.gae_n_enc_1,
        gae_n_enc_2=args.gae_n_enc_2,
        gae_n_dec_1=args.gae_n_dec_1,
        gae_n_dec_2=args.gae_n_dec_2,
        n_input=input_dim,
        n_z=args.embed_dim,
        dropout=args.dropout,
    )


def parse_student_hidden_dims(args) -> List[int]:
    dims_cfg = getattr(args, "student_hidden_dims", "")
    if isinstance(dims_cfg, (list, tuple)):
        hidden_dims = [int(dim) for dim in dims_cfg]
        if hidden_dims:
            return hidden_dims

    if isinstance(dims_cfg, (int, float)):
        return [int(dims_cfg)]

    if isinstance(dims_cfg, str) and dims_cfg.strip():
        hidden_dims = [int(dim.strip()) for dim in dims_cfg.split(",") if dim.strip()]
        if hidden_dims:
            return hidden_dims

    if args.student_layers <= 1:
        return []
    return [int(args.student_hidden_dim)] * (int(args.student_layers) - 1)


def build_student_encoder(args, input_dim: int):
    hidden_dims = parse_student_hidden_dims(args)
    return StudentAE(
        input_dim=input_dim,
        hidden_dim=hidden_dims,
        embed_dim=args.embed_dim,
        num_layers=len(hidden_dims) + 1,
        dropout=args.student_dropout,
        norm_type=args.norm_type,
    )


def ensemble_students(students: List[Dict], device: torch.device):
    raw_alpha = torch.tensor([s["alpha"] for s in students], device=device, dtype=torch.float32)
    alpha = raw_alpha / (raw_alpha.sum() + EPS)

    pred_list = []
    z_list = []
    for student in students:
        pred_list.append(student["q"])
        z_list.append(student["z"])

    pred_all = torch.stack(pred_list, dim=0)
    z_all = torch.stack(z_list, dim=0)
    alpha_view = alpha.unsqueeze(1).unsqueeze(1)

    logits = torch.sum(pred_all * alpha_view, dim=0)
    logits = logits / (logits.sum(dim=1, keepdim=True) + EPS)
    z_ens = torch.sum(z_all * alpha_view, dim=0)

    pred = logits.argmax(dim=1).detach().cpu().numpy()
    return logits.detach(), z_ens.detach(), pred, alpha.detach().cpu().numpy()
