import random
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    weight = q.pow(2) / (q.sum(dim=0, keepdim=True) + EPS)
    p = weight / (weight.sum(dim=1, keepdim=True) + EPS)
    return p


def kl_per_cell(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * (torch.log(p + EPS) - torch.log(q + EPS))).sum(dim=1)


def mean_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return kl_per_cell(p, q).mean()


def entropy_per_cell(q: torch.Tensor) -> torch.Tensor:
    return -(q * torch.log(q + EPS)).sum(dim=1)


def feature_dropout(x: torch.Tensor, rate: float) -> torch.Tensor:
    if rate <= 0:
        return x
    keep = (torch.rand_like(x) > rate).float()
    return x * keep


def feature_jitter(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


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
        "f1_macro": float(f1_score(y_true, mapped_pred, average="macro")),
    }


def init_centers_from_kmeans(z: torch.Tensor, n_clusters: int, seed: int) -> torch.Tensor:
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    kmeans.fit(z.detach().cpu().numpy())
    centers = torch.from_numpy(kmeans.cluster_centers_).to(z.device, dtype=z.dtype)
    return centers


def sample_feature_subspace(num_features: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    subset_size = max(2, int(round(num_features * ratio)))
    subset_size = min(num_features, subset_size)
    indices = rng.choice(num_features, size=subset_size, replace=False)
    indices.sort()
    return indices


class sc2Dpr(Dataset):
    def __init__(self, X_hvg, X_pca,  adj, adj_norm, label,  multi_splits=False, **kwargs):

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
        A = A.cpu().numpy()  # 先转换为 NumPy 数组
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def get_adj(count, k=10, mode="connectivity"):
    countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="cosine", include_self=True)
    print("A", A.shape)
    adj = A.toarray()

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



def pretrain_gae(model, x, adj, label, cfg):
    print("Pretraining GAE...")
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_pretrain)
    z_last = None
    for epoch in range(1, cfg.rec_epoch + 1):
        model.train()
        z, a, _, _ = model.encoder(x, adj)
        z_hat, z_adj_hat, _, _ = model.decoder(z, adj)
        a_hat = a + z_adj_hat
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, x))
        loss_a = F.mse_loss(a_hat, adj.to_dense())
        loss = loss_w + cfg.alpha_value * loss_a

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        z_last = z

        if epoch % cfg.log_every == 0 or epoch == cfg.rec_epoch:
            print(
                f"[Teacher GAE Pretrain] Epoch {epoch:04d} | "
                f"Total {loss.item():.6f}"
            )

    if label is not None:
        x_np = z_last.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=cfg.n_clusters, n_init=20, random_state=42).fit(x_np)
        pred = kmeans.predict(x_np)
        acc, mapped_pred = cluster_accuracy(label, pred)
        nmi = float(normalized_mutual_info_score(label, pred))
        ari = float(adjusted_rand_score(label, pred))
        f1 = float(f1_score(label, mapped_pred, average="macro"))
        print(f"GAE pretrain metrics: acc={acc:.4f}, nmi={nmi:.4f}, f1={f1:.4f}, ari={ari:.4f}")
    else:
        x_np = z_last.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=cfg.n_clusters, n_init=20, random_state=42).fit(x_np)
        pred = kmeans.predict(x_np)

    return model, z_last.detach(), pred, kmeans.cluster_centers_.astype(np.float32)


def teacher_kmeans_distribution(z, cfg, label=None, prefix="Teacher"):
    z_np = z.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=cfg.n_clusters, n_init=20, random_state=42).fit(z_np)
    pred = kmeans.predict(z_np)
    centers = torch.from_numpy(kmeans.cluster_centers_).to(z.device, dtype=z.dtype)
    q = soft_assign(z, centers, cfg.dec_alpha)
    p = target_distribution(q)

    if label is not None:
        acc, mapped_pred = cluster_accuracy(label, pred)
        nmi = float(normalized_mutual_info_score(label, pred))
        ari = float(adjusted_rand_score(label, pred))
        f1 = float(f1_score(label, mapped_pred, average="macro"))
        print(f"{prefix} pretrain metrics: acc={acc:.4f}, nmi={nmi:.4f}, f1={f1:.4f}, ari={ari:.4f}")

    return centers, q, p


def train_teacher_dec(cfg, encoder, adj, x, label=None):
    if cfg.teacher_model.upper() != "IGAE":
        raise ValueError(f"Only IGAE teacher is supported, but got: {cfg.teacher_model}")

    if not isinstance(adj, torch.Tensor):
        raise TypeError("adj must be a torch.Tensor for IGAE teacher training.")
    if not adj.is_sparse:
        adj = adj.to_sparse().coalesce()
    else:
        adj = adj.coalesce()
    adj = adj.to(x.device)

    encoder, z, pred, centers_np = pretrain_gae(encoder, x, adj, label, cfg)

    centers = torch.from_numpy(centers_np).to(z.device, dtype=z.dtype)
    q = soft_assign(z, centers, cfg.dec_alpha)
    p = target_distribution(q)

    for param in encoder.parameters():
        param.requires_grad = False

    return {
        "encoder": encoder,
        "centers": centers.detach(),
        "z": z.detach(),
        "q": q.detach(),
        "p": p.detach(),
    }

def adag2m_update_weights(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    node_weights: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, float]:
    criterion = nn.KLDivLoss(reduction="none", log_target=True)
    with torch.no_grad():
        out_s = logits_s.log_softmax(dim=1)
        out_t = logits_t.log_softmax(dim=-1)
        loss = criterion(out_s, out_t).sum(dim=1)
        errors = 1.0 - torch.exp(-beta * loss)
        error = torch.sum(node_weights * errors) / (torch.sum(node_weights) + EPS)
        error = torch.clamp(error, min=1e-16, max=1.0 - 1e-16)
        alpha = torch.clamp(torch.log((1.0 - error) / error + 1e-16), min=1e-16)
        new_weights = node_weights * torch.exp(alpha * errors)
        new_weights = new_weights / (new_weights.sum() + EPS)
    return new_weights, float(alpha.item())


def train_single_student(
    cfg,
    x_sub: torch.Tensor,
    teacher_q: torch.Tensor,
    teacher_p: torch.Tensor,
    weights: torch.Tensor,
    uncertainty_norm: torch.Tensor,
    student_idx: int,
    labels: np.ndarray = None,
):
    model = build_student_encoder(cfg, x_sub.shape[1]).to(x_sub.device)

    with torch.no_grad():
        z_init, _ = model(x_sub)
    centers = nn.Parameter(
        init_centers_from_kmeans(
            z_init,
            n_clusters=cfg.n_clusters,
            seed=cfg.seed + student_idx * 17,
        )
    )

    optimizer = optim.Adam(
        list(model.parameters()) + [centers],
        lr=cfg.student_dec_lr,
        weight_decay=cfg.weight_decay,
    )
    teacher_target = teacher_p if cfg.kd_target == "p" else teacher_q
    teacher_target = teacher_target.detach()
    kd_conf_quantile = float(getattr(cfg, "kd_conf_quantile", 0.5))
    kd_conf_quantile = min(max(kd_conf_quantile, 0.0), 1.0)
    entropy_threshold = torch.quantile(uncertainty_norm.detach(), kd_conf_quantile)
    reliable_mask = (uncertainty_norm < entropy_threshold).float().detach()
    if reliable_mask.sum().item() <= 0:
        reliable_mask = torch.ones_like(reliable_mask)
    print(
        f"[Student {student_idx}] KD confidence gating | "
        f"quantile={kd_conf_quantile:.2f}, threshold={entropy_threshold.item():.6f}, "
        f"reliable_ratio={reliable_mask.mean().item():.4f}"
    )

    for epoch in range(1, cfg.student_dec_epochs + 1):
        model.train()
        x_aug = feature_jitter(feature_dropout(x_sub, cfg.mask_rate), cfg.aug_noise_std)
        z, x_hat = model(x_aug)
        loss_rec = F.mse_loss(x_hat, x_sub)

        q = soft_assign(z, centers, cfg.dec_alpha)
        p = target_distribution(q).detach()
        loss_kl = mean_kl(p, q)


        x_aug2 = feature_jitter(feature_dropout(x_sub, cfg.mask_rate), cfg.aug_noise_std)
        z_aug2, _ = model(x_aug2)
        loss_na = F.mse_loss(z, z_aug2)

        kl_val = kl_per_cell(teacher_target, q)
        loss_kd = torch.sum(weights * reliable_mask * kl_val)
        loss = (
            cfg.lambda_dec * loss_kl
            + cfg.lambda_kd * loss_kd
            + cfg.lambda_na * loss_na
            + cfg.lambda_rec * loss_rec
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % cfg.log_every == 0 or epoch == cfg.student_dec_epochs:
            print(
                f"[Student {student_idx} Train] Epoch {epoch:04d} | "
                f"KL loss {loss_kl.item():.6f} | KD {loss_kd.item():.6f} | "
                f"NA {loss_na.item():.6f} | REC {loss_rec.item():.6f}"
            )

    model.eval()
    with torch.no_grad():
        z, _ = model(x_sub)
        q = soft_assign(z, centers, cfg.dec_alpha)

    beta = float(getattr(cfg, "boost_beta", getattr(cfg, "boost_alpha", 1.0)))
    student_logits = torch.log(torch.clamp(q, min=EPS))
    teacher_logits = torch.log(torch.clamp(teacher_q, min=EPS))
    new_weights, student_alpha = adag2m_update_weights(
        logits_s=student_logits,
        logits_t=teacher_logits,
        node_weights=weights,
        beta=beta,
    )


    return {
        "encoder": model,
        "centers": centers.detach(),
        "q": q.detach(),
        "z": z.detach(),
        "alpha": float(student_alpha),
    }, new_weights


def ensemble_students(cfg, students: List[Dict], seed: int, device: torch.device):
    raw_alpha = torch.tensor([s["alpha"] for s in students], device=device, dtype=torch.float32)
    alpha = raw_alpha / (raw_alpha.sum() + EPS)

    pred_list = []
    z_list = []
    for student in students:
        pred_list.append(student["q"])
        z_list.append(student["z"])

    pred_all = torch.stack(pred_list, dim=0)  # [K, N, C]
    z_all = torch.stack(z_list, dim=0)  # [K, N, D]
    alpha_view = alpha.unsqueeze(1).unsqueeze(1)  # [K, 1, 1]

    logits = torch.sum(pred_all * alpha_view, dim=0)  # [N, C], weighted average prob
    logits = logits / (logits.sum(dim=1, keepdim=True) + EPS)
    out = torch.log(logits + 1e-16)  # [N, C], log probability
    z_ens = torch.sum(z_all * alpha_view, dim=0)  # [N, D]

    pred = out.argmax(dim=1).detach().cpu().numpy()
    q_ens = logits
    return q_ens.detach(), z_ens.detach(), pred, alpha.detach().cpu().numpy()
