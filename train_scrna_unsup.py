import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch

from config import parse_config
from dataload.main import getscData
from utils import (
    build_student_encoder,
    build_teacher_encoder,
    clustering_metrics,
    ensemble_students,
    get_device,
    set_seed,
    soft_assign,
)


DEFAULT_CHECKPOINT_PATH = Path("outputs") / "Human1" / "checkpoint.pt"


INFERENCE_CFG_KEYS = [
    "n_input",
    "knn",
    "n_clusters",
    "embed_dim",
    "dec_alpha",
    "teacher_model",
    "gae_n_enc_1",
    "gae_n_enc_2",
    "gae_n_dec_1",
    "gae_n_dec_2",
    "dropout",
    "n_students",
    "student_layers",
    "student_hidden_dim",
    "student_hidden_dims",
    "student_dropout",
    "norm_type",
    "preprocess_top_gene_select",
]


def to_numpy_float32(array_like) -> np.ndarray:
    if sp.issparse(array_like):
        return np.asarray(array_like.toarray(), dtype=np.float32)
    return np.asarray(array_like, dtype=np.float32)


def to_int_labels(labels):
    if labels is None:
        return None
    y = np.asarray(labels)
    if y.ndim > 1:
        y = np.squeeze(y)
    if y.size == 0:
        return None
    if np.issubdtype(y.dtype, np.integer):
        return y.astype(np.int64)
    _, encoded = np.unique(y.astype(str), return_inverse=True)
    return encoded.astype(np.int64)


def load_data_from_main(cfg):
    data = getscData(cfg)
    x_hvg = torch.from_numpy(to_numpy_float32(data.X_hvg))
    x_teacher = torch.from_numpy(to_numpy_float32(data.X_pca))
    labels = to_int_labels(getattr(data, "labels", None))
    adj_raw = data.adj_norm if getattr(data, "adj_norm", None) is not None else data.adj
    adj_norm = torch.tensor(adj_raw, dtype=torch.float32)
    return x_hvg, x_teacher, adj_norm, labels


def to_csv_scalar(value):
    if value is None:
        return ""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict, set)):
        return str(value)
    return value


def cfg_to_summary_dict(cfg) -> Dict[str, object]:
    return {key: to_csv_scalar(val) for key, val in vars(cfg).items()}


def to_plain_dict(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return {key: val for key, val in vars(value).items() if not key.startswith("_")}
    return {}


def get_checkpoint_metadata(checkpoint: Dict) -> Dict:
    for key in ("meta", "metadata"):
        metadata = to_plain_dict(checkpoint.get(key))
        if metadata:
            return metadata
    return {}


def get_checkpoint_config(checkpoint: Dict) -> Dict:
    metadata = get_checkpoint_metadata(checkpoint)
    for candidate in (
        metadata.get("cfg"),
        metadata.get("config"),
        metadata.get("args"),
        checkpoint.get("config"),
        checkpoint.get("cfg"),
    ):
        config_dict = to_plain_dict(candidate)
        if config_dict:
            return config_dict
    return {}


def resolve_output_dir(cfg) -> Path:
    output_root = Path(getattr(cfg, "output_dir", "outputs") or "outputs")
    output_dir = output_root if output_root.name == cfg.load_dataset_name else output_root / cfg.load_dataset_name
    cfg.output_dir = str(output_dir)
    return output_dir


def normalize_checkpoint_candidate(path_like) -> Path:
    path = Path(path_like).expanduser()
    if path.is_dir():
        return path / "checkpoint.pt"
    return path


def resolve_checkpoint_path(cfg) -> Path:
    explicit_path = getattr(cfg, "checkpoint_path", None)
    if explicit_path not in (None, ""):
        checkpoint_path = normalize_checkpoint_candidate(explicit_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        cfg.checkpoint_path = str(checkpoint_path)
        return checkpoint_path

    output_root = Path(getattr(cfg, "output_dir", "outputs") or "outputs")
    dataset_checkpoint = output_root / cfg.load_dataset_name / "checkpoint.pt"
    for candidate in (dataset_checkpoint, DEFAULT_CHECKPOINT_PATH):
        checkpoint_path = normalize_checkpoint_candidate(candidate)
        if checkpoint_path.exists():
            cfg.checkpoint_path = str(checkpoint_path)
            return checkpoint_path

    raise FileNotFoundError(
        "Checkpoint path was not provided and no default checkpoint was found. "
        f"Tried: {dataset_checkpoint} and {DEFAULT_CHECKPOINT_PATH}"
    )


def ensure_sparse_adj(adj: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not isinstance(adj, torch.Tensor):
        raise TypeError("adj must be a torch.Tensor.")
    if not adj.is_sparse:
        adj = adj.to_sparse().coalesce()
    else:
        adj = adj.coalesce()
    return adj.to(device)


def restore_cfg_from_checkpoint(cfg, checkpoint: Dict) -> None:
    saved_cfg = get_checkpoint_config(checkpoint)
    for key in INFERENCE_CFG_KEYS:
        if key in saved_cfg:
            setattr(cfg, key, saved_cfg[key])


def load_checkpoint(cfg, device: torch.device) -> Dict:
    checkpoint_path = resolve_checkpoint_path(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint loaded from: {checkpoint_path}")
    return checkpoint


def resolve_n_clusters(cfg, labels, checkpoint: Dict) -> int:
    metadata = get_checkpoint_metadata(checkpoint)
    if metadata.get("n_clusters") is not None:
        cfg.n_clusters = int(metadata["n_clusters"])
        return cfg.n_clusters

    saved_cfg = get_checkpoint_config(checkpoint)
    if saved_cfg.get("n_clusters") is not None:
        cfg.n_clusters = int(saved_cfg["n_clusters"])
        return cfg.n_clusters

    if labels is not None:
        cfg.n_clusters = int(np.unique(labels).shape[0])
        return cfg.n_clusters

    if getattr(cfg, "n_clusters", None) is not None:
        cfg.n_clusters = int(cfg.n_clusters)
        return cfg.n_clusters

    raise ValueError("n_clusters is missing. Store it in the checkpoint metadata or pass --n_clusters.")


def require_checkpoint_field(container: Dict, key: str, context: str):
    if key not in container:
        raise KeyError(f"Checkpoint is missing `{context}.{key}`.")
    return container[key]


def load_teacher_outputs(cfg, checkpoint: Dict, x_teacher: torch.Tensor, adj_norm: torch.Tensor, device: torch.device) -> Dict:
    teacher_block = require_checkpoint_field(checkpoint, "teacher", "root")
    teacher_input_dim = teacher_block.get("input_dim")
    if teacher_input_dim is None:
        teacher_input_dim = get_checkpoint_metadata(checkpoint).get("teacher_input_dim")
    if teacher_input_dim is not None and int(teacher_input_dim) != int(x_teacher.shape[1]):
        raise ValueError(
            "Teacher input dimension mismatch: "
            f"checkpoint expects {int(teacher_input_dim)} PCA features but current data produced {x_teacher.shape[1]}. "
            "Make sure the checkpoint-compatible preprocessing settings are used for this dataset."
        )

    teacher_model = build_teacher_encoder(cfg, x_teacher.shape[1]).to(device)
    teacher_model.load_state_dict(require_checkpoint_field(teacher_block, "state_dict", "teacher"))
    teacher_model.eval()

    centers = require_checkpoint_field(teacher_block, "centers", "teacher").to(device=device, dtype=x_teacher.dtype)
    adj_sparse = ensure_sparse_adj(adj_norm, device)

    with torch.no_grad():
        z, _, _, _ = teacher_model.encoder(x_teacher, adj_sparse)
        q = soft_assign(z, centers, cfg.dec_alpha)

    return {
        "encoder": teacher_model,
        "centers": centers.detach(),
        "z": z.detach(),
        "q": q.detach(),
    }


def load_student_outputs(cfg, checkpoint: Dict, x_hvg: torch.Tensor, device: torch.device) -> List[Dict]:
    student_blocks = require_checkpoint_field(checkpoint, "students", "root")
    if not isinstance(student_blocks, (list, tuple)) or len(student_blocks) == 0:
        raise ValueError("Checkpoint does not contain any student models.")

    if getattr(cfg, "n_students", None) not in (None, len(student_blocks)):
        print(
            f"Warning: cfg.n_students={cfg.n_students} but checkpoint contains {len(student_blocks)} students. "
            "Using checkpoint contents."
        )
    cfg.n_students = len(student_blocks)

    ensemble_alpha = checkpoint.get("ensemble_alpha")
    if ensemble_alpha is not None:
        ensemble_alpha = np.asarray(ensemble_alpha, dtype=np.float32)

    students: List[Dict] = []

    for student_idx, student_ckpt in enumerate(student_blocks):
        gene_idx_np = np.asarray(require_checkpoint_field(student_ckpt, "gene_idx", f"students[{student_idx}]"), dtype=np.int64)
        if gene_idx_np.ndim != 1 or gene_idx_np.size == 0:
            raise ValueError(f"students[{student_idx}].gene_idx must be a non-empty 1D array.")
        if int(gene_idx_np.min()) < 0:
            raise ValueError(f"students[{student_idx}].gene_idx contains negative indices.")
        if int(gene_idx_np.max()) >= int(x_hvg.shape[1]):
            raise ValueError(
                f"Student {student_idx} expects gene index up to {int(gene_idx_np.max())}, "
                f"but the current dataset only has {x_hvg.shape[1]} HVGs after preprocessing. "
                "The checkpoint is not compatible with the current feature space."
            )

        expected_input_dim = student_ckpt.get("input_dim")
        if expected_input_dim is not None and int(expected_input_dim) != int(gene_idx_np.size):
            raise ValueError(
                f"students[{student_idx}].input_dim={int(expected_input_dim)} does not match "
                f"gene_idx length {gene_idx_np.size}."
            )

        gene_idx = torch.from_numpy(gene_idx_np).to(device=device, dtype=torch.long)
        x_sub = x_hvg.index_select(1, gene_idx)

        student_model = build_student_encoder(cfg, x_sub.shape[1]).to(device)
        student_model.load_state_dict(require_checkpoint_field(student_ckpt, "state_dict", f"students[{student_idx}]"))
        student_model.eval()

        centers = require_checkpoint_field(student_ckpt, "centers", f"students[{student_idx}]").to(device=device, dtype=x_sub.dtype)
        alpha_value = student_ckpt.get("alpha")
        if alpha_value is None and ensemble_alpha is not None and student_idx < ensemble_alpha.shape[0]:
            alpha_value = ensemble_alpha[student_idx]
        if alpha_value is None:
            raise KeyError(f"Checkpoint is missing students[{student_idx}].alpha and ensemble_alpha fallback.")
        alpha = float(alpha_value)

        with torch.no_grad():
            z, _ = student_model(x_sub)
            q = soft_assign(z, centers, cfg.dec_alpha)

        print(f"[Student {student_idx}] loaded feature_subspace={x_sub.shape[1]}")
        students.append(
            {
                "encoder": student_model,
                "centers": centers.detach(),
                "q": q.detach(),
                "z": z.detach(),
                "alpha": alpha,
                "gene_idx": gene_idx_np,
            }
        )

    return students


def peak_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_alloc = float(torch.cuda.max_memory_allocated())
        peak_reserved = float(torch.cuda.max_memory_reserved())
    else:
        peak_alloc = 0.0
        peak_reserved = 0.0
    print(f"[Peak_alloc {peak_alloc / 1024**2}] Peak_reserved={peak_reserved / 1024**2}")
    return peak_alloc, peak_reserved


def export_artifacts(
    cfg,
    output_dir: Path,
    teacher_out: Dict,
    teacher_scores: Optional[Dict],
    ensemble_scores: Optional[Dict],
    students: List[Dict],
    q_ens: torch.Tensor,
    z_ens: torch.Tensor,
    pred,
    alpha,
    labels,
    elapsed_seconds: float,
    peak_alloc: float,
    peak_reserved: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_z_np = teacher_out["z"].detach().cpu().numpy()
    if labels is not None:
        labels_col = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        if teacher_z_np.shape[0] != labels_col.shape[0]:
            raise ValueError(
                f"teacher_z rows ({teacher_z_np.shape[0]}) must match labels length ({labels_col.shape[0]})."
            )
        teacher_z_to_save = np.concatenate([teacher_z_np, labels_col], axis=1)
    else:
        teacher_z_to_save = teacher_z_np

    teacher_q_np = teacher_out["q"].detach().cpu().numpy()
    teacher_pred_np = teacher_q_np.argmax(axis=1).astype(np.int64)
    ensemble_q_np = q_ens.detach().cpu().numpy()
    ensemble_z_np = z_ens.detach().cpu().numpy()
    ensemble_pred_np = np.asarray(pred, dtype=np.int64)

    np.save(output_dir / "teacher_z.npy", teacher_z_to_save)
    np.save(output_dir / "teacher_q.npy", teacher_q_np)
    np.save(output_dir / "teacher_pred.npy", teacher_pred_np)
    np.save(output_dir / "ensemble_q.npy", ensemble_q_np)
    np.save(output_dir / "ensemble_z.npy", ensemble_z_np)
    np.save(output_dir / "ensemble_pred.npy", ensemble_pred_np)
    np.save(output_dir / "student_alpha.npy", np.asarray(alpha, dtype=np.float32))

    ensemble_z_pred_csv = output_dir / "ensemble_z_pred.csv"
    with open(ensemble_z_pred_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = [f"z_{i}" for i in range(ensemble_z_np.shape[1])] + ["pred"]
        writer.writerow(header)
        for row_z, row_pred in zip(ensemble_z_np, ensemble_pred_np):
            writer.writerow([float(v) for v in row_z] + [int(row_pred)])

    for student_idx, student in enumerate(students):
        np.save(output_dir / f"student_{student_idx}_q.npy", student["q"].detach().cpu().numpy())
        np.save(output_dir / f"student_{student_idx}_z.npy", student["z"].detach().cpu().numpy())
        np.save(output_dir / f"student_{student_idx}_genes.npy", np.asarray(student["gene_idx"], dtype=np.int64))

    summary_row = cfg_to_summary_dict(cfg)
    summary_row.update(
        {
            "dataset_name": cfg.load_dataset_name,
            "checkpoint_path": str(cfg.checkpoint_path),
            "teacher_acc": "" if teacher_scores is None else float(teacher_scores.get("acc", "")),
            "teacher_nmi": "" if teacher_scores is None else float(teacher_scores.get("nmi", "")),
            "teacher_ari": "" if teacher_scores is None else float(teacher_scores.get("ari", "")),
            "ensemble_acc": "" if ensemble_scores is None else float(ensemble_scores.get("acc", "")),
            "ensemble_nmi": "" if ensemble_scores is None else float(ensemble_scores.get("nmi", "")),
            "ensemble_ari": "" if ensemble_scores is None else float(ensemble_scores.get("ari", "")),
            "Time": float(elapsed_seconds),
            "peak_alloc": float(peak_alloc),
            "peak_reserved": float(peak_reserved),
        }
    )

    summary_csv_path = output_dir / "summary.csv"
    fieldnames = list(summary_row.keys())
    csv_exists = summary_csv_path.exists() and summary_csv_path.stat().st_size > 0
    write_mode = "a"
    encoding = "utf-8" if csv_exists else "utf-8-sig"

    if csv_exists:
        with open(summary_csv_path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline().strip()
        current_header = ",".join(fieldnames)
        if first_line != current_header:
            write_mode = "w"
            encoding = "utf-8-sig"

    with open(summary_csv_path, write_mode, newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if write_mode == "w" or not csv_exists:
            writer.writeheader()
        writer.writerow(summary_row)

    print(f"Artifacts saved to: {output_dir}")


def run(cfg):
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    checkpoint = load_checkpoint(cfg, device)
    restore_cfg_from_checkpoint(cfg, checkpoint)

    output_dir = resolve_output_dir(cfg)
    x_hvg, x_teacher, adj_norm, labels = load_data_from_main(cfg)
    resolve_n_clusters(cfg, labels, checkpoint)

    x_hvg = x_hvg.to(device)
    x_teacher = x_teacher.to(device)
    adj_norm = adj_norm.to(device)

    print(
        f"Inference data loaded: cells={x_hvg.shape[0]}, hvg={x_hvg.shape[1]}, "
        f"pca_dim={x_teacher.shape[1]}, clusters={cfg.n_clusters}"
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    teacher_out = load_teacher_outputs(cfg, checkpoint, x_teacher, adj_norm, device)
    teacher_pred = teacher_out["q"].argmax(dim=1).detach().cpu().numpy()
    if labels is not None:
        teacher_scores = clustering_metrics(labels, teacher_pred)
        print(f"Teacher inference metrics: {teacher_scores}")
    else:
        teacher_scores = None
        print("Teacher inference metrics skipped (labels unavailable).")

    students = load_student_outputs(cfg, checkpoint, x_hvg, device)
    q_ens, z_ens, pred, alpha = ensemble_students(students, device)
    if labels is not None:
        ensemble_scores = clustering_metrics(labels, pred)
        print(f"Ensemble inference metrics: {ensemble_scores}")
    else:
        ensemble_scores = None
        print("Ensemble inference metrics skipped (labels unavailable).")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start
    peak_alloc, peak_reserved = peak_memory_stats()

    # export_artifacts(
    #     cfg=cfg,
    #     output_dir=output_dir,
    #     teacher_out=teacher_out,
    #     teacher_scores=teacher_scores,
    #     ensemble_scores=ensemble_scores,
    #     students=students,
    #     q_ens=q_ens,
    #     z_ens=z_ens,
    #     pred=pred,
    #     alpha=alpha,
    #     labels=labels,
    #     elapsed_seconds=elapsed_seconds,
    #     peak_alloc=peak_alloc,
    #     peak_reserved=peak_reserved,
    # )

    # print("Inference time:", elapsed_seconds, "seconds")


if __name__ == "__main__":
    run(parse_config())
