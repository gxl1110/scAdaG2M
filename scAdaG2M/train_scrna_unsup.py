import csv
from pathlib import Path
from typing import Dict, List, Optional
import time
import numpy as np
import scipy.sparse as sp
import torch
import os
import pandas as pd

from config import parse_config
from dataload.main import getscData
from utils import (
    build_teacher_encoder,
    clustering_metrics,
    ensemble_students,
    entropy_per_cell,
    get_device,
    sample_feature_subspace,
    set_seed,
    train_single_student,
    train_teacher_dec,
)


def _to_numpy_float32(array_like) -> np.ndarray:
    if sp.issparse(array_like):
        return np.asarray(array_like.toarray(), dtype=np.float32)
    return np.asarray(array_like, dtype=np.float32)


def _to_int_labels(labels):
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


def _load_data_from_main(cfg):
    if getattr(cfg, "n_input", None) is None:
        cfg.n_input = getattr(cfg, "pca_dim", 100)

    data = getscData(cfg)
    x_hvg = torch.from_numpy(_to_numpy_float32(data.X_hvg))
    x_teacher = torch.from_numpy(_to_numpy_float32(data.X_pca))
    labels = _to_int_labels(getattr(data, "labels", None))

    adj_raw = data.adj_norm if getattr(data, "adj_norm", None) is not None else data.adj
    adj_norm = torch.tensor(adj_raw, dtype=torch.float32)
    return x_hvg, x_teacher, adj_norm, labels


def _to_csv_scalar(value):
    if value is None:
        return ""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, dict, set)):
        return str(value)
    return value


def _cfg_to_summary_dict(cfg) -> Dict[str, object]:
    return {key: _to_csv_scalar(val) for key, val in vars(cfg).items()}


def save_results_to_csv(filename, dataset_name, acc, nmi, ari, times, peak_alloc, peak_reserved):
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file_exists = os.path.isfile(filename)
    is_empty = not file_exists or os.path.getsize(filename) == 0
    headers = ['Dataset', 'ACC', 'NMI', 'ARI', 'Times', 'Peak_Alloc', 'Peak_Reserved']
    row_data = [
        dataset_name,
        f"{acc:.4f}",
        f"{nmi:.4f}",
        f"{ari:.4f}",
        f"{times:.4f}",
        int(peak_alloc),
        int(peak_reserved),
    ]

    try:
        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if is_empty:
                writer.writerow(headers)
            writer.writerow(row_data)
        print(f"Dataset '{dataset_name}' The results have been saved to '{filename}'")
    except IOError:
        print(f"Error: Unable to write to the file '{filename}'")
    except Exception as e:
        print(f"An unknown error occurred whilst writing to CSV: {e}")


def save_embedding_with_labels(fusion_emb, cluster_pred, dataset_name):


    output_dir = "./result/Ours_embedding/"
    fusion_emb_np = fusion_emb
    cluster_pred_np = np.asarray(cluster_pred).reshape(-1, 1)
    emb_with_labels = np.concatenate((fusion_emb_np, cluster_pred_np), axis=1)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{dataset_name}_fusion_embedding_with_labels.csv")

    dim_names = [f"dim_{i}" for i in range(fusion_emb_np.shape[1])]
    column_names = dim_names + ['label']
    df_save = pd.DataFrame(emb_with_labels, columns=column_names)
    df_save.to_csv(save_path, index=False)

    print(f"Fusion embedding and tags have been saved: {save_path}")


def make_dir(directory_path, new_folder_name):
    """Creates an expected directory if it does not exist"""
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

def run(cfg):

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    output_root = Path(cfg.output_dir)
    resolved_output_dir = output_root / cfg.load_dataset_name
    cfg.output_dir = str(resolved_output_dir)

    if cfg.teacher_model.upper() != "IGAE":
        raise ValueError("Only IGAE teacher is supported in current code. Please set --teacher_model IGAE.")

    x_hvg, x_teacher, adj_norm, labels = _load_data_from_main(cfg)
    x_hvg = x_hvg.to(device)
    x_teacher = x_teacher.to(device)
    adj_norm = adj_norm.to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    if labels is None:
        raise ValueError("labels are required to infer n_clusters from int(np.unique(labels).shape[0]).")
    cfg.n_clusters = int(np.unique(labels).shape[0])

    print(
        f"Data loaded: cells={x_hvg.shape[0]}, hvg={x_hvg.shape[1]}, "
        f"pca_dim={x_teacher.shape[1]}, clusters={cfg.n_clusters}"
    )

    teacher_encoder = build_teacher_encoder(cfg, x_teacher.shape[1]).to(device)
    teacher_out = train_teacher_dec(cfg, teacher_encoder, adj_norm, x_teacher, label=labels)
    teacher_z = teacher_out["z"]
    teacher_q = teacher_out["q"]
    teacher_p = teacher_out["p"]
    teacher_pred = teacher_q.argmax(dim=1).detach().cpu().numpy()

    if labels is not None:
        teacher_scores = clustering_metrics(labels, teacher_pred)
        print(f"Teacher metrics: {teacher_scores}")
    else:
        teacher_scores = None
        print("Teacher metrics skipped (labels unavailable).")

    num_cells = x_hvg.shape[0]
    weights = torch.full((num_cells,), 1.0 / num_cells, device=device)
    uncertainty = entropy_per_cell(teacher_q)
    uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-12)

    students: List[Dict] = []
    student_metrics: List[Dict] = []
    rng = np.random.default_rng(cfg.seed)
    for student_idx in range(cfg.n_students):
        gene_idx_np = sample_feature_subspace(x_hvg.shape[1], cfg.rc1_ratio, rng)
        gene_idx = torch.from_numpy(gene_idx_np).to(device=device, dtype=torch.long)
        x_sub = x_hvg[:, gene_idx]
        print(
            f"[Student {student_idx}] feature_subspace={x_sub.shape[1]} "
            f"({cfg.rc1_ratio:.2f} of HVG)"
        )
        student_out, weights = train_single_student(
            cfg=cfg,
            x_sub=x_sub,
            teacher_q=teacher_q,
            teacher_p=teacher_p,
            weights=weights,
            uncertainty_norm=uncertainty_norm,
            student_idx=student_idx,
            labels=labels,
        )
        student_out["gene_idx"] = gene_idx_np
        student_pred = student_out["q"].argmax(dim=1).detach().cpu().numpy()
        if labels is not None:
            scores = clustering_metrics(labels, student_pred)
            print(f"Student {student_idx} metrics: {scores}")
        else:
            scores = None
            print(f"Student {student_idx} metrics skipped (labels unavailable).")
        student_metrics.append({"student_idx": int(student_idx), "metrics": scores})
        students.append(student_out)

    q_ens, z_ens, pred, alpha = ensemble_students(cfg, students, cfg.seed, device)
    if labels is not None:
        ensemble_scores = clustering_metrics(labels, pred)
        print(f"Ensemble metrics: {ensemble_scores}")
    else:
        ensemble_scores = None
        print("Ensemble metrics skipped (labels unavailable).")

    output_dir = resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_z_np = teacher_z.detach().cpu().numpy()
    labels_col = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    if teacher_z_np.shape[0] != labels_col.shape[0]:
        raise ValueError(
            f"teacher_z rows ({teacher_z_np.shape[0]}) must match labels length ({labels_col.shape[0]})."
        )
    teacher_z_with_label = np.concatenate([teacher_z_np, labels_col], axis=1)
    np.save(output_dir / "teacher_z.npy", teacher_z_with_label)
    np.save(output_dir / "teacher_q.npy", teacher_q.detach().cpu().numpy())
    np.save(output_dir / "teacher_pred.npy", teacher_pred)
    ensemble_q_np = q_ens.detach().cpu().numpy()
    ensemble_z_np = z_ens.detach().cpu().numpy()
    ensemble_pred_np = np.asarray(pred, dtype=np.int64)

    np.save(output_dir / "ensemble_q.npy", ensemble_q_np)
    np.save(output_dir / "ensemble_z.npy", ensemble_z_np)
    np.save(output_dir / "ensemble_pred.npy", ensemble_pred_np)
    np.save(output_dir / "student_alpha.npy", alpha)
    np.save(output_dir / "final_cell_weights.npy", weights.detach().cpu().numpy())

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
        np.save(output_dir / f"student_{student_idx}_genes.npy", student["gene_idx"])

    ensemble_acc = ensemble_scores.get("acc") if ensemble_scores is not None else None
    ensemble_nmi = ensemble_scores.get("nmi") if ensemble_scores is not None else None
    ensemble_ari = ensemble_scores.get("ari") if ensemble_scores is not None else None
    teacher_acc = teacher_scores.get("acc") if teacher_scores is not None else None
    teacher_nmi = teacher_scores.get("nmi") if teacher_scores is not None else None
    teacher_ari = teacher_scores.get("ari") if teacher_scores is not None else None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    times = end - start

    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()



    print(f"[Peak_alloc {peak_alloc / 1024**2}] Peak_reserved={peak_reserved / 1024**2} ")

    summary_row = _cfg_to_summary_dict(cfg)
    summary_row.update(
        {
            "dataset_name": cfg.load_dataset_name,
            "teacher_acc": "" if teacher_acc is None else float(teacher_acc),
            "teacher_nmi": "" if teacher_nmi is None else float(teacher_nmi),
            "teacher_ari": "" if teacher_ari is None else float(teacher_ari),
            "ensemble_acc": "" if ensemble_acc is None else float(ensemble_acc),
            "ensemble_nmi": "" if ensemble_nmi is None else float(ensemble_nmi),
            "ensemble_ari": "" if ensemble_ari is None else float(ensemble_ari),
            "Time": "" if times is None else float(times),
            "peak_alloc": "" if peak_alloc is None else float(peak_alloc),
            "peak_reserved": "" if peak_reserved is None else float(peak_reserved),
        }
    )

    summary_csv_path = output_dir / "summary.csv"
    csv_exists = summary_csv_path.exists() and summary_csv_path.stat().st_size > 0
    encoding = "utf-8" if csv_exists else "utf-8-sig"
    with open(summary_csv_path, "a", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if not csv_exists:
            writer.writeheader()
        writer.writerow(summary_row)

    print(f"Artifacts saved to: {output_dir}")

    csv_filename = "./results/Ours_results.csv"

    save_embedding_with_labels(ensemble_z_np, ensemble_pred_np, cfg.load_dataset_name)
    save_results_to_csv(csv_filename, cfg.load_dataset_name, ensemble_acc, ensemble_nmi, ensemble_ari, times, peak_alloc, peak_reserved)
    print("End-to-end time:", times, "seconds")




if __name__ == "__main__":
    run(parse_config())
