import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


def cluster_accuracy(y_pred, y_true, cluster_num=None):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    dim = int(max(y_true.max(), y_pred.max()) + 1)
    weight = np.zeros((dim, dim), dtype=np.int64)
    for idx in range(y_pred.shape[0]):
        weight[y_pred[idx], y_true[idx]] += 1

    row, col = linear_sum_assignment(weight.max() - weight)
    mapping = np.zeros(dim, dtype=np.int64)
    mapping[row] = col
    mapped_pred = mapping[y_pred]

    acc = float((mapped_pred == y_true).mean())
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    f1 = float(f1_score(y_true, mapped_pred, average="macro"))
    ari = float(adjusted_rand_score(y_true, y_pred))

    return acc, nmi, f1, ari
