# scAdaG2M

This paper proposes scAdaG2M, an AdaBoost-enhanced graph-to-MLP knowledge distillation framework that transfers topological knowledge learned by a GNN teacher to lightweight graph-free MLP students, thereby improving the identification of ambiguous and rare cell populations, mitigating over-smoothing, reducing graph dependence, and lowering computational costs for large-scale scRNA-seq clustering.

## Environment

- Python `3.9`
- PyTorch `2.1.0+cu118`

```bash
conda create -n scadag2m python=3.9 -y
conda activate scadag2m
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas scikit-learn scanpy h5py
```

## Run

The example dataset is at `data/Human1/Human1.h5`.

```bash
python train.py
```

## Output

Results are saved to:

```text
outputs/<dataset_name>/
```

Main files include:

- `summary.csv`
- `teacher_z.npy`
- `teacher_q.npy`
- `teacher_pred.npy`
- `ensemble_z.npy`
- `ensemble_q.npy`
- `ensemble_pred.npy`
- `ensemble_z_pred.csv
