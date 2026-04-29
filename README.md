# scAdaG2M

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

Default run:

```bash
python train.py
```

Run a specific dataset:

```bash
python train.py --load_dataset_name Human1
python train.py --load_dataset_name Human2
python train.py --load_dataset_name Human3
python train.py --load_dataset_name Human4
```

If the raw `.h5` file name is not `<dataset>.h5`, specify it explicitly:

