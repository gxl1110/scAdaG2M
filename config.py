import argparse
from pathlib import Path


DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_CHECKPOINT_PATH = str(Path(DEFAULT_OUTPUT_DIR) / "Human1" / "checkpoint.pt")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inference-only scRNA-seq clustering from a saved checkpoint."
    )

    # Data loading
    parser.add_argument("--load_dataset_dir", type=str, default="./data", help="Dataset root directory.")
    parser.add_argument("--load_dataset_name", type=str, default="Human1", help="Dataset name subdirectory.")
    parser.add_argument("--load_h5", type=str, default=None, help="Use legacy .h5 benchmark format file name.")
    parser.add_argument(
        "--load_h5_2",
        type=str,
        default=None,
        help="Use .h5 file with X/Y keys. If omitted, auto-detect <dataset>.h5 or a single .h5 file.",
    )
    parser.add_argument("--load_sc_dataset", type=str, default=None, help="Dense expression matrix file for load_dense route.")
    parser.add_argument("--load_cell_type_labels", type=str, default=None, help="Optional cell type label file for evaluation.")

    # Runtime
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Path to the saved inference checkpoint. If omitted, try "
            "outputs/<dataset>/checkpoint.pt then fallback to outputs/Human1/checkpoint.pt."
        ),
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")

    # Preprocess and graph
    parser.add_argument("--preprocess_top_gene_select", type=int, default=2000, help="Number of HVGs. -1 disables HVG selection.")
    parser.add_argument("--n_input", type=int, default=100, help="PCA dimension used by dataload.main.")
    parser.add_argument("--knn", type=int, default=15, help="k of KNN.")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters if not stored in checkpoint metadata.")

    # Model architecture
    parser.add_argument("--embed_dim", type=int, default=20, help="Embedding dimension.")
    parser.add_argument("--dec_alpha", type=float, default=1.0, help="DEC alpha.")
    parser.add_argument("--teacher_model", type=str, choices=["IGAE"], default="IGAE", help="Teacher encoder type.")
    parser.add_argument("--gae_n_enc_1", type=int, default=256, help="IGAE encoder hidden dim 1.")
    parser.add_argument("--gae_n_enc_2", type=int, default=128, help="IGAE encoder hidden dim 2.")
    parser.add_argument("--gae_n_dec_1", type=int, default=128, help="IGAE decoder hidden dim 1.")
    parser.add_argument("--gae_n_dec_2", type=int, default=256, help="IGAE decoder hidden dim 2.")
    parser.add_argument("--dropout", type=float, default=0.0, help="IGAE dropout.")
    parser.add_argument("--n_students", type=int, default=3, help="Number of student models in the checkpoint.")
    parser.add_argument("--student_layers", type=int, default=3, help="Student MLP layers.")
    parser.add_argument("--student_hidden_dim", type=int, default=20, help="Student hidden dim.")
    parser.add_argument(
        "--student_hidden_dims",
        type=str,
        default="512,256,128",
        help="Student MLP hidden dims (comma-separated), e.g. 512,256,128.",
    )
    parser.add_argument("--student_dropout", type=float, default=0.4, help="Student dropout.")
    parser.add_argument("--norm_type", type=str, default="layer", help="Student normalization type.")

    return parser


def parse_config():
    return build_parser().parse_args()


def getConfig():
    return parse_config()
