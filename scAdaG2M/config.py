import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unsupervised scRNA-seq deep clustering with teacher-student DEC boosting."
    )

    # Data loading (keep the three original scRNA routes)
    parser.add_argument("--load_dataset_dir", type=str, default="./data", help="Dataset root directory.")
    parser.add_argument("--load_dataset_name", type=str, default="Human1", help="Dataset name subdirectory.")
    parser.add_argument("--load_h5", type=str, default=None, help="Use legacy .h5 benchmark format file name.")
    parser.add_argument("--load_h5_2", type=str, default="Human1.h5", help="Use .h5 file with X/Y keys.")
    parser.add_argument("--load_sc_dataset", type=str, default=None, help="Dense expression matrix file for load_dense route.")
    parser.add_argument("--load_cell_type_labels", type=str, default=None, help="Cell type label file for load_dense route.")

    # Runtime
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train", help="Run full training or inference-only from a saved checkpoint.")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Output directory.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path for saving (train) or loading (infer).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--log_every", type=int, default=20, help="Logging interval (epochs).")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument(
        "--disable_thop",
        action="store_true",
        help="Disable THOP Params/GFLOPs profiling.",
    )

    # Preprocess
    parser.add_argument("--preprocess_top_gene_select", type=int, default=2000, help="Number of HVGs. -1 disables HVG selection.")

    # Graph and clustering
    parser.add_argument("--n_input", type=int, default=100, help="PCA dimension used by dataload.main.")
    parser.add_argument("--knn", type=int, default=15, help="k of KNN.")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters. Required when labels are unavailable in train mode.")

    # Model
    parser.add_argument("--embed_dim", type=int, default=20, help="Embedding dimension.")
    parser.add_argument("--dec_alpha", type=float, default=1.0, help="DEC alpha.")

    parser.add_argument("--teacher_model", type=str, choices=["IGAE"], default="IGAE", help="Teacher encoder (fixed): IGAE.")
    parser.add_argument("--gae_n_enc_1", type=int, default=256, help="IGAE encoder hidden dim 1.")
    parser.add_argument("--gae_n_enc_2", type=int, default=128, help="IGAE encoder hidden dim 2.")
    parser.add_argument("--gae_n_dec_1", type=int, default=128, help="IGAE decoder hidden dim 1.")
    parser.add_argument("--gae_n_dec_2", type=int, default=256, help="IGAE decoder hidden dim 2.")
    parser.add_argument("--dropout", type=float, default=0.0, help="IGAE dropout.")
    parser.add_argument("--lr_pretrain", type=float, default=1e-2, help="IGAE pretraining LR.")
    parser.add_argument("--rec_epoch", type=int, default=50, help="IGAE pretraining epochs.")
    parser.add_argument("--alpha_value", type=float, default=0.1, help="IGAE adjacency reconstruction loss weight.")

    parser.add_argument("--n_students", type=int, default=3, help="Number of students.")
    parser.add_argument("--rc1_ratio", type=float, default=1, help="RC-1 feature subspace ratio.")
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
    parser.add_argument("--student_dec_epochs", type=int, default=100, help="Student DEC epochs.")
    parser.add_argument("--student_dec_lr", type=float, default=1e-2, help="Student DEC LR.")

    # Losses
    parser.add_argument("--mask_rate", type=float, default=0.2, help="Feature mask rate for NA/aug.")
    parser.add_argument("--aug_noise_std", type=float, default=0.2, help="Feature jitter std.")
    parser.add_argument("--kd_target", type=str, choices=["q", "p"], default="q", help="KD target distribution.")
    parser.add_argument(
        "--kd_conf_quantile",
        type=float,
        default=0.8,
        help="Only distill samples below this teacher-uncertainty quantile (confidence gating).",
    )
    parser.add_argument("--lambda_dec", type=float, default=2, help="DEC loss weight.")
    parser.add_argument("--lambda_kd", type=float, default=2, help="KD loss weight.")
    parser.add_argument("--lambda_na", type=float, default=2, help="NA loss weight.")
    parser.add_argument("--lambda_rec", type=float, default=2, help="Reconstruction loss weight for student AE.")
    parser.add_argument("--lambda_na_pretrain", type=float, default=2, help="NA loss weight in student pretrain.")
    parser.add_argument("--boost_beta", type=float, default=2, help="AdaG2M-style node-weight update beta.")
    parser.add_argument("--boost_alpha", type=float, default=2, help="Boosting alpha.")


    return parser


def parse_config():
    args = build_parser().parse_args()
    return args


def getConfig():
    return parse_config()
