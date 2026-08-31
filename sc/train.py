from config import parse_config
from train_scrna_unsup import run
import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    run(parse_config())
