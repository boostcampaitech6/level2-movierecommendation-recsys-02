import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="../data/train/", type=str, help="Data path")
    parser.add_argument("--max_len", default=50, type=int, help="Maximum length")
    parser.add_argument("--hidden_units", default=50, type=int, help="Embedding size")
    parser.add_argument("--num_heads", default=1, type=int, help="Number of multi-head layers")
    parser.add_argument("--num_layers", default=2, type=int, help="Number of blocks")
    parser.add_argument("--dropout_rate", default=0.5, type=float, help="Dropout rate")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--num_epochs", default=2, type=int, help="Number of epochs")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers")
    parser.add_argument("--mask_prob", default=0.15, type=float, help="Mask probability for cloze task")
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    # parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")
    parser.add_argument("--model", default="bert4rec", type=str, help="Model name")
    parser.add_argument("--output_dir", default="outputs/", type=str, help="Submission path")
    
    args = parser.parse_args()

    return args