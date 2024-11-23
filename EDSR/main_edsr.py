import argparse
from train import train_model
from eval import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EDSR')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'], help='train or eval mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'eval':
        evaluate()
