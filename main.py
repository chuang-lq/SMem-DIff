import yaml
from types import SimpleNamespace
import argparse
import os

from train import Trainer

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='train config file path')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    args = ParseArgs()
    with open(args.config, mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)
    config.exp_name = os.path.basename(args.config)[:-4]

    trainer = Trainer(config, config_path=args.config)
    trainer.run()
