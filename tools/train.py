"""Train RetainDet with MMEngine from the repository root."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default='configs/dota_v1_train.py')
    parser.add_argument('--work-dir')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = args.launcher
    cfg.resume = args.resume
    if args.work_dir:
        cfg.work_dir = args.work_dir
    Runner.from_cfg(cfg).train()


if __name__ == '__main__':
    main()
