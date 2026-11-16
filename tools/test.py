"""Evaluate detector predictions on the DOTA-v1.0 GZSD or ZSD split."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--protocol', choices=['gzsd', 'zsd'], default='gzsd')
    parser.add_argument('--work-dir', default='work_dirs/dota_v1_eval')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = args.checkpoint
    cfg.work_dir = args.work_dir
    cfg.launcher = 'none'
    dataset = cfg.test_dataloader.dataset
    scope = 'all' if args.protocol == 'gzsd' else 'novel'
    dataset.ann_file = cfg.val_gzsd_ann if args.protocol == 'gzsd' else cfg.val_zsd_ann
    dataset.prompt_scope = scope
    dataset.annotation_scope = scope
    dataset.forbid_novel_annotations = False
    dataset.forbid_novel_prompt = False
    dataset.pop('metainfo', None)
    cfg.test_evaluator.ann_file = dataset.ann_file
    cfg.test_evaluator.eval_protocol = args.protocol
    cfg.model.test_cfg.max_per_img = 100
    Runner.from_cfg(cfg).test()


if __name__ == '__main__':
    main()
