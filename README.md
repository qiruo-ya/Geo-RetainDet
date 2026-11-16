# RetainDet

**RetainDet — Novel Category Discovery via Proposal Retention for Open-Vocabulary Aerial Object Detection**

## Overview

RetainDet adapts Grounding DINO for open-vocabulary aerial object detection. Semantic Distribution Retention (SDR) preserves pretrained semantic responses during base-category adaptation. Geometric Objectness Learning (GOL) learns box-alignment quality through shared visual features. Remote-Sensing Expert Rectification (RER) uses a frozen remote-sensing vision-language expert to conditionally refine region labels.

## Included implementation

The release includes the SDR/GOL detector, geometric optimizer, native-evidence routing utility, DOTA-v1.0 dataset and metrics, a complete expanded training configuration, teacher architecture and training/evaluation entry points. The configuration comes from the completed server G2 run with portable paths and focused module imports.

| Path | Purpose |
| --- | --- |
| `core/sgr_grounding_dino.py` | Semantic retention and geometric quality learning |
| `core/geo_optim_wrapper.py` | Geometric-head optimizer handling |
| `core/native_preserving_expert.py` | Native-evidence preservation utility |
| `core/ovad/` | Protocol-aware dataset and evaluation metrics |
| `configs/dota_v1_train.py` | Full DOTA-v1.0 training recipe |
| `configs/dota_v1_teacher.py` | Frozen teacher architecture |
| `configs/ovad_protocol_11_4.yaml` | DOTA-v1.0 class split and protocol |
| `tools/train.py` | Single-GPU/distributed training and resume |
| `tools/test.py` | Detector GZSD/ZSD evaluation |

Legacy source names include SGR, SCR and `geo_cfg`. The server-specific checkpoint-provenance audit hook is omitted from the portable configuration. Training losses, model settings and optimizer settings are retained. The source detector includes experimental branches selected through configuration.

## Setup

Install a compatible MMDetection environment with Grounding DINO support, including PyTorch, torchvision, MMCV, MMEngine, transformers, SciPy, NumPy, PyYAML, pycocotools and terminaltables. Follow the MMDetection installation guide for CUDA-compatible versions.

Run commands from this repository root. Place the generic Grounding DINO checkpoint at `groundingdino_swint_ogc_mmdet-822d7e9d.pth` and the BERT tokenizer/model directory at `bert-base-uncased/`. These paths occur in both the training and teacher configurations and can be edited for your environment.

Prepare the following dataset layout using the reference 800-pixel patch protocol.

```text
data/DOTA_v1_0/
  train/images/
  val/images/
  annotations/
    instances_train_image_exclusive.json
    val_base.json
    eval_gzsd_all.json
    eval_zsd_novel.json
```

Annotation category IDs and names must match `configs/ovad_protocol_11_4.yaml`. Training excludes original images containing novel objects before patch extraction.

## Training

The reference recipe uses two GPUs, batch size 4 per GPU and accumulation 2, with seed 3407 and 12 epochs.

```bash
python -m torch.distributed.run --nproc_per_node=2 tools/train.py configs/dota_v1_train.py --launcher pytorch
```

Resume the latest checkpoint in the same work directory.

```bash
python -m torch.distributed.run --nproc_per_node=2 tools/train.py configs/dota_v1_train.py --launcher pytorch --resume
```

A single-GPU invocation is also supported. Adjust accumulation to preserve the reference effective batch size.

```bash
python tools/train.py configs/dota_v1_train.py --cfg-options optim_wrapper.accumulative_counts=4
```

## Detector evaluation

```bash
python tools/test.py configs/dota_v1_train.py work_dirs/dota_v1/epoch_12.pth --protocol gzsd
python tools/test.py configs/dota_v1_train.py work_dirs/dota_v1/epoch_12.pth --protocol zsd
```

The entry point selects the appropriate annotation file and vocabulary and reports protocol-aware detector metrics at a 100-detection budget. Full RER evaluation additionally requires RS5M crop encoding, candidate selection, calibration and expert score combination. Those orchestration components remain outside this release. The native-preservation utility provides a reusable part of that inference path.

The packaging changes and new entry points have not been executed as training or evaluation jobs. The source configuration was collected from the completed server experiment.

## DOTA-v1.0 protocol

- Horizontal-box open-vocabulary detection with 11 base classes and 4 novel classes.
- Novel classes are tennis court, helicopter, soccer-ball field and swimming pool.
- Strict image-level exclusion removes original training images containing novel instances before patch selection.
- The reference data preparation uses 800-pixel patches, with 13,350 retained training patches and 6,259 validation patches.
- Evaluation uses IoU 0.5 and a maximum of 100 detections per image.

Obtain DOTA-v1.0 from its official distribution and prepare annotations according to the chosen split. Model initialization and expert checkpoints are obtained separately.

## Acknowledgements

RetainDet builds on Grounding DINO, MMDetection and RS5M. README organization follows the overview, setup and evaluation sections commonly used in open-vocabulary detection releases such as Detic.

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [DOTA](https://captain-whu.github.io/DOTA/)
- [Detic](https://github.com/facebookresearch/Detic)

## License

See [LICENSE](LICENSE). Third-party dependencies retain their respective licenses.
