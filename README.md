# RetainDet

**RetainDet — Novel Category Discovery via Proposal Retention for Open-Vocabulary Aerial Object Detection**

## Overview

RetainDet adapts Grounding DINO for open-vocabulary aerial object detection. Semantic Distribution Retention (SDR) preserves pretrained semantic responses during base-category adaptation. Geometric Objectness Learning (GOL) learns box-alignment quality through shared visual features. Remote-Sensing Expert Rectification (RER) uses a frozen remote-sensing vision-language expert to conditionally refine region labels.

## Release scope

This repository provides selected core implementation files and one DOTA-v1.0 configuration fragment. End-to-end training and evaluation require integration with the original MMDetection-based project, dataset preparation, teacher initialization and expert inference components. The fragment exposes the SDR/GOL settings and optimizer recipe. The full detector architecture, data pipelines and teacher configuration must be supplied by the host project.

## Code

| File | Purpose |
| --- | --- |
| `core/sgr_grounding_dino.py` | Grounding DINO integration, semantic retention losses and geometric quality head |
| `core/geo_optim_wrapper.py` | Optimizer wrapper for geometric-head gradient clipping |
| `core/native_preserving_expert.py` | Native-evidence preservation and budgeted expert-query routing utility |
| `configs/dota_v1_core.py` | DOTA-v1.0 SDR/GOL training settings |

The research source retains legacy identifiers such as `SGR`, `SCR` and `geo_cfg`. The detector file also contains experimental branches. The configuration selects the released SDR/GOL settings. The native-preservation utility is one inference component. Full RER additionally requires region crops, RS5M features, calibration, score combination and final suppression.

## Dependencies and integration

The detector implementation imports PyTorch, torchvision, SciPy, MMEngine and MMDetection with Grounding DINO support. The inference routing utility requires NumPy. Use a compatible MMDetection environment and place this repository on `PYTHONPATH`.

Merge `configs/dota_v1_core.py` into a complete Grounding DINO training configuration in the host project. Set the teacher configuration and checkpoint through `model.sgr_cfg`, prepare the dataset loaders, and register the custom modules through the supplied `custom_imports`. The recorded recipe uses 12 epochs, seed 3407, batch size 4 per GPU, gradient accumulation 2, and a two-GPU reference setup.

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
