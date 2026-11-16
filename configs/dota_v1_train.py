"""Expanded server G2 training recipe with portable release paths."""
auto_scale_lr = {'base_batch_size': 16, 'enable': False}

backend_args = None

custom_hooks = []

custom_imports = {'allow_failed_imports': False,
 'imports': ['mmdet.datasets.transforms',
             'core.ovad',
             'core.geo_optim_wrapper',
             'core.sgr_grounding_dino']}

data_root = 'data/DOTA_v1_0/'

dataset_type = 'OVADDota15CocoDataset'

default_hooks = {'checkpoint': {'interval': 4,
                'max_keep_ckpts': -1,
                'save_best': None,
                'save_last': True,
                'type': 'CheckpointHook'},
 'logger': {'interval': 100, 'type': 'LoggerHook'},
 'param_scheduler': {'type': 'ParamSchedulerHook'},
 'sampler_seed': {'type': 'DistSamplerSeedHook'},
 'timer': {'type': 'IterTimerHook'},
 'visualization': {'draw': False, 'type': 'DetVisualizationHook'}}

default_scope = 'mmdet'

dota_classes = ('plane',
 'ship',
 'storage-tank',
 'baseball-diamond',
 'basketball-court',
 'ground-track-field',
 'harbor',
 'bridge',
 'large-vehicle',
 'small-vehicle',
 'roundabout')

dota_metainfo = {'classes': ('plane',
             'ship',
             'storage-tank',
             'baseball-diamond',
             'basketball-court',
             'ground-track-field',
             'harbor',
             'bridge',
             'large-vehicle',
             'small-vehicle',
             'roundabout')}

env_cfg = {'cudnn_benchmark': False,
 'dist_cfg': {'backend': 'nccl'},
 'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0}}

generic_checkpoint = 'groundingdino_swint_ogc_mmdet-822d7e9d.pth'

lang_model_name = 'bert-base-uncased'

launcher = 'none'

load_from = 'groundingdino_swint_ogc_mmdet-822d7e9d.pth'

log_level = 'INFO'

log_processor = {'by_epoch': True, 'type': 'LogProcessor', 'window_size': 50}

max_epochs = 12

model = {'as_two_stage': True,
 'backbone': {'attn_drop_rate': 0.0,
              'convert_weights': True,
              'depths': [2, 2, 6, 2],
              'drop_path_rate': 0.2,
              'drop_rate': 0.0,
              'embed_dims': 96,
              'mlp_ratio': 4,
              'num_heads': [3, 6, 12, 24],
              'out_indices': (1, 2, 3),
              'patch_norm': True,
              'qk_scale': None,
              'qkv_bias': True,
              'type': 'SwinTransformer',
              'window_size': 7,
              'with_cp': True},
 'bbox_head': {'contrastive_cfg': {'bias': False, 'log_scale': 0.0, 'max_text_len': 512},
               'loss_bbox': {'loss_weight': 5.0, 'type': 'L1Loss'},
               'loss_cls': {'alpha': 0.25,
                            'gamma': 2.0,
                            'loss_weight': 1.0,
                            'type': 'FocalLoss',
                            'use_sigmoid': True},
               'loss_iou': {'loss_weight': 2.0, 'type': 'GIoULoss'},
               'num_classes': 256,
               'sgr_alpha_bg': 1.0,
               'sync_cls_avg_factor': True,
               'type': 'SGRGroundingDINOHead'},
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [123.675, 116.28, 103.53],
                       'pad_mask': False,
                       'std': [58.395, 57.12, 57.375],
                       'type': 'DetDataPreprocessor'},
 'decoder': {'layer_cfg': {'cross_attn_cfg': {'dropout': 0.0, 'embed_dims': 256, 'num_heads': 8},
                           'cross_attn_text_cfg': {'dropout': 0.0,
                                                   'embed_dims': 256,
                                                   'num_heads': 8},
                           'ffn_cfg': {'embed_dims': 256,
                                       'feedforward_channels': 2048,
                                       'ffn_drop': 0.0},
                           'self_attn_cfg': {'dropout': 0.0, 'embed_dims': 256, 'num_heads': 8}},
             'num_layers': 6,
             'post_norm_cfg': None,
             'return_intermediate': True},
 'dn_cfg': {'box_noise_scale': 1.0,
            'group_cfg': {'dynamic': True, 'num_dn_queries': 100, 'num_groups': None},
            'label_noise_scale': 0.5},
 'encoder': {'fusion_layer_cfg': {'embed_dim': 1024,
                                  'init_values': 0.0001,
                                  'l_dim': 256,
                                  'num_heads': 4,
                                  'v_dim': 256},
             'layer_cfg': {'ffn_cfg': {'embed_dims': 256,
                                       'feedforward_channels': 2048,
                                       'ffn_drop': 0.0},
                           'self_attn_cfg': {'dropout': 0.0, 'embed_dims': 256, 'num_levels': 4}},
             'num_cp': 0,
             'num_layers': 6,
             'text_layer_cfg': {'ffn_cfg': {'embed_dims': 256,
                                            'feedforward_channels': 1024,
                                            'ffn_drop': 0.0},
                                'self_attn_cfg': {'dropout': 0.0,
                                                  'embed_dims': 256,
                                                  'num_heads': 4}}},
 'language_model': {'add_pooling_layer': False,
                    'name': 'bert-base-uncased',
                    'pad_to_max': False,
                    'special_tokens_list': ['[CLS]', '[SEP]', '.', '?'],
                    'type': 'BertModel',
                    'use_sub_sentence_represent': True},
 'neck': {'act_cfg': None,
          'bias': True,
          'in_channels': [192, 384, 768],
          'kernel_size': 1,
          'norm_cfg': {'num_groups': 32, 'type': 'GN'},
          'num_outs': 4,
          'out_channels': 256,
          'type': 'ChannelMapper'},
 'num_queries': 900,
 'positional_encoding': {'normalize': True, 'num_feats': 128, 'offset': 0.0, 'temperature': 20},
 'sgr_cfg': {'alpha_bg': 1.0,
             'base_iou_thr': 0.5,
             'build_teacher_zero_effect': True,
             'debug_stats': True,
             'enabled': True,
             'exclude_dn_queries': True,
             'forbid_novel_prompt': True,
             'generic_prompts': ('object',
                                 'foreground object',
                                 'physical entity',
                                 'man-made object'),
             'geo_cfg': {'detach_visual': False,
                         'enabled': True,
                         'lambda_geo': 0.1,
                         'max_gt_per_image': 16,
                         'max_real_per_image': 64,
                         'roi_scale': 1.25,
                         'seed': 3407,
                         'warmup_iters': 1669},
             'lambda_budget': 0.0,
             'lambda_crsb': 0.0,
             'lambda_gfr': 0.0,
             'lambda_grr': 0.0,
             'lambda_rvrd': 0.0,
             'lambda_scr': 0.05,
             'log_loss_keys': False,
             'log_scr_keys': True,
             'match_iou_thr': 0.3,
             'max_protected': 16,
             'novel_class_strings': ('tennis-court',
                                     'helicopter',
                                     'soccer-ball-field',
                                     'swimming-pool'),
             'score_thr': 0.106708481908,
             'scr_temperature': 1.0,
             'scr_weight_max': 4.0,
             'scr_weight_min': 0.25,
             'semantic_anchor_prompts': ('plane',
                                         'ship',
                                         'storage-tank',
                                         'baseball-diamond',
                                         'basketball-court',
                                         'ground-track-field',
                                         'harbor',
                                         'bridge',
                                         'large-vehicle',
                                         'small-vehicle',
                                         'roundabout',
                                         'object',
                                         'foreground object',
                                         'physical entity',
                                         'man-made object'),
             'semantic_base_count': 11,
             'stability_thr': 0.7,
             'teacher_checkpoint': 'groundingdino_swint_ogc_mmdet-822d7e9d.pth',
             'teacher_config': 'configs/dota_v1_teacher.py',
             'teacher_frozen': True,
             'teacher_no_grad': True,
             'teacher_prompt_set': 'D1',
             'warmup_iters': 1669},
 'test_cfg': {'max_per_img': 300},
 'train_cfg': {'assigner': {'match_costs': [{'type': 'BinaryFocalLossCost', 'weight': 2.0},
                                            {'box_format': 'xywh',
                                             'type': 'BBoxL1Cost',
                                             'weight': 5.0},
                                            {'iou_mode': 'giou', 'type': 'IoUCost', 'weight': 2.0}],
                            'type': 'HungarianAssigner'}},
 'type': 'SelectiveGenericRetentionGroundingDINO',
 'with_box_refine': True}

optim_wrapper = {'accumulative_counts': 2,
 'clip_grad': {'max_norm': 0.1, 'norm_type': 2},
 'optimizer': {'lr': 0.0001, 'type': 'AdamW', 'weight_decay': 0.0001},
 'paramwise_cfg': {'custom_keys': {'absolute_pos_embed': {'decay_mult': 0.0},
                                   'backbone': {'lr_mult': 0.1},
                                   'geo_head': {'geo_group': True, 'lr_mult': 1.0},
                                   'language_model': {'lr_mult': 0.0}}},
 'type': 'GeoIsolatedClipOptimWrapper'}

ovad_arm = 'G2'

ovad_checkpoint_selection = 'fixed_final_epoch_12_no_novel_selection'

ovad_eval_protocol = 'base_val_only'

ovad_module = 'sgr_dscr_ubp_scr'

ovad_protocol_hash = 'a01aedabda599587377fea24c14830694f476b94f6eb2c5f64041c86ed81097f'

ovad_protocol_id = 'dota_v1_0_descreg_vkdet_11_4_uniform800'

ovad_protocol_path = 'configs/ovad_protocol_11_4.yaml'

ovad_reference_run = 'DOTA_V1_LONGITUDINAL_D_A2_SEED3407_2X4_ACC2'

ovad_reference_system = 'drift_aware_semantic_competition_retention'

ovad_trainable_mask = 'baseline_optimizer_mask_language_model_lr_mult_0'

ovad_training_regime = 'strict_image_exclusive_descreg_vkdet_11_4'

param_scheduler = [{'begin': 0, 'by_epoch': False, 'end': 1000, 'start_factor': 0.001, 'type': 'LinearLR'},
 {'begin': 0,
  'by_epoch': True,
  'end': 12,
  'gamma': 0.1,
  'milestones': [8, 11],
  'type': 'MultiStepLR'}]

randomness = {'deterministic': False, 'seed': 3407}

resume = False

semantic_anchor_prompts = ('plane',
 'ship',
 'storage-tank',
 'baseball-diamond',
 'basketball-court',
 'ground-track-field',
 'harbor',
 'bridge',
 'large-vehicle',
 'small-vehicle',
 'roundabout',
 'object',
 'foreground object',
 'physical entity',
 'man-made object')

test_cfg = {'type': 'TestLoop'}

test_dataloader = {'batch_size': 1,
 'dataset': {'ann_file': 'data/DOTA_v1_0/annotations/val_base.json',
             'annotation_scope': 'base',
             'backend_args': None,
             'data_prefix': {'img': 'val/images/'},
             'data_root': 'data/DOTA_v1_0/',
             'forbid_novel_annotations': True,
             'forbid_novel_prompt': True,
             'pipeline': [{'backend_args': None, 'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (1024, 1024), 'type': 'FixScaleResize'},
                          {'type': 'LoadAnnotations', 'with_bbox': True},
                          {'meta_keys': ('img_id',
                                         'img_path',
                                         'ori_shape',
                                         'img_shape',
                                         'scale_factor',
                                         'text',
                                         'custom_entities'),
                           'type': 'PackDetInputs'}],
             'prompt_scope': 'base',
             'protocol_path': 'configs/ovad_protocol_11_4.yaml',
             'return_classes': True,
             'test_mode': True,
             'type': 'OVADDota15CocoDataset'},
 'drop_last': False,
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

test_evaluator = {'ann_file': 'data/DOTA_v1_0/annotations/eval_gzsd_all.json',
 'ap_max_dets': 100,
 'backend_args': None,
 'classwise': True,
 'eval_protocol': 'gzsd',
 'format_only': False,
 'metric': 'bbox',
 'proposal_nums': (100, 300, 900),
 'protocol_path': 'configs/ovad_protocol_11_4.yaml',
 'type': 'OVADCocoMetric'}

test_pipeline = [{'backend_args': None, 'type': 'LoadImageFromFile'},
 {'keep_ratio': True, 'scale': (1024, 1024), 'type': 'FixScaleResize'},
 {'type': 'LoadAnnotations', 'with_bbox': True},
 {'meta_keys': ('img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'text',
                'custom_entities'),
  'type': 'PackDetInputs'}]

train_ann = 'data/DOTA_v1_0/annotations/instances_train_image_exclusive.json'

train_cfg = {'max_epochs': 12, 'type': 'EpochBasedTrainLoop', 'val_interval': 13}

train_dataloader = {'batch_sampler': {'type': 'AspectRatioBatchSampler'},
 'batch_size': 4,
 'dataset': {'ann_file': 'data/DOTA_v1_0/annotations/instances_train_image_exclusive.json',
             'annotation_scope': 'base',
             'backend_args': None,
             'data_prefix': {'img': 'train/images/'},
             'data_root': 'data/DOTA_v1_0/',
             'filter_cfg': {'filter_empty_gt': False, 'min_size': 32},
             'forbid_novel_annotations': True,
             'forbid_novel_prompt': True,
             'pipeline': [{'backend_args': None, 'type': 'LoadImageFromFile'},
                          {'type': 'LoadAnnotations', 'with_bbox': True},
                          {'prob': 0.5, 'type': 'RandomFlip'},
                          {'keep_ratio': True,
                           'scales': [(800, 1024), (896, 1024), (1024, 1024), (1152, 1024)],
                           'type': 'RandomChoiceResize'},
                          {'meta_keys': ('img_id',
                                         'img_path',
                                         'ori_shape',
                                         'img_shape',
                                         'scale_factor',
                                         'flip',
                                         'flip_direction',
                                         'text',
                                         'custom_entities'),
                           'type': 'PackDetInputs'}],
             'prompt_scope': 'base',
             'protocol_path': 'configs/ovad_protocol_11_4.yaml',
             'return_classes': True,
             'type': 'OVADDota15CocoDataset'},
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'shuffle': True, 'type': 'DefaultSampler'}}

train_pipeline = [{'backend_args': None, 'type': 'LoadImageFromFile'},
 {'type': 'LoadAnnotations', 'with_bbox': True},
 {'prob': 0.5, 'type': 'RandomFlip'},
 {'keep_ratio': True,
  'scales': [(800, 1024), (896, 1024), (1024, 1024), (1152, 1024)],
  'type': 'RandomChoiceResize'},
 {'meta_keys': ('img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
                'flip',
                'flip_direction',
                'text',
                'custom_entities'),
  'type': 'PackDetInputs'}]

val_base_ann = 'data/DOTA_v1_0/annotations/val_base.json'

val_cfg = {'type': 'ValLoop'}

val_dataloader = {'batch_size': 1,
 'dataset': {'ann_file': 'data/DOTA_v1_0/annotations/val_base.json',
             'annotation_scope': 'base',
             'backend_args': None,
             'data_prefix': {'img': 'val/images/'},
             'data_root': 'data/DOTA_v1_0/',
             'forbid_novel_annotations': True,
             'forbid_novel_prompt': True,
             'pipeline': [{'backend_args': None, 'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (1024, 1024), 'type': 'FixScaleResize'},
                          {'type': 'LoadAnnotations', 'with_bbox': True},
                          {'meta_keys': ('img_id',
                                         'img_path',
                                         'ori_shape',
                                         'img_shape',
                                         'scale_factor',
                                         'text',
                                         'custom_entities'),
                           'type': 'PackDetInputs'}],
             'prompt_scope': 'base',
             'protocol_path': 'configs/ovad_protocol_11_4.yaml',
             'return_classes': True,
             'test_mode': True,
             'type': 'OVADDota15CocoDataset'},
 'drop_last': False,
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

val_evaluator = {'ann_file': 'data/DOTA_v1_0/annotations/eval_gzsd_all.json',
 'ap_max_dets': 100,
 'backend_args': None,
 'classwise': True,
 'eval_protocol': 'gzsd',
 'format_only': False,
 'metric': 'bbox',
 'proposal_nums': (100, 300, 900),
 'protocol_path': 'configs/ovad_protocol_11_4.yaml',
 'type': 'OVADCocoMetric'}

val_gzsd_ann = 'data/DOTA_v1_0/annotations/eval_gzsd_all.json'

val_zsd_ann = 'data/DOTA_v1_0/annotations/eval_zsd_novel.json'

vis_backends = [{'type': 'LocalVisBackend'}]

visualizer = {'name': 'visualizer', 'type': 'DetLocalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}]}

work_dir = 'work_dirs/dota_v1'
