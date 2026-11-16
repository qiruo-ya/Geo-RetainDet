"""DOTA-v1.0 core training settings, applied to an MMDetection Grounding DINO config.

This fragment records the released SDR/GOL recipe. Supply the Grounding DINO
architecture, teacher config/checkpoint and dataset pipelines in the host project.
"""
custom_imports = dict(
    imports=['core.sgr_grounding_dino', 'core.geo_optim_wrapper'],
    allow_failed_imports=False)
model = dict(
    type='SelectiveGenericRetentionGroundingDINO',
    bbox_head=dict(type='SGRGroundingDINOHead', sgr_alpha_bg=1.0),
    sgr_cfg=dict(
        enabled=True, alpha_bg=1.0, lambda_scr=0.05,
        lambda_budget=0.0, warmup_iters=1669,
        teacher_frozen=True, teacher_no_grad=True,
        exclude_dn_queries=True, forbid_novel_prompt=True,
        geo_cfg=dict(
            enabled=True, detach_visual=False, lambda_geo=0.1,
            warmup_iters=1669, max_gt_per_image=16,
            max_real_per_image=64, roi_scale=1.25, seed=3407)))
optim_wrapper = dict(
    type='GeoIsolatedClipOptimWrapper', accumulative_counts=2,
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1),
        'language_model': dict(lr_mult=0.0),
        'geo_head': dict(lr_mult=1.0, geo_group=True)}))
train_dataloader = dict(batch_size=4)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12)
randomness = dict(seed=3407)
