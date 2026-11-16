"""Protocol-driven DOTA-v1.5 OVAD COCO dataset."""

from __future__ import annotations

import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.base_det_dataset import BaseDetDataset
from .protocol import (assert_annotation_policy, build_prompt_texts,
                       class_ids_for_scope, class_names_for_scope,
                       load_protocol)


@DATASETS.register_module()
class OVADDota15CocoDataset(BaseDetDataset):
    """DOTA-v1.5 COCO dataset driven by a single OVAD protocol file."""

    COCOAPI = COCO
    ANN_ID_UNIQUE = True

    def __init__(self,
                 *args,
                 protocol_path: str,
                 prompt_scope: str,
                 annotation_scope: str,
                 forbid_novel_annotations: bool = False,
                 forbid_novel_prompt: bool = False,
                 **kwargs) -> None:
        self.protocol_path = protocol_path
        self.protocol = load_protocol(protocol_path)
        self.protocol_hash = self.protocol['_sha256']
        self.prompt_scope = prompt_scope
        self.annotation_scope = annotation_scope
        self.forbid_novel_annotations = bool(forbid_novel_annotations)
        self.forbid_novel_prompt = bool(forbid_novel_prompt)
        self.prompt_classes = class_names_for_scope(self.protocol, prompt_scope)
        self.prompt_texts = build_prompt_texts(self.protocol, prompt_scope)
        self.annotation_cat_ids = class_ids_for_scope(self.protocol,
                                                      annotation_scope)
        self.prompt_cat_ids = class_ids_for_scope(self.protocol, prompt_scope)
        self.prompt_cat_id_to_label = {
            cat_id: label
            for label, cat_id in enumerate(self.prompt_cat_ids)
        }

        if self.forbid_novel_prompt:
            novel = set(class_names_for_scope(self.protocol, 'novel'))
            if set(self.prompt_classes) & novel:
                raise AssertionError(
                    'strict inductive train prompt contains novel classes')

        kwargs.setdefault('metainfo', dict(classes=self.prompt_classes))
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        assert_annotation_policy(
            self.ann_file,
            self.protocol,
            self.annotation_scope,
            forbid_novel=self.forbid_novel_annotations)

        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)

        self.cat_ids = list(self.annotation_cat_ids)
        self.cat2label = {
            cat_id: self.prompt_cat_id_to_label[cat_id]
            for cat_id in self.cat_ids
            if cat_id in self.prompt_cat_id_to_label
        }
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info
            })
            data_list.append(parsed_data_info)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids), (
                f"Annotation ids in '{self.ann_file}' are not unique!")

        del self.coco
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        data_info['ovad_protocol_hash'] = self.protocol_hash
        data_info['ovad_prompt_scope'] = self.prompt_scope
        data_info['ovad_annotation_scope'] = self.annotation_scope

        if self.return_classes:
            data_info['text'] = self.prompt_texts
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue

            cat_id = ann['category_id']
            if cat_id not in self.cat2label:
                continue

            instance = dict(
                bbox=[x1, y1, x1 + w, y1 + h],
                bbox_label=self.cat2label[cat_id],
                ignore_flag=1 if ann.get('iscrowd', False) else 0)
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            instances.append(instance)

        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        if self.test_mode or self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)

        ids_in_cat = set()
        for class_id in self.cat_ids:
            ids_in_cat |= set(self.cat_img_map.get(class_id, []))
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for data_info in self.data_list:
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)
        return valid_data_infos
