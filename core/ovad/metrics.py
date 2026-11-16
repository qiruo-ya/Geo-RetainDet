"""Protocol-aware OVAD COCO metric."""

from __future__ import annotations

import json
import os.path as osp
import tempfile
from statistics import mean
from typing import List, Optional, Sequence, Union

import numpy as np
from mmdet.registry import METRICS
from mmdet.datasets.api_wrappers import COCOeval, COCOevalMP
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmengine.fileio import load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable
from .protocol import (class_ids_for_scope, class_names_for_scope,
                       load_protocol, normalize_class_name)


def _valid_mean(values):
    valid = [v for v in values if v is not None and v >= 0]
    return round(mean(valid), 3) if valid else -1.0


def harmonic_mean(base: float, novel: float) -> float:
    """Return HM with explicit zero handling for GZSD."""
    if base < 0 or novel < 0:
        return -1.0
    if base == 0 or novel == 0:
        return 0.0
    return round(2 * base * novel / (base + novel), 3)


def _rounded_stat(value: float) -> float:
    if value is None or value < 0:
        return -1.0
    return float(f'{round(float(value), 3)}')


@METRICS.register_module()
class OVADCocoMetric(CocoMetric):
    """CocoMetric plus ZSD/GZSD aggregation from one protocol object."""

    def __init__(self,
                 protocol_path: str,
                 eval_protocol: str,
                 diagnostic_classes: Optional[Sequence[str]] = None,
                 ap_max_dets: int = 100,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = True,
                 **kwargs) -> None:
        kwargs.setdefault('proposal_nums', (100, 300, 900))
        self.protocol = load_protocol(protocol_path)
        self.protocol_hash = self.protocol['_sha256']
        self.eval_protocol = eval_protocol
        if eval_protocol not in self.protocol['eval_protocols']:
            raise AssertionError(f'Unknown OVAD eval protocol: {eval_protocol}')
        prompt_scope = self.protocol['eval_protocols'][eval_protocol][
            'prompt_scope']
        self.eval_classes = list(
            class_names_for_scope(self.protocol, prompt_scope))
        self.prompt_cat_ids = list(class_ids_for_scope(self.protocol,
                                                       prompt_scope))
        self.base_classes = list(class_names_for_scope(self.protocol, 'base'))
        self.novel_classes = list(class_names_for_scope(self.protocol, 'novel'))
        self.base_cat_ids = list(class_ids_for_scope(self.protocol, 'base'))
        self.novel_cat_ids = list(class_ids_for_scope(self.protocol, 'novel'))
        self.diagnostic_classes = list(diagnostic_classes or [])
        self.ap_max_dets = int(ap_max_dets)
        self.ovad_classwise = bool(classwise)
        super().__init__(
            ann_file=ann_file, metric=metric, classwise=False, **kwargs)
        if self.ap_max_dets not in self.proposal_nums:
            raise AssertionError(
                f'ap_max_dets={self.ap_max_dets} must be in proposal_nums '
                f'{self.proposal_nums}')

    def compute_metrics(self, results: list):
        logger = MMLogger.get_current_instance()
        logger.info('OVAD protocol %s hash=%s eval=%s',
                    self.protocol['protocol_id'], self.protocol_hash,
                    self.eval_protocol)
        logger.info('OVAD AP/classwise maxDets=%s recall maxDets=%s',
                    self.ap_max_dets, self.proposal_nums)

        tmp_dir = None
        original_outfile_prefix = self.outfile_prefix
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            self.outfile_prefix = osp.join(tmp_dir.name, 'results')
        current_prefix = self.outfile_prefix
        try:
            # Prediction labels are prompt indices, so category-id mapping must
            # follow protocol prompt order rather than COCO category order.
            self.cat_ids = list(self.prompt_cat_ids)
            self._assert_category_order()
            self._log_category_order(logger)
            eval_results = super().compute_metrics(results)
            if self.format_only or 'bbox' not in self.metrics:
                return eval_results

            bbox_json = current_prefix + '.bbox.json'
            if self.eval_protocol == 'zsd':
                novel = self._evaluate_bbox_subset(
                    bbox_json, self.novel_cat_ids, 'novel')
                self._override_parent_bbox_stats(eval_results, novel, logger)
                self._add_classwise_stats(eval_results, novel, logger)
                self._add_prefixed_stats(eval_results, 'ovad/zsd', 'novel',
                                         novel)
                eval_results['ovad/AP_novel'] = novel['AP']
                eval_results['ovad/APs_novel'] = novel['APs']
            else:
                base = self._evaluate_bbox_subset(
                    bbox_json, self.base_cat_ids, 'base')
                novel = self._evaluate_bbox_subset(
                    bbox_json, self.novel_cat_ids, 'novel')
                all_stats = self._evaluate_bbox_subset(
                    bbox_json, self.base_cat_ids + self.novel_cat_ids, 'all')
                hm = harmonic_mean(base['AP'], novel['AP'])

                self._override_parent_bbox_stats(eval_results, all_stats,
                                                 logger)
                self._add_classwise_stats(eval_results, all_stats, logger)
                self._add_prefixed_stats(eval_results, 'ovad/gzsd', 'base',
                                         base)
                self._add_prefixed_stats(eval_results, 'ovad/gzsd', 'novel',
                                         novel)
                self._add_prefixed_stats(eval_results, 'ovad/gzsd', 'all',
                                         all_stats)
                eval_results['ovad/gzsd/HM'] = hm
                eval_results['ovad/AP_all'] = all_stats['AP']
                eval_results['ovad/AP_base'] = base['AP']
                eval_results['ovad/AP_novel'] = novel['AP']
                eval_results['ovad/APs_novel'] = novel['APs']
                eval_results['ovad/HM'] = hm
                eval_results['ovd/AP_all'] = all_stats['AP']
                eval_results['ovd/AP_base'] = base['AP']
                eval_results['ovd/AP_novel'] = novel['AP']
                eval_results['ovd/HM'] = hm

            eval_results['ovad/protocol_id'] = self.protocol['protocol_id']
            eval_results['ovad/protocol_hash'] = self.protocol_hash
            eval_results['ovad/eval_protocol'] = self.eval_protocol
            eval_results['ovad/AP_maxDets'] = self.ap_max_dets
            for name in self.diagnostic_classes:
                value = eval_results.get(f'{name}_precision', None)
                if value is not None:
                    eval_results[f'ovad/diagnostic/{name}'] = value
            if original_outfile_prefix is not None:
                self._write_sidecar(current_prefix, eval_results, logger)
            return eval_results
        finally:
            self.outfile_prefix = original_outfile_prefix
            if tmp_dir is not None:
                tmp_dir.cleanup()

    def _evaluate_bbox_subset(self, result_file: str, cat_ids: list[int],
                              name: str) -> dict[str, float]:
        predictions = load(result_file)
        gt_count = sum(
            len(self._coco_api.get_ann_ids(img_ids=[img_id], cat_ids=cat_ids))
            for img_id in self.img_ids)
        if len(predictions) == 0:
            value = 0.0 if gt_count > 0 else -1.0
            return {
                'AP': value,
                'AP50': value,
                'AP75': value,
                'APs': value,
                'APm': value,
                'APl': value,
                'AR@100': value,
                'AR@300': value,
                'AR@900': value,
            }
        coco_dt = self._coco_api.loadRes(predictions)
        if self.use_mp_eval:
            coco_eval = COCOevalMP(self._coco_api, coco_dt, 'bbox')
        else:
            coco_eval = COCOeval(self._coco_api, coco_dt, 'bbox')
        coco_eval.params.catIds = list(cat_ids)
        coco_eval.params.imgIds = self.img_ids
        coco_eval.params.maxDets = list(self.proposal_nums)
        coco_eval.params.iouThrs = self.iou_thrs
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return self._collect_stats(coco_eval, cat_ids, name)

    def _mean_precision(self,
                        coco_eval,
                        cat_index: Optional[int] = None,
                        area_label: str = 'all',
                        iou_thr: Optional[float] = None) -> float:
        params = coco_eval.params
        precision = coco_eval.eval['precision']
        if iou_thr is not None:
            iou_indices = np.where(np.isclose(params.iouThrs, iou_thr))[0]
            if iou_indices.size == 0:
                return -1.0
            precision = precision[iou_indices]
        if cat_index is not None:
            precision = precision[:, :, cat_index:cat_index + 1, :, :]
        area_indices = [
            idx for idx, label in enumerate(params.areaRngLbl)
            if label == area_label
        ]
        max_det_indices = [
            idx for idx, value in enumerate(params.maxDets)
            if value == self.ap_max_dets
        ]
        if not area_indices or not max_det_indices:
            return -1.0
        precision = precision[:, :, :, area_indices, max_det_indices]
        precision = precision[precision > -1]
        if precision.size == 0:
            return -1.0
        return _rounded_stat(float(np.mean(precision)))

    def _mean_recall(self,
                     coco_eval,
                     max_dets: int,
                     area_label: str = 'all') -> float:
        params = coco_eval.params
        recall = coco_eval.eval['recall']
        area_indices = [
            idx for idx, label in enumerate(params.areaRngLbl)
            if label == area_label
        ]
        max_det_indices = [
            idx for idx, value in enumerate(params.maxDets)
            if value == max_dets
        ]
        if not area_indices or not max_det_indices:
            return -1.0
        recall = recall[:, :, area_indices, max_det_indices]
        recall = recall[recall > -1]
        if recall.size == 0:
            return -1.0
        return _rounded_stat(float(np.mean(recall)))

    def _collect_stats(self, coco_eval, cat_ids: list[int],
                       split: str) -> dict:
        stats = {
            'AP': self._mean_precision(coco_eval),
            'AP50': self._mean_precision(coco_eval, iou_thr=0.5),
            'AP75': self._mean_precision(coco_eval, iou_thr=0.75),
            'APs': self._mean_precision(coco_eval, area_label='small'),
            'APm': self._mean_precision(coco_eval, area_label='medium'),
            'APl': self._mean_precision(coco_eval, area_label='large'),
        }
        for max_dets in self.proposal_nums:
            stats[f'AR@{max_dets}'] = self._mean_recall(coco_eval, max_dets)

        # pycocotools normalizes catIds with np.unique(), which sorts them.
        # The precision K dimension follows coco_eval.params.catIds after that
        # normalization, not the prompt order used to serialize predictions.
        eval_cat_ids = [int(cat_id) for cat_id in coco_eval.params.catIds]
        if set(eval_cat_ids) != {int(cat_id) for cat_id in cat_ids}:
            raise AssertionError(
                f'COCOeval catIds changed unexpectedly: {eval_cat_ids} vs '
                f'{cat_ids}')
        if len(eval_cat_ids) != coco_eval.eval['precision'].shape[2]:
            raise AssertionError(
                'COCOeval precision category dimension does not match catIds')
        prompt_label_by_cat_id = {
            int(cat_id): index
            for index, cat_id in enumerate(self.prompt_cat_ids)
        }
        per_category = []
        for idx, cat_id in enumerate(eval_cat_ids):
            name = self._coco_api.loadCats([cat_id])[0]['name']
            per_category.append(
                dict(
                    split=split,
                    eval_index=idx,
                    index=idx,
                    prompt_label=prompt_label_by_cat_id.get(cat_id, -1),
                    category_id=int(cat_id),
                    class_name=str(name),
                    AP=self._mean_precision(coco_eval, cat_index=idx),
                    AP50=self._mean_precision(
                        coco_eval, cat_index=idx, iou_thr=0.5),
                    AP75=self._mean_precision(
                        coco_eval, cat_index=idx, iou_thr=0.75),
                    APs=self._mean_precision(
                        coco_eval, cat_index=idx, area_label='small'),
                    APm=self._mean_precision(
                        coco_eval, cat_index=idx, area_label='medium'),
                    APl=self._mean_precision(
                        coco_eval, cat_index=idx, area_label='large')))
        stats['per_category'] = per_category
        return stats

    def _add_prefixed_stats(self, eval_results: dict, prefix: str, split: str,
                            stats: dict[str, float]) -> None:
        eval_results[f'{prefix}/AP_{split}'] = stats['AP']
        eval_results[f'{prefix}/AP@[.5:.95]_{split}'] = stats['AP']
        eval_results[f'{prefix}/AP50_{split}'] = stats['AP50']
        eval_results[f'{prefix}/AP75_{split}'] = stats['AP75']
        eval_results[f'{prefix}/APs_{split}'] = stats['APs']
        eval_results[f'{prefix}/APm_{split}'] = stats['APm']
        eval_results[f'{prefix}/APl_{split}'] = stats['APl']
        eval_results[f'{prefix}/Recall@100_{split}'] = stats['AR@100']
        eval_results[f'{prefix}/Recall@300_{split}'] = stats['AR@300']
        eval_results[f'{prefix}/Recall@900_{split}'] = stats['AR@900']

    def _override_parent_bbox_stats(self, eval_results: dict, stats: dict,
                                    logger: MMLogger) -> None:
        eval_results['bbox_mAP'] = stats['AP']
        eval_results['bbox_mAP_50'] = stats['AP50']
        eval_results['bbox_mAP_75'] = stats['AP75']
        eval_results['bbox_mAP_s'] = stats['APs']
        eval_results['bbox_mAP_m'] = stats['APm']
        eval_results['bbox_mAP_l'] = stats['APl']
        logger.info('OVAD bbox_mAP_copypaste maxDets=%s: %.3f %.3f %.3f '
                    '%.3f %.3f %.3f', self.ap_max_dets, stats['AP'],
                    stats['AP50'], stats['AP75'], stats['APs'],
                    stats['APm'], stats['APl'])

    def _add_classwise_stats(self, eval_results: dict, stats: dict,
                             logger: MMLogger) -> None:
        per_category = stats.get('per_category', [])
        if not per_category:
            return
        for item in per_category:
            eval_results[f'{item["class_name"]}_precision'] = item['AP']
            eval_results[f'{item["class_name"]}_precision_50'] = item['AP50']
            eval_results[f'{item["class_name"]}_precision_75'] = item['AP75']
        if not self.ovad_classwise:
            return
        table_data = [[
            'eval_idx', 'prompt_label', 'cat_id', 'category', 'mAP',
            'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
        ]]
        for item in per_category:
            table_data.append([
                item['eval_index'], item['prompt_label'],
                item['category_id'], item['class_name'], item['AP'],
                item['AP50'], item['AP75'], item['APs'], item['APm'],
                item['APl']
            ])
        logger.info('\nOVAD classwise AP maxDets=%s\n%s',
                    self.ap_max_dets, AsciiTable(table_data).table)

    def _category_order(self) -> list[dict]:
        return [
            dict(
                prompt_label=index,
                category_id=int(cat_id),
                class_name=str(class_name))
            for index, (cat_id, class_name) in enumerate(
                zip(self.prompt_cat_ids, self.eval_classes))
        ]

    def _assert_category_order(self) -> None:
        if list(self.cat_ids) != list(self.prompt_cat_ids):
            raise AssertionError(
                f'OVAD cat_ids must follow prompt order: {self.cat_ids} != '
                f'{self.prompt_cat_ids}')
        if self._coco_api is None:
            return
        for item in self._category_order():
            cats = self._coco_api.loadCats([item['category_id']])
            if not cats:
                raise AssertionError(
                    f'OVAD category_id missing from COCO API: {item}')
            actual = normalize_class_name(cats[0]['name'])
            expected = normalize_class_name(item['class_name'])
            if actual != expected:
                raise AssertionError(
                    f'OVAD category order mismatch at label '
                    f'{item["prompt_label"]}: category_id '
                    f'{item["category_id"]} is {cats[0]["name"]}, '
                    f'expected {item["class_name"]}')

    def _log_category_order(self, logger: MMLogger) -> None:
        order = ', '.join(
            f'{item["prompt_label"]}->{item["category_id"]}:'
            f'{item["class_name"]}' for item in self._category_order())
        logger.info('OVAD prompt label/category order: %s', order)

    def _write_sidecar(self, outfile_prefix: str, eval_results: dict,
                       logger: MMLogger) -> None:
        sidecar_path = outfile_prefix + '.ovad_protocol.json'
        with open(sidecar_path, 'w', encoding='utf-8') as handle:
            json.dump(
                dict(
                    protocol_id=self.protocol['protocol_id'],
                    protocol_hash=self.protocol_hash,
                    eval_protocol=self.eval_protocol,
                    ap_max_dets=self.ap_max_dets,
                    recall_max_dets=list(self.proposal_nums),
                    category_order=self._category_order(),
                    metrics={
                        key: value
                        for key, value in eval_results.items()
                        if key.startswith(('ovad/', 'ovd/'))
                    }),
                handle,
                indent=2)
        logger.info('OVAD protocol metadata written to %s',
                    osp.abspath(sidecar_path))
