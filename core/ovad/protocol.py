"""OVAD protocol loading and validation utilities."""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


class OVADProtocolError(AssertionError):
    """Raised when an OVAD protocol or artifact violates the contract."""


def normalize_class_name(name: str) -> str:
    text = re.sub(r'[-_]+', ' ', str(name).strip().lower())
    return re.sub(r'\s+', ' ', text)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OVADProtocolError(message)


def protocol_file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict):
        raise OVADProtocolError(f'Protocol is not a mapping: {path}')
    protocol = deepcopy(protocol)
    protocol['_path'] = str(path)
    protocol['_sha256'] = protocol_file_sha256(path)
    assert_protocol(protocol)
    return protocol


def class_entries(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return list(protocol.get('classes', []))


def class_names_for_scope(protocol: dict[str, Any],
                          scope: str,
                          field: str = 'canonical') -> tuple[str, ...]:
    if scope not in ('base', 'novel', 'all'):
        raise OVADProtocolError(f'Unknown class scope: {scope}')
    order = protocol['prompt_orders'][scope]
    by_name = {entry['canonical']: entry for entry in class_entries(protocol)}
    return tuple(str(by_name[name][field]) for name in order)


def class_ids_for_scope(protocol: dict[str, Any], scope: str) -> tuple[int, ...]:
    if scope not in ('base', 'novel', 'all'):
        raise OVADProtocolError(f'Unknown class scope: {scope}')
    order = protocol['prompt_orders'][scope]
    by_name = {entry['canonical']: entry for entry in class_entries(protocol)}
    return tuple(int(by_name[name]['dataset_id']) for name in order)


def build_prompt_texts(protocol: dict[str, Any], scope: str) -> tuple[str, ...]:
    return class_names_for_scope(protocol, scope, field='prompt_text')


def canonical_by_dataset_id(protocol: dict[str, Any]) -> dict[int, str]:
    return {
        int(entry['dataset_id']): str(entry['canonical'])
        for entry in class_entries(protocol)
    }


def _names(values: Iterable[str]) -> set[str]:
    return {normalize_class_name(value) for value in values}


def assert_protocol(protocol: dict[str, Any]) -> None:
    entries = class_entries(protocol)
    _require(entries, 'protocol must contain at least one class')

    canonical = [entry['canonical'] for entry in entries]
    dataset_ids = [int(entry['dataset_id']) for entry in entries]
    continuous_ids = [int(entry['continuous_id']) for entry in entries]
    class_count = len(entries)
    _require(len(set(canonical)) == class_count,
             'canonical names are not unique')
    _require(len(set(dataset_ids)) == class_count,
             'dataset ids are not unique')
    _require(sorted(continuous_ids) == list(range(class_count)),
             f'continuous ids must be exactly 0..{class_count - 1}')

    split_by_name = {entry['canonical']: entry['split'] for entry in entries}
    base = [name for name, split in split_by_name.items() if split == 'base']
    novel = [name for name, split in split_by_name.items() if split == 'novel']
    _require(base, 'protocol must contain base classes')
    protocol_id = str(protocol.get('protocol_id', ''))
    dataset_name = str(protocol.get('dataset', ''))
    if 'expected_novel_count' in protocol:
        expected_novel = int(protocol['expected_novel_count'])
    else:
        expected_novel = 2 if (
            dataset_name == 'DOTA-v1.0' or protocol_id.startswith('dota1_')
        ) else 4
    _require(len(novel) == expected_novel,
             f'protocol must contain exactly {expected_novel} novel classes')
    _require(not (_names(base) & _names(novel)),
             'base and novel classes overlap')
    _require(_names(base) | _names(novel) == _names(canonical),
             'base + novel does not cover all classes')

    if protocol.get('dataset') == 'DOTA-v1.5' or protocol_id.startswith(
            'dota15_'):
        expected_base = [
            'large-vehicle', 'small-vehicle', 'plane', 'ship', 'harbor',
            'roundabout', 'bridge', 'baseball-diamond',
            'soccer-ball-field', 'storage-tank', 'tennis-court',
            'container-crane'
        ]
        expected_novel = [
            'helicopter', 'basketball-court', 'ground-track-field',
            'swimming-pool'
        ]
        _require(protocol['prompt_orders']['base'] == expected_base,
                 'base prompt order does not match ITC-OVAD DOTA-v1.5')
        _require(protocol['prompt_orders']['novel'] == expected_novel,
                 'novel prompt order does not match ITC-OVAD DOTA-v1.5')

    if protocol.get('dataset') == 'DOTA-v1.0' or protocol_id.startswith(
            'dota1_'):
        split_variant = str(protocol.get('split_variant', '13/2'))
        if split_variant == '11/4':
            expected_base = [
                'plane', 'baseball-diamond', 'bridge', 'small-vehicle',
                'large-vehicle', 'ship', 'tennis-court', 'storage-tank',
                'soccer-ball-field', 'roundabout', 'harbor'
            ]
            expected_novel = [
                'ground-track-field', 'basketball-court', 'swimming-pool',
                'helicopter'
            ]
        elif split_variant in (
                'DescReg/VK-Det 11/4', 'descreg_vkdet_11_4'):
            expected_base = [
                'plane', 'ship', 'storage-tank', 'baseball-diamond',
                'basketball-court', 'ground-track-field', 'harbor', 'bridge',
                'large-vehicle', 'small-vehicle', 'roundabout'
            ]
            expected_novel = [
                'tennis-court', 'helicopter', 'soccer-ball-field',
                'swimming-pool'
            ]
        else:
            expected_base = [
                'plane', 'baseball-diamond', 'bridge', 'small-vehicle',
                'large-vehicle', 'ship', 'tennis-court', 'storage-tank',
                'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
                'helicopter'
            ]
            expected_novel = ['ground-track-field', 'basketball-court']
        _require(protocol['prompt_orders']['base'] == expected_base,
                 f'base prompt order does not match DOTA-v1.0 '
                 f'{split_variant}')
        _require(protocol['prompt_orders']['novel'] == expected_novel,
                 f'novel prompt order does not match DOTA-v1.0 '
                 f'{split_variant}')

    _require(
        protocol['prompt_orders']['all'] == protocol['prompt_orders']['base'] +
        protocol['prompt_orders']['novel'],
        'GZSD prompt order must be base order followed by novel order')

    for scope in ('base', 'novel', 'all'):
        _require(_names(protocol['prompt_orders'][scope]) <= _names(canonical),
                 f'unknown class in prompt order: {scope}')

    strict = protocol.get('strict_inductive', {})
    _require(strict.get('primary') is True,
             'strict_inductive must be marked as primary')
    _require(strict.get('allow_novel_names_in_train_prompt') is False,
             'strict inductive training cannot expose novel names')
    _require(strict.get('allow_novel_gt_boxes_in_train') is False,
             'strict inductive training cannot use novel boxes')
    _require(strict.get('use_novel_boxes_as_ignore') is False,
             'strict inductive training cannot use novel boxes as ignore')


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def assert_annotation_policy(annotation_path: str | os.PathLike[str],
                             protocol: dict[str, Any],
                             annotation_scope: str,
                             forbid_novel: bool = False) -> dict[str, int]:
    data = _load_json(annotation_path)
    canonical = canonical_by_dataset_id(protocol)
    allowed = set(class_names_for_scope(protocol, annotation_scope))
    novel = set(class_names_for_scope(protocol, 'novel'))

    categories = data.get('categories', [])
    category_ids = {int(cat['id']) for cat in categories}
    category_names = {str(cat['name']) for cat in categories}
    allowed_ids = {
        int(entry['dataset_id'])
        for entry in class_entries(protocol)
        if entry['canonical'] in allowed
    }
    novel_ids = {
        int(entry['dataset_id'])
        for entry in class_entries(protocol)
        if entry['canonical'] in novel
    }

    _require(category_ids <= allowed_ids,
             f'{annotation_path} contains categories outside {annotation_scope}')
    _require(_names(category_names) == _names(allowed),
             f'{annotation_path} category names do not match {annotation_scope}')

    leaked_novel = 0
    out_of_scope = 0
    for ann in data.get('annotations', []):
        cat_id = int(ann['category_id'])
        if cat_id not in allowed_ids:
            out_of_scope += 1
        if cat_id in novel_ids:
            leaked_novel += 1
        if cat_id not in canonical:
            out_of_scope += 1

    _require(out_of_scope == 0,
             f'{annotation_path} has {out_of_scope} out-of-scope annotations')
    if forbid_novel:
        _require(leaked_novel == 0,
                 f'{annotation_path} has {leaked_novel} novel annotations')

    positive_map_novel = count_novel_strings(data, protocol)
    if forbid_novel:
        _require(positive_map_novel == 0,
                 f'{annotation_path} leaks novel strings in text/positive maps')

    return dict(
        images=len(data.get('images', [])),
        annotations=len(data.get('annotations', [])),
        categories=len(categories),
        novel_annotations=leaked_novel,
        novel_string_hits=positive_map_novel)


def count_novel_strings(data: Any, protocol: dict[str, Any]) -> int:
    novel_terms: set[str] = set()
    by_name = {entry['canonical']: entry for entry in class_entries(protocol)}
    for name in protocol['prompt_orders']['novel']:
        entry = by_name[name]
        novel_terms.add(normalize_class_name(entry['canonical']))
        novel_terms.add(normalize_class_name(entry['prompt_text']))
        for alias in entry.get('aliases', []):
            novel_terms.add(normalize_class_name(alias))

    def walk(value: Any) -> int:
        if isinstance(value, dict):
            return sum(walk(item) for item in value.values())
        if isinstance(value, list):
            if not any(isinstance(item, (dict, list, str)) for item in value):
                return 0
            return sum(walk(item) for item in value)
        if isinstance(value, str):
            text = normalize_class_name(value)
            return int(any(term and term in text for term in novel_terms))
        return 0

    return walk(data)


def resolve_protocol_path(path: str | os.PathLike[str]) -> str:
    path_obj = Path(path)
    return str(path_obj if path_obj.is_absolute() else path_obj.resolve())
