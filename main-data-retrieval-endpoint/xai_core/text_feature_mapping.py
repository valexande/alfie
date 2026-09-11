"""Bounded, passive JSON sidecar parsing. Mappings describe vocabulary, not importance."""

import json
from pathlib import Path

from xai_core.utils import InputValidationError

MAX_MAPPING_BYTES = 2 * 1024 * 1024
MAX_TERMS = 50000
MAX_GROUPS = 32
MAX_NAME_LENGTH = 2048


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise InputValidationError(f'Duplicate text mapping key: {key[:80]}')
        obj[key] = value
    return obj


def _name(value):
    return isinstance(value, str) and 0 < len(value) <= MAX_NAME_LENGTH


def validate_text_feature_mapping(mapping: dict) -> dict:
    """Validate term order against vocabulary indices, without interpreting any paths."""
    def invalid(message):
        raise InputValidationError(f'Invalid text_feature_mapping.json: {message}')

    if not isinstance(mapping, dict) or set(mapping) != {'ngram_features', 'multimodal_text_model_dirs'}:
        invalid('expected ngram_features and multimodal_text_model_dirs')
    dirs = mapping['multimodal_text_model_dirs']
    if not isinstance(dirs, list) or len(dirs) > MAX_GROUPS or not all(_name(v) for v in dirs):
        invalid('invalid descriptive multimodal directory list')
    groups = mapping['ngram_features']
    if not isinstance(groups, dict) or len(groups) > MAX_GROUPS:
        invalid('invalid ngram feature groups')
    total = 0
    for group, entry in groups.items():
        if not _name(group) or not isinstance(entry, dict) or set(entry) != {'feature_names', 'vocabulary'}:
            invalid('each ngram group needs feature_names and vocabulary')
        names, vocabulary = entry['feature_names'], entry['vocabulary']
        if not isinstance(names, list) or not isinstance(vocabulary, dict):
            invalid('feature_names must be a list and vocabulary an object')
        total += len(names)
        if total > MAX_TERMS or not all(_name(n) for n in names) or len(names) != len(set(names)):
            invalid('oversized, invalid or duplicate feature names')
        if len(vocabulary) != len(names):
            invalid('vocabulary and feature_names lengths differ')
        for index, name in enumerate(names):
            if type(vocabulary.get(name)) is not int or vocabulary[name] != index:
                invalid('vocabulary indices must match feature_names order exactly')
    return mapping


def parse_text_feature_mapping(data: bytes) -> dict:
    if len(data) > MAX_MAPPING_BYTES:
        raise InputValidationError('text_feature_mapping.json exceeds 2 MiB limit')
    try:
        mapping = json.loads(data.decode('utf-8'), object_pairs_hook=_unique_object)
        return validate_text_feature_mapping(mapping)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InputValidationError(f'Invalid text_feature_mapping.json: {exc}') from exc


def load_text_feature_mapping(predictor_root: Path):
    """Only the adjacent sidecar belongs to the selected root; never search elsewhere."""
    path = predictor_root / 'text_feature_mapping.json'
    if path.is_symlink():
        raise InputValidationError('text_feature_mapping.json must not be a symbolic link')
    if not path.exists():
        return None
    if not path.is_file():
        raise InputValidationError('text_feature_mapping.json must be a file')
    with path.open('rb') as stream:
        return parse_text_feature_mapping(stream.read(MAX_MAPPING_BYTES + 1))
