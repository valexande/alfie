"""Passive sidecar/root tests only; synthetic predictor marker files are never loaded."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest

from xai_core import model_loader
from xai_core.text_feature_mapping import (
    MAX_MAPPING_BYTES, load_text_feature_mapping, parse_text_feature_mapping,
)
from xai_core.utils import InputValidationError
from xai_core.explainers.autogluon_tabular import AutoGluonTabularExplainer
from xai_core.report_builder import ReportBuilder
from tests.test_autogluon_tabular import FakePredictor


def mapping():
    names = ['<img src=x onerror=alert(1)>', 'risk', 'risk level', 'high risk level']
    return {'multimodal_text_model_dirs': [],
            'ngram_features': {'__nlp__': {'feature_names': names, 'vocabulary': dict(zip(names, range(len(names))))}}}


def test_valid_mapping_and_html_escaping():
    value = parse_text_feature_mapping(json.dumps(mapping()).encode())
    model = FakePredictor(('<feature>',))
    model.class_labels = ['<label>', 'a', 'b']
    explainer = AutoGluonTabularExplainer(model, pd.DataFrame({'<feature>': ['<script> risk']}),
                                       pd.Series(['<label>']), text_feature_mapping=value)
    report = ReportBuilder(explainer).build('expert')
    for tag in ['<script>', '<feature>', '<label>', '<img src=x']:
        assert tag not in report
    for escaped in ['&lt;feature&gt;', '&lt;label&gt;', '&lt;img src=x']:
        assert escaped in report
    assert '4 vocabulary terms' in report
    assert 'NOT feature importance' in report
    assert 'CountVectorizer' in report
    assert 'Multimodal text model directories listed in the sidecar: 0' in report


@pytest.mark.parametrize('change', ['order', 'duplicate', 'index', 'negative', 'boolean', 'missing', 'shape', 'dirs'])
def test_invalid_mapping_shape_order_indices(change):
    value = mapping()
    entry = value['ngram_features']['__nlp__']
    if change == 'order':
        entry['feature_names'].reverse()
    elif change == 'duplicate':
        entry['feature_names'][1] = entry['feature_names'][0]
    elif change == 'index':
        entry['vocabulary']['risk'] = 4000
    elif change == 'negative':
        entry['vocabulary']['risk'] = -1
    elif change == 'boolean':
        entry['vocabulary']['risk'] = True
    elif change == 'missing':
        del entry['vocabulary']
    elif change == 'shape':
        value['ngram_features'] = []
    else:
        value['multimodal_text_model_dirs'] = '../outside'
    with pytest.raises(InputValidationError):
        parse_text_feature_mapping(json.dumps(value).encode())


@pytest.mark.parametrize('data', [b'{' + b' ' * MAX_MAPPING_BYTES, b'{broken', b'[]',
                                b'{"ngram_features":{},"ngram_features":{},"multimodal_text_model_dirs":[]}',
                                b'[' * 2000 + b']' * 2000])
def test_mapping_bounds_and_duplicate_json_keys(data):
    with pytest.raises(InputValidationError):
        parse_text_feature_mapping(data)


def marker(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'predictor.pkl').write_bytes(b'NOT A PICKLE - do not deserialize')
    return root


def test_unique_shallowest_root_wrapper_and_clone_mapping_association(tmp_path):
    root = marker(tmp_path / 'wrapper' / 'predictor')
    clone = marker(root / '-clone-opt')
    (clone / 'text_feature_mapping.json').write_text('invalid ignored clone mapping')
    (tmp_path / 'text_feature_mapping.json').write_text('invalid ignored parent mapping')
    assert model_loader._select_predictor_root(tmp_path) == root
    assert load_text_feature_mapping(root) is None
    (root / 'text_feature_mapping.json').write_text(json.dumps(mapping()))
    assert load_text_feature_mapping(root) == mapping()


def test_same_depth_roots_rejected(tmp_path):
    marker(tmp_path / 'one')
    marker(tmp_path / 'two')
    with pytest.raises(InputValidationError, match='multiple predictor roots'):
        model_loader._select_predictor_root(tmp_path)


def test_zip_root_selection_and_non_autogluon_zip_unchanged(tmp_path, monkeypatch):
    extraction = tmp_path / 'extracted'
    extraction.mkdir()
    monkeypatch.setattr(model_loader.tempfile, 'mkdtemp', lambda **kw: str(extraction))
    archive = tmp_path / 'model.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('wrapper/root/predictor.pkl', b'not deserialized')
        zf.writestr('wrapper/root/-clone-opt/predictor.pkl', b'not deserialized')
    assert model_loader._extract_zip(archive) == extraction / 'wrapper' / 'root'
    empty_extract = tmp_path / 'vision-extracted'
    empty_extract.mkdir()
    monkeypatch.setattr(model_loader.tempfile, 'mkdtemp', lambda **kw: str(empty_extract))
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('model.pt', b'not deserialized')
        zf.writestr('labels.json', '{}')
    assert model_loader._extract_zip(archive) == empty_extract


def test_mapping_symlink_rejected(tmp_path):
    outside = tmp_path / 'outside.json'
    outside.write_text(json.dumps(mapping()))
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'text_feature_mapping.json').symlink_to(outside)
    with pytest.raises(InputValidationError, match='symbolic link'):
        load_text_feature_mapping(root)


def test_loader_propagates_only_adjacent_mapping_and_label_with_fake_load(tmp_path, monkeypatch):
    root = marker(tmp_path / 'root')
    (root / 'text_feature_mapping.json').write_text(json.dumps(mapping()))
    model = FakePredictor()
    class FakeTabularClass:
        @staticmethod
        def load(*args, **kwargs):
            return model  # No serialization/deserialization involved.
    monkeypatch.setattr(model_loader, 'TabularPredictor', FakeTabularClass)
    monkeypatch.setattr(model_loader, 'MultiModalPredictor', None)
    monkeypatch.setattr(model_loader, 'TimeSeriesPredictor', None)
    monkeypatch.setattr(model_loader, 'create_adapter', lambda obj: SimpleNamespace(problem_type='classification'))
    monkeypatch.setattr(model_loader, '_test_predictor_compatibility', lambda *args: True)
    result = model_loader._try_autogluon_load(root, [])
    assert result.model is model
    assert result.label == 'labels'
    assert result.text_feature_mapping == mapping()


@pytest.mark.parametrize('filename', ['../../escaped.pkl', '/tmp/escaped.pkl', r'..\..\escaped.pkl'])
def test_upload_filename_is_not_used_as_path(tmp_path, monkeypatch, filename):
    storage = tmp_path / 'storage'
    storage.mkdir()
    monkeypatch.setattr(model_loader.tempfile, 'mkdtemp', lambda **kw: str(storage))
    seen = []
    def fake_load(path, errors):
        seen.append(path)
        assert path.read_bytes() == b'synthetic bytes'
        return model_loader.ModelInfo(None, 'sklearn_unknown', 'regression', False)
    monkeypatch.setattr(model_loader, '_try_pickle_load', fake_load)
    model_loader.load_model_from_bytes(b'synthetic bytes', filename)
    assert seen == [storage / 'upload.pkl']
    assert not (tmp_path / 'escaped.pkl').exists()


def test_duplicate_archive_mapping_entries_rejected_before_extraction(tmp_path, monkeypatch):
    archive = tmp_path / 'duplicate.zip'
    extraction = tmp_path / 'extract'
    extraction.mkdir()
    monkeypatch.setattr(model_loader.tempfile, 'mkdtemp', lambda **kw: str(extraction))
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('root/text_feature_mapping.json', '{}')
        zf.writestr('root/./text_feature_mapping.json', '{}')
    with pytest.raises(InputValidationError, match='Duplicate text_feature_mapping'):
        model_loader._extract_zip(archive)
    assert not extraction.exists()
