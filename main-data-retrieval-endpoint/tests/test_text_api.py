"""Request-level regression through the REAL Factory/report, with a fake predictor loader."""
import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from xai_core.model_loader import ModelInfo
from tests.test_autogluon_tabular import FakePredictor
from tests.test_text_feature_mapping import mapping

api = importlib.import_module('api.app')


@pytest.fixture
def request_context(monkeypatch):
    predictor = FakePredictor()
    info = ModelInfo(predictor, 'tabular', 'classification', True, label='labels')
    monkeypatch.setattr(api, 'load_model_from_bytes', lambda *args: info)
    # Only unrelated expensive EDA plots are omitted, not report/model explanations.
    monkeypatch.setattr(api.DataInterpretabilityService, '_generate_visualizations', lambda self: {})
    def forbidden_legacy(*args, **kwargs):
        raise AssertionError('AutoGluon must never enter the legacy service')
    monkeypatch.setattr(api, 'ExplainerService', forbidden_legacy)
    return TestClient(api.app), predictor, info


def post(client, frame, data=None, endpoint='/explain-model'):
    csv = frame if isinstance(frame, bytes) else frame.to_csv(index=False).encode()
    return client.post(endpoint, files={
        'model_file': ('fake.zip', b'FAKE LOADER ONLY', 'application/zip'),
        'data_file': ('test.csv', csv, 'text/csv'),
    }, data=data or {})


@pytest.mark.parametrize('columns', [['labels', 'text'], ['text', 'labels']])
@pytest.mark.parametrize('explicit', [True, False])
def test_labels_text_factory_report_regression(request_context, columns, explicit):
    client, predictor, info = request_context
    info.text_feature_mapping = mapping()
    frame = pd.DataFrame({'labels': [1, 2] + [3] * 38,
                          'text': [f'risk review for evaluation example number {i}' for i in range(40)]})[columns]
    response = post(client, frame, {'target_col': 'labels'} if explicit else None)
    assert response.status_code == 200, response.text
    for expected in ['autogluon_tabular', 'Dataset Overview', 'Text Data Profile',
                     'Token-Removal Sensitivity', 'Macro F1', 'Balanced Accuracy',
                     'Per-Class Performance', 'Confusion Matrix', '95.0%',
                     'Class Distribution and Majority Baseline', 'Text Preprocessing Context',
                     '40 evaluation rows', 'CountVectorizer', 'not SHAP']:
        assert expected in response.text
    assert 'excellent performance' not in response.text
    assert 'trained on' not in response.text
    assert all(list(call.columns) == ['text'] for call in predictor.calls + predictor.prediction_calls)
    assert predictor.prediction_calls[0]['text'].tolist() == frame['text'].tolist()
    assert '<img src=x onerror=' not in response.text


@pytest.mark.parametrize('frame,data,error', [
    (b'labels,other\n1,hi\n3,bye', {}, 'text'),
    (b'text,other\nhi,1\nbye,3', {}, 'labels'),
    (b'labels,text\n1,hi', {'target_col': 'text'}, 'conflicts'),
    (b'labels,text\n1,hi', {'target_col': 'missing'}, 'conflicts'),
    (b'labels,text,text\n1,hi,bye', {}, 'Duplicate'),
    (b'labels,text\n,hi', {}, 'empty'),
    (b'labels,text\n', {}, 'empty'),
    (b'labels,text\n9,hi', {}, 'class_labels'),
])
def test_invalid_schema_returns_clear_400(request_context, frame, data, error):
    client, predictor, _ = request_context
    response = post(client, frame, data)
    assert response.status_code == 400, response.text
    assert error in response.json()['detail']
    assert not predictor.calls
    assert not predictor.prediction_calls


@pytest.mark.parametrize('method', ['predict', 'predict_proba'])
def test_prediction_dependency_failure_is_not_legacy_success(request_context, method):
    client, predictor, _ = request_context
    calls = []
    def fail(X, **kwargs):
        calls.append(kwargs)
        raise ModuleNotFoundError("No module named 'required_backend'")
    setattr(predictor, method, fail)
    response = post(client, b'labels,text\n3,risk\n1,review')
    assert response.status_code == 500
    assert 'required_backend' in response.json()['detail']
    assert calls == [{}]
    assert 'html' not in response.headers['content-type']


def test_missing_mapping_is_optional_and_info_exposes_metadata(request_context):
    client, _, info = request_context
    response = post(client, b'labels,text\n3,risk\n1,review')
    assert response.status_code == 200
    assert 'No text_feature_mapping.json supplied' in response.text
    info.text_feature_mapping = mapping()
    response = post(client, b'labels,text\n3,risk\n1,review', endpoint='/explain-model/info')
    assert response.status_code == 200, response.text
    assert response.json()['label'] == 'labels'
    assert response.json()['text_feature_mapping'] == mapping()
    assert response.json()['metrics']['per_class']['2']['support'] == 0


def test_malformed_mapping_is_400(request_context):
    client, _, info = request_context
    info.text_feature_mapping = {'ngram_features': []}
    response = post(client, b'labels,text\n3,risk')
    assert response.status_code == 400
    assert 'text_feature_mapping.json' in response.json()['detail']


def test_model_label_resolves_ambiguous_csv(request_context):
    client, predictor, _ = request_context
    response = post(client, b'labels,target,text\n3,0,risk\n1,0,review')
    assert response.status_code == 200, response.text
    assert predictor.prediction_calls[0].columns.tolist() == ['text']


def test_unrecognized_and_ambiguous_target_without_model_label(request_context):
    client, predictor, info = request_context
    predictor.label = None
    info.label = None
    for data in [b'answer,text\n3,risk', b'labels,target,text\n3,3,risk']:
        response = post(client, data)
        assert response.status_code == 400
        assert 'ambiguous or unrecognized' in response.json()['detail']


def test_info_explicit_custom_target_and_missing_input(request_context):
    client, predictor, info = request_context
    predictor.label = None
    info.label = None
    response = post(client, b'answer,text\n3,risk\n1,review', {'target_col': 'answer'}, '/explain-model/info')
    assert response.status_code == 200
    assert response.json()['label'] == 'answer'
    response = post(client, b'answer,other\n3,risk', {'target_col': 'answer'}, '/explain-model/info')
    assert response.status_code == 400
    assert 'text' in response.json()['detail']


def test_raw_column_names_and_labels_are_escaped_in_combined_report(request_context):
    client, _, info = request_context
    info.model = FakePredictor(('<feature>',))
    info.model.class_labels = ['<label>', 'a', 'b']
    response = post(client, pd.DataFrame({'labels': ['<label>', 'a', 'b'],
                                        '<feature>': ['risk <script>alert(1)</script>'] * 3}))
    assert response.status_code == 200, response.text
    assert '<feature>' not in response.text
    assert '<label>' not in response.text
    assert '<script>' not in response.text
    assert '&lt;feature&gt;' in response.text
    assert '&lt;label&gt;' in response.text


def test_existing_numeric_sklearn_request_still_uses_factory(request_context):
    from sklearn.ensemble import RandomForestClassifier
    client, _, info = request_context
    frame = pd.DataFrame({'number': [1., 2., 3., 4.], 'labels': [0, 0, 1, 1]})
    info.model = RandomForestClassifier(n_estimators=2, random_state=42).fit(frame[['number']], frame['labels'])
    info.model_type = 'tree_ensemble'
    info.is_autogluon = False
    info.label = None
    response = post(client, frame)
    assert response.status_code == 200, response.text
    assert 'Feature Importance' in response.text
    assert 'Accuracy' in response.text
    assert '4 evaluation rows' in response.text


def test_synthetic_sklearn_text_pipeline_autodetected_despite_generic_loader_type(request_context):
    from tests.test_sklearn_text_explainer import _text_pipeline
    client, _, info = request_context
    frame = pd.DataFrame({'labels': [1, 1, 2, 2], 'text': [
        'critical account risk was detected during the international transfer review',
        'critical payment risk was detected during the customer transaction review',
        'routine payment was approved after completing the standard customer review',
        'routine account was approved after completing the standard compliance review',
    ]})
    info.model = _text_pipeline().fit(frame[['text']], frame['labels'])
    info.model_type = 'generic'  # The API must not let a generic loader label override graph detection.
    info.is_autogluon = False
    info.label = None
    response = post(client, frame, {'target_col': 'labels'})
    assert response.status_code == 200, response.text
    # request_context forbids legacy service construction; these sections require SklearnTextExplainer.
    assert 'sklearn_text' in response.text
    assert 'SHAP Text Explanations' in response.text
    assert 'LIME Text Explanations' in response.text
    assert 'token-positive' in response.text
