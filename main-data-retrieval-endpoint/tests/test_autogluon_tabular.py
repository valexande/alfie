"""Fake-predictor tests: no AutoGluon backend or archive deserialization."""
from types import SimpleNamespace
import re

import numpy as np
import pandas as pd
import pytest

from xai_core.autogluon_input import align_autogluon_features
from xai_core.explainers.autogluon_tabular import AutoGluonTabularExplainer
from xai_core.explainer_factory import ExplainerFactory
from xai_core.report_builder import ReportBuilder
from xai_core.utils import InputValidationError


class FakePredictor:
    label = 'labels'
    problem_type = 'multiclass'
    class_labels = [3, 1, 2]  # Deliberately not sorted or derived from evaluation y.
    eval_metric = SimpleNamespace(name='accuracy')

    def __init__(self, columns=('text',), text=True):
        self.feature_metadata_in = SimpleNamespace(
            type_map_raw={c: 'object' if text else 'float' for c in columns},
            type_group_map_special={'text': [columns[0]]} if text else {},
        )
        self.feature_metadata = SimpleNamespace(type_map_raw={'__nlp__.risk': 'int'})
        self.calls = []
        self.prediction_calls = []
        self.importance_calls = []

    def predict(self, X):
        self.prediction_calls.append(X.copy())
        assert list(X.columns) == list(self.feature_metadata_in.type_map_raw)
        return pd.Series([self.class_labels[0]] * len(X), index=X.index)

    def predict_proba(self, X):
        self.calls.append(X.copy())
        assert list(X.columns) == list(self.feature_metadata_in.type_map_raw)
        # Change only when the selected word is actually removed.
        first = X.iloc[:, 0].astype(str).map(lambda s: 0.8 if 'risk' in s else 0.6).to_numpy()
        return pd.DataFrame({self.class_labels[1]: (1 - first) / 2,
                             self.class_labels[2]: (1 - first) / 2,
                             self.class_labels[0]: first}, index=X.index)

    def feature_importance(self, **kwargs):
        self.importance_calls.append(kwargs)
        return pd.DataFrame({'importance': [0.25] * len(self.feature_metadata_in.type_map_raw)},
                            index=list(self.feature_metadata_in.type_map_raw))


def test_raw_metadata_alignment_never_uses_transformed_ngrams():
    model = FakePredictor(('text', 'amount'))
    frame = pd.DataFrame({'amount': [4], 'extra': [9], 'text': ['risk!']})
    aligned = align_autogluon_features(model, frame)
    assert list(aligned.columns) == ['text', 'amount']
    assert aligned.iloc[0].tolist() == ['risk!', 4]
    assert list(frame.columns) == ['amount', 'extra', 'text']
    assert AutoGluonTabularExplainer._filter_to_model_features(model, frame).equals(aligned)
    with pytest.raises(InputValidationError, match='text'):
        align_autogluon_features(model, frame.drop(columns='text'))
    model.feature_metadata_in.type_map_raw = {'gender_Female': 'int', 'gender_Male': 'int'}
    with pytest.raises(InputValidationError, match='gender_Female'):
        align_autogluon_features(model, pd.DataFrame({'gender': ['Female']}))


def test_public_original_features_fallback_and_no_transformed_fallback():
    class PublicPredictor:
        def features(self, *, feature_stage):
            assert feature_stage == 'original'
            return ['text']
    frame = pd.DataFrame({'text': ['hi']})
    assert align_autogluon_features(PublicPredictor(), frame).equals(frame)
    with pytest.raises(RuntimeError, match='original'):
        align_autogluon_features(SimpleNamespace(feature_metadata={'text': 'object'}), frame)


@pytest.mark.parametrize('text', ['  risk,\tword!\nSuffix?!',
                                 ' risk,\t' + ' '.join(f'word{i}' for i in range(90)) + '\nSUFFIX <safe>!'])
def test_word_removal_preserves_full_suffix_whitespace_punctuation_other_columns(text):
    model = FakePredictor(('text', 'amount'))
    X = pd.DataFrame({'text': [text, 'risk other'], 'amount': [17, 99]}, index=[7, 7])
    y = pd.Series([1, 3], name='labels', index=[7, 7])
    explainer = AutoGluonTabularExplainer(model, X, y)
    card = explainer._build_text_explanation_card(0, 'text', 3)
    assert 'Evaluation row position: 0' in card
    assert 'True label: 1' in card
    assert 'Highest-probability class: 3' in card
    assert '+0.2000' in card  # Actual full-model probability delta, not fabricated scores.
    assert model.calls[0].iloc[0]['text'] == text
    spans = list(re.finditer(r'\w+', text))[:3]
    perturbations = pd.concat(model.calls[1:], ignore_index=True)
    assert len(perturbations) == len(spans)
    for pos, span in enumerate(spans):
        assert perturbations.iloc[pos]['text'] == text[:span.start()] + text[span.end():]
        assert perturbations.iloc[pos]['amount'] == 17
    assert explainer._select_text_rows('text', 6) == [1, 0]
    assert X.iloc[0]['text'] == text


def test_hard_bounds_class_aware_examples_and_batch_size():
    model = FakePredictor()
    X = pd.DataFrame({'text': ['risk ' + 'word ' * 100] * 20}, index=[0] * 20)
    y = pd.Series([1, 2] + [3] * 18, name='labels', index=[0] * 20)
    explainer = AutoGluonTabularExplainer(model, X, y)
    result = explainer.get_text_explanations_html(max_examples=100, max_tokens=1000)
    assert result.count('class="text-card"') == 6
    assert [len(frame) for frame in model.calls] == [1, 32, 28] * 6
    assert explainer._select_text_rows('text', 6) == [2, 0, 1, 3, 4, 5]


def test_model_class_order_includes_missing_evaluation_class():
    model = FakePredictor()
    explainer = AutoGluonTabularExplainer(model, pd.DataFrame({'text': ['risk', 'word']}),
                                       pd.Series([3, 1], name='labels'))
    probabilities = explainer.get_prediction_probabilities()
    assert list(explainer.classes) == [3, 1, 2]
    np.testing.assert_allclose(probabilities[0], [0.8, 0.1, 0.1])
    metrics = explainer.get_metrics()
    assert metrics['class_labels'] == ['3', '1', '2']
    assert metrics['per_class']['2']['support'] == 0
    assert metrics['confusion_matrix'] == [[1, 0, 0], [1, 0, 0], [0, 0, 0]]
    model.predict_proba = lambda X: probabilities[:len(X)]
    np.testing.assert_allclose(explainer.get_prediction_probabilities(), probabilities)


@pytest.mark.parametrize('bad', [pd.DataFrame({3: [0.8], 1: [0.2]}),
                               np.array([[np.nan, 0.1, 0.1]]), np.array([[0.8, 0.1]]),
                               np.array([[0.8, 0.5, 0.1]])])
def test_probability_contract_failures_are_not_zero_contributions(bad):
    model = FakePredictor()
    model.predict_proba = lambda X: bad
    explainer = AutoGluonTabularExplainer(model, pd.DataFrame({'text': ['risk']}), pd.Series([3]))
    with pytest.raises(RuntimeError):
        explainer.get_metrics()
    assert explainer.get_text_explanations_html() is None
    assert any('failed' in note for note in explainer.explanation_notes)


def test_dependency_errors_never_switch_ensemble_members():
    model = FakePredictor()
    calls = []
    def fail(X, **kwargs):
        calls.append(kwargs)
        raise ModuleNotFoundError("No module named 'fasttransform'")
    model.predict = fail
    model.predict_proba = fail
    model.model_names = lambda: ['OtherModel']
    explainer = AutoGluonTabularExplainer(model, pd.DataFrame({'text': ['risk']}), pd.Series([3]))
    with pytest.raises(ModuleNotFoundError):
        explainer.get_metrics()
    with pytest.raises(ModuleNotFoundError):
        explainer.get_prediction_probabilities()
    assert calls == [{}, {}]


def test_imbalance_metrics_report_no_numeric_shap_or_raw_pca(monkeypatch):
    import xai_core.visualizations as plots
    def forbidden(*args, **kwargs):
        raise AssertionError('Raw text must not use generic numeric SHAP/PCA')
    monkeypatch.setattr(plots, 'plot_pca_analysis', forbidden)
    monkeypatch.setattr(plots, 'plot_shap_summary', forbidden)
    model = FakePredictor()
    X = pd.DataFrame({'text': ['risk <script>alert(1)</script>'] * 100})
    y = pd.Series([1, 1, 2, 2, 2] + [3] * 95, name='labels')
    explainer = ExplainerFactory.create(model, X, y, model_type='tabular', label='labels')
    metrics = explainer.get_metrics()
    assert metrics['accuracy'] == metrics['majority_baseline'] == .95
    assert metrics['balanced_accuracy'] == pytest.approx(1/3)
    assert metrics['macro_f1'] == pytest.approx((190/195)/3)
    for mode in ('expert', 'beginner'):
        report = ReportBuilder(explainer).build(mode)
        assert 'Per-Class Performance' in report
        assert '95.0%' in report
        assert 'Majority Baseline' in report
        assert '100 evaluation rows' in report
        assert 'trained on' not in report.lower()
        assert 'excellent performance' not in report
        assert 'highly reliable' not in report
        assert '<script>' not in report
        assert 'skipped' in report
        assert 'Raw-Column Feature Importance' in report
        assert 'Metric basis: accuracy' in report
        assert '100 evaluation rows selected with random_state=42' in report
    assert len(model.importance_calls) == 1
    assert model.importance_calls[0]['feature_stage'] == 'original'
    assert model.importance_calls[0]['features'] == ['text']
    assert model.importance_calls[0]['num_shuffle_sets'] == 3
    assert 'model' not in model.importance_calls[0]
    assert 'not SHAP' in ReportBuilder(explainer).build('expert')


def test_perturbation_failure_transparent_no_partial_card_or_fake_deltas():
    model = FakePredictor()
    original = model.predict_proba
    def fail_batch(X):
        if len(X) > 1:
            raise RuntimeError('perturbation <failure>')
        return original(X)
    model.predict_proba = fail_batch
    explainer = AutoGluonTabularExplainer(model, pd.DataFrame({'text': ['risk other']}), pd.Series([3]))
    report = ReportBuilder(explainer).build('expert')
    assert 'Word-removal explanation failed' in report
    assert 'no contributions fabricated' in report
    assert '<div class="text-card">' not in report
    assert 'perturbation &lt;failure&gt;' in report


def test_nontext_autogluon_still_uses_native_importance():
    model = FakePredictor(('number',), text=False)
    explainer = ExplainerFactory.create(model, pd.DataFrame({'number': [1., 2.]}),
                                       pd.Series([3, 1]), model_type='tabular')
    assert explainer._detect_text_column() is None
    assert explainer.get_feature_importance().to_dict('records') == [{'feature': 'number', 'importance': .25}]
    assert explainer.get_metrics()['accuracy'] == .5


def test_short_text_hint_and_long_text_heuristic():
    for name, value in [('text', 'ok'), ('other', 'this sufficiently long sentence contains many distinct words')]:
        model = FakePredictor((name,), text=False)
        explainer = AutoGluonTabularExplainer(model, pd.DataFrame({name: [value]}), pd.Series([3]))
        assert explainer._detect_text_column() == name


def test_float_class_labels_confusion_plot_keeps_model_order(monkeypatch):
    from xai_core.visualizations import performance_plots
    actual_display = performance_plots.ConfusionMatrixDisplay
    captured = []
    def capture(**kwargs):
        captured.append(kwargs)
        return actual_display(**kwargs)
    monkeypatch.setattr(performance_plots, 'ConfusionMatrixDisplay', capture)
    image = performance_plots.plot_confusion_matrix(
        pd.Series([3., 1., 2.]), np.array([3., 3., 3.]), [3., 1., 2.])
    assert image
    assert captured[0]['confusion_matrix'].tolist() == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    assert captured[0]['display_labels'] == [3., 1., 2.]


def test_nontext_regression_metrics_and_native_info():
    model = FakePredictor(('number',), text=False)
    model.problem_type = 'regression'
    model.predict = lambda X: X['number'].to_numpy() * 2
    model.model_best = 'FullEnsemble'
    model.model_names = lambda: ['BaseOne', 'FullEnsemble']
    explainer = AutoGluonTabularExplainer(model, pd.DataFrame({'number': [1., 2., 3.]}),
                                       pd.Series([2., 4., 6.]))
    metrics = explainer.get_metrics()
    assert metrics['r2'] == 1
    assert metrics['mae'] == 0
    assert metrics['rmse'] == 0
    assert metrics['best_model'] == 'FullEnsemble'
    assert metrics['ensemble_models'] == ['BaseOne', 'FullEnsemble']


@pytest.mark.parametrize('max_samples, expected_rows', [(1000, 200), (30, 30)])
def test_raw_text_importance_uses_bounded_reproducible_original_sample(max_samples, expected_rows):
    model = FakePredictor(('text', 'amount'))
    X = pd.DataFrame({'text': [f'risk text {i}\n  suffix!' for i in range(300)],
                      'amount': list(range(300))}, index=[7] * 300)
    y = pd.Series([1, 2, 3] * 100, index=[7] * 300, name='labels')
    explainer = AutoGluonTabularExplainer(model, X, y, max_samples=max_samples)
    importance = explainer.get_feature_importance()
    assert importance['feature'].tolist() == ['text', 'amount']
    call = model.importance_calls[0]
    assert call['feature_stage'] == 'original'
    assert call['features'] == ['text', 'amount']
    assert call['num_shuffle_sets'] == 3
    assert call['subsample_size'] == expected_rows
    assert 'model' not in call
    expected = X.copy()
    expected['labels'] = y.to_numpy()
    pd.testing.assert_frame_equal(call['data'], expected.sample(n=expected_rows, random_state=42))
    second = AutoGluonTabularExplainer(model, X, y, max_samples=max_samples)
    second.get_feature_importance()
    pd.testing.assert_frame_equal(call['data'], model.importance_calls[1]['data'])
    assert all(not c.startswith('__nlp__') for c in call['features'])


def test_native_importance_skips_above_raw_feature_budget():
    columns = ['text'] + [f'number{i}' for i in range(20)]
    model = FakePredictor(columns)
    X = pd.DataFrame({c: ['risk'] if c == 'text' else [1] for c in columns})
    explainer = AutoGluonTabularExplainer(model, X, pd.Series([3]))
    assert explainer.get_feature_importance() is None
    assert not model.importance_calls
    assert any('21 original features exceed the 20-feature budget' in note for note in explainer.explanation_notes)


def test_native_importance_failure_is_transparent_not_fake_scores_or_model_switch():
    model = FakePredictor()
    calls = []
    def fail_importance(**kwargs):
        calls.append(kwargs)
        raise ModuleNotFoundError('missing <backend>')
    model.feature_importance = fail_importance
    model.model_names = lambda: ['OtherMember']
    explainer = AutoGluonTabularExplainer(model, pd.DataFrame({'text': ['risk', 'other']}), pd.Series([3, 1]))
    report = ReportBuilder(explainer).build('expert')
    assert 'Native raw-column feature importance failed' in report
    assert 'missing &lt;backend&gt;' in report
    assert 'no replacement model or fabricated scores were used' in report
    assert 'Raw-Column Feature Importance' not in report
    assert len(calls) == 1
    assert 'model' not in calls[0]
