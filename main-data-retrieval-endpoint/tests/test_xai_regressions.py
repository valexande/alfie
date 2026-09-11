"""XAI input, rendering and metric regressions using synthetic predictors."""
import html
import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from tests.test_autogluon_tabular import FakePredictor
from tests.test_text_api import post, request_context  # Shared mocked-loader API fixture.
from xai_core.explainers.autogluon_tabular import AutoGluonTabularExplainer
from xai_core.report_builder import ReportBuilder
from xai_core.utils import InputValidationError, read_evaluation_csv


@pytest.mark.parametrize('csv', [
    b'labels,text\n1,3,risk',
    b'labels,text\n1,risk,',
    b'labels,text\n1',
    b'labels,text\n1,risk\n2,3,review',
])
def test_csv_record_width_cannot_shift_the_target(csv):
    with pytest.raises(InputValidationError, match='fields'):
        read_evaluation_csv(csv)


@pytest.mark.parametrize('text', [
    'x' * 131073,
    'prefix, ' + 'x' * 131073,
    'first line\n' + 'x' * 131073 + '\nlast line',
])
def test_long_csv_fields_round_trip_without_a_hidden_128k_limit(text):
    import csv
    capacity = csv.field_size_limit()
    data = pd.DataFrame({'labels': [3], 'text': [text]}).to_csv(index=False).encode()
    frame = read_evaluation_csv(data)
    assert frame.labels.tolist() == [3]
    assert frame.text.tolist() == [text]
    # Requests do not temporarily alter/restore a process-global parser setting.
    assert csv.field_size_limit() == capacity


def test_csv_quoted_commas_newlines_and_blank_records_are_preserved():
    frame = read_evaluation_csv(b'\nlabels,text\n1,"risk, review\nsecond line"\n\n3,"quote ""here"""\n')
    assert frame.index.tolist() == [0, 1]
    assert frame.labels.tolist() == [1, 3]
    assert frame.text.tolist() == ['risk, review\nsecond line', 'quote "here"']


@pytest.mark.parametrize('csv', [b'labels,text\n1,3,risk', b'labels,text\n1'])
def test_malformed_csv_is_400_before_inference(request_context, csv):
    client, predictor, _ = request_context
    response = post(client, csv)
    assert response.status_code == 400, response.text
    assert 'fields' in response.json()['detail']
    assert not predictor.prediction_calls
    assert not predictor.calls


@pytest.mark.parametrize('classes', [[1, 0], ['yes', 'no']])
def test_binary_roc_legend_agrees_with_model_ordered_metrics(monkeypatch, classes):
    predictor = FakePredictor(('number',), text=False)
    predictor.problem_type = 'binary'
    predictor.class_labels = classes
    X = pd.DataFrame({'number': [0, 1, 0, 1]})
    # The first probability column denotes classes[0], irrespective of sorted order.
    predictor.predict = lambda frame: np.where(frame.number == 1, classes[0], classes[1])
    predictor.predict_proba = lambda frame: pd.DataFrame({classes[1]: 1 - frame.number,
                                                        classes[0]: frame.number})
    y = pd.Series(predictor.predict(X), name='labels')
    explainer = AutoGluonTabularExplainer(predictor, X, y)
    monkeypatch.setattr(explainer, 'get_shap_values', lambda *args: None)
    monkeypatch.setattr('xai_core.visualizations.plot_pca_analysis', lambda *args: (None, None))
    labels = []
    original = Axes.plot

    def capture(self, *args, **kwargs):
        if 'label' in kwargs:
            labels.append(kwargs['label'])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, 'plot', capture)
    assert explainer.get_metrics()['roc_auc'] == 1.0
    assert explainer.generate_plots()['roc_curve']
    assert 'AUC = 1.000' in labels
    assert 'AUC = 0.000' not in labels


def test_long_word_is_bounded_in_ranked_table_but_not_inference():
    word = 'x' * 10000
    predictor = FakePredictor()
    explainer = AutoGluonTabularExplainer(predictor, pd.DataFrame({'text': [word]}),
                                       pd.Series([3], name='labels'))
    card = explainer.get_text_explanations_html(max_examples=1)
    cells = [html.unescape(cell) for cell in re.findall(r'<td>(.*?)</td>', card)]
    assert word not in card
    assert len(cells[0]) <= 120
    assert cells[0].endswith('…')
    assert 'display truncated; full text used for predictions' in card
    assert predictor.calls[0].text.iloc[0] == word
    assert predictor.calls[1].text.iloc[0] == ''


@pytest.mark.parametrize('mode', ['beginner', 'expert'])
def test_unobserved_classes_are_not_claimed_to_be_poorly_predicted(mode):
    predictor = FakePredictor()
    explainer = AutoGluonTabularExplainer(predictor, pd.DataFrame({'text': ['risk']}),
                                       pd.Series([3], name='labels'))
    metrics = explainer.get_metrics()
    assert metrics['accuracy'] == 1.0
    assert metrics['macro_f1'] == pytest.approx(1 / 3)
    report = ReportBuilder(explainer).build(mode)
    assert 'Some model classes have no evaluation examples' in report
    assert 'cannot be assessed' in report
    assert 'weaker performance on minority classes' not in report
