"""Regression coverage for report fidelity and missing report guidance."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xai_core.data_interpretability_service import DataInterpretabilityService
from xai_core.report_builder import ReportBuilder


class _NoopExplainer:
    pass


def _service_without_embedded_plots(monkeypatch, frame):
    monkeypatch.setattr(
        DataInterpretabilityService,
        '_generate_visualizations',
        lambda self: {},
    )
    return DataInterpretabilityService(frame)


def test_numeric_distribution_uses_every_finite_raw_value(monkeypatch):
    source = pd.DataFrame({
        'right_eye_ar': list(np.linspace(0.10, 0.40, 21)) + [np.nan, np.inf],
    })
    service = _service_without_embedded_plots(monkeypatch, source)

    values = service._numeric_distribution_values('right_eye_ar')
    assert values.tolist() == list(np.linspace(0.10, 0.40, 21))

    profile = service.analysis_results['distribution_profile']['right_eye_ar']
    assert profile == {
        'observed_count': 21,
        'missing_or_non_numeric_count': 1,
        'non_finite_count': 1,
        'unique_count': 21,
        'plot_kind': 'histogram',
        'min': 0.1,
        'median': 0.25,
        'max': 0.4,
    }

    service._plot_numeric_distributions(['right_eye_ar'])
    plotted_rows = sum(patch.get_height() for patch in plt.gca().patches)
    assert plotted_rows == 21
    plt.close('all')


def test_data_report_documents_distribution_source_and_counts(monkeypatch):
    service = _service_without_embedded_plots(
        monkeypatch,
        pd.DataFrame({'right_eye_ar': [0.2, 0.2, 0.3], 'label': [0, 1, 0]}),
    )
    service.plots['numeric_distributions'] = 'placeholder'
    html = ReportBuilder(_NoopExplainer(), data_service=service)._build_data_section({}, 'expert')

    assert 'without sampling, scaling, imputation' in html
    assert '<strong>right_eye_ar</strong>' in html
    assert '<td>3</td>' in html
    assert 'Exact value counts' in html


def test_vision_report_has_section_one_data_description():
    metrics = {
        'model_type': 'pytorch_vision',
        'n_samples': 3,
        'data_profile': {
            'image_count': 3,
            'readable_count': 2,
            'unreadable_count': 1,
            'class_counts': {'open': 2, 'closed': 1},
            'width_range': [320, 640],
            'height_range': [240, 480],
            'formats': ['JPEG', 'PNG'],
            'color_modes': ['L', 'RGB'],
            'resize_size': 256,
            'model_input_size': 224,
            'normalization_mean': [0.485, 0.456, 0.406],
            'normalization_std': [0.229, 0.224, 0.225],
        },
    }
    html = ReportBuilder(_NoopExplainer())._build_data_section(metrics, 'expert')

    assert 'SECTION 1 — YOUR IMAGE DATA' in html
    assert 'Image Dataset Description' in html
    assert '320–640 px' in html
    assert 'Convert to RGB, resize shorter edge to 256 px' in html
    assert '<strong>open</strong>' in html


def test_each_text_explanation_has_an_explicit_colour_legend():
    plots = {
        'text_explanations': '<div>removal</div>',
        'shap_text_explanations': '<div>shap</div>',
        'lime_text_explanations': '<div>lime</div>',
    }
    html = ReportBuilder(_NoopExplainer())._build_advanced_section(
        {'model_type': 'sklearn_text'}, plots, 'expert'
    )

    assert html.count('Colour guide:') == 3
    assert 'Green — supports displayed class' in html
    assert 'Red — opposes displayed class' in html
    assert 'Darker colour = larger effect relative to other words in the same example.' in html
    assert 'No colour — negligible measured effect' in html
