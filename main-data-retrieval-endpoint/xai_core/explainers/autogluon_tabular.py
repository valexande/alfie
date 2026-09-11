"""AutoGluon explanations through the original predictor and its raw-input schema."""

from typing import Any, Dict, Optional
import html
import re

import numpy as np
import pandas as pd

from xai_core.base_explainer import BaseModelExplainer
from xai_core.autogluon_input import align_autogluon_features, text_columns
from xai_core.text_feature_mapping import validate_text_feature_mapping
from xai_core.utils import InputValidationError

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class AutoGluonTabularExplainer(BaseModelExplainer):
    """Do not replace ensemble members or reconstruct the fitted preprocessing pipeline."""

    MAX_TEXT_EXAMPLES = 6
    MAX_TEXT_TOKENS = 60
    TEXT_BATCH_SIZE = 32
    MAX_DISPLAY_CHARS = 2000
    MAX_TOKEN_DISPLAY_CHARS = 120
    MAX_IMPORTANCE_ROWS = 200
    MAX_IMPORTANCE_FEATURES = 20
    IMPORTANCE_SHUFFLES = 3

    def __init__(self, model: Any, X: pd.DataFrame, y: pd.Series,
                 label: Optional[str] = None, text_feature_mapping=None, **kwargs):
        X = align_autogluon_features(model, X)
        super().__init__(model, X, y, **kwargs)
        self.predictor = model
        model_label = getattr(model, 'label', None)
        if label is not None and model_label is not None and label != model_label:
            raise InputValidationError('Explicit target conflicts with AutoGluon model label')
        self.label = model_label or label or y.name or 'target'
        if y.isna().any() or y.map(lambda v: isinstance(v, str) and not v.strip()).any():
            raise InputValidationError('Evaluation target contains empty values')
        self.text_feature_mapping = (
            validate_text_feature_mapping(text_feature_mapping) if text_feature_mapping is not None else None
        )
        self._text_columns = text_columns(model, X)
        self._text_explanations_html = None
        self.explanation_notes = []
        self._full_data = X.copy()
        self._full_data[self.label] = y.to_numpy()
        if self.is_classification:
            unknown = [value for value in pd.unique(y) if value not in self.classes]
            if unknown:
                raise InputValidationError(f'Evaluation labels absent from model class_labels: {unknown}')

    _filter_to_model_features = staticmethod(align_autogluon_features)

    @property
    def model_type(self):
        return 'autogluon_tabular'

    @property
    def problem_type(self):
        return ('classification' if getattr(self.predictor, 'problem_type', None)
                in ('binary', 'multiclass') else 'regression')

    @property
    def classes(self):
        if not self.is_classification:
            return None
        labels = getattr(self.predictor, 'class_labels', None)
        if labels is None or len(labels) < 2 or len(set(labels)) != len(labels):
            raise RuntimeError('AutoGluon classifier must expose unique class_labels')
        return np.asarray(labels)

    def _note(self, note):
        if note not in self.explanation_notes:
            self.explanation_notes.append(note)

    def get_predictions(self, X=None):
        if X is None and self._predictions is not None:
            return self._predictions
        data = self.X if X is None else align_autogluon_features(self.predictor, X)
        predictions = np.asarray(self.predictor.predict(data))
        if predictions.shape != (len(data),):
            raise RuntimeError('AutoGluon returned an invalid prediction shape')
        if self.is_classification and not np.isin(predictions, self.classes).all():
            raise RuntimeError('AutoGluon predictions contain unknown class labels')
        if X is None:
            self._predictions = predictions
        return predictions

    def _predict_proba_df(self, X):
        """Align labelled columns to MODEL order, including classes absent from y."""
        X = align_autogluon_features(self.predictor, X)
        proba = self.predictor.predict_proba(X)
        classes = list(self.classes)
        if isinstance(proba, pd.DataFrame):
            if proba.columns.has_duplicates or set(proba.columns) != set(classes):
                raise RuntimeError('AutoGluon probability columns do not match model class_labels')
            values = proba.loc[:, classes].to_numpy(dtype=float)
        else:
            values = np.asarray(proba, dtype=float)
        if values.shape != (len(X), len(classes)):
            raise RuntimeError('AutoGluon probability shape does not match rows and model class_labels')
        if (not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
                or not np.allclose(values.sum(axis=1), 1, atol=1e-5)):
            raise RuntimeError('AutoGluon returned invalid class probabilities')
        return pd.DataFrame(values, columns=classes)

    def get_prediction_probabilities(self, X=None):
        if not self.is_classification:
            return None
        return self._predict_proba_df(self.X if X is None else X).to_numpy()

    def _detect_text_column(self):
        return self._text_columns[0] if self._text_columns else None

    def _select_text_rows(self, text_col, max_examples):
        """All row identities are positions: duplicate dataframe indices are valid."""
        valid = [pos for pos, value in enumerate(self.X[text_col])
                 if isinstance(value, str) and value.strip()]
        selected = []
        for label in self.classes:
            positions = [pos for pos in valid if self.y.iloc[pos] == label]
            if positions and len(selected) < max_examples:
                selected.append(positions[0])
        for pos in valid:
            if len(selected) >= max_examples:
                break
            if pos not in selected:
                selected.append(pos)
        return selected

    def get_text_explanations_html(self, max_examples=6, max_tokens=60):
        if self._text_explanations_html is not None:
            return self._text_explanations_html
        text_col = self._detect_text_column()
        if text_col is None or not self.is_classification:
            return None
        max_examples = max(0, min(max_examples, self.MAX_TEXT_EXAMPLES))
        max_tokens = max(0, min(max_tokens, self.MAX_TEXT_TOKENS))
        self._note('Removal sensitivity is not SHAP, additive attribution or causal evidence. Removing a word can disrupt overlapping ngrams and create phrase interactions; do not sum deltas.')
        self._note(
            f'Word removal uses the full predictor.predict_proba ensemble on at most {max_examples} '
            f'deterministic, class-aware evaluation examples and the first {max_tokens} word spans per example; '
            f'batches contain at most {self.TEXT_BATCH_SIZE} perturbations. Only input column {text_col!r} '
            f'is explained; other columns and all unremoved text remain unchanged. '
            f'The text excerpt is limited to {self.MAX_DISPLAY_CHARS} characters; the top 8 deltas '
            f'show at most {self.MAX_TOKEN_DISPLAY_CHARS} characters per word, with an ellipsis for truncation. '
            'Unselected spans are not scored. These examples are not a population-level importance estimate.'
        )
        cards = []
        for pos in self._select_text_rows(text_col, max_examples):
            try:
                card = self._build_text_explanation_card(pos, text_col, max_tokens)
                if card:
                    cards.append(card)
            except Exception as exc:
                self._note(f'Word-removal explanation failed for evaluation row position {pos}: {exc}')
        if not cards:
            self._note('Word-removal explanations unavailable: no successful explainable examples; no contributions fabricated.')
            return None
        self._text_explanations_html = '<div class="text-explanations">' + ''.join(cards) + '</div>'
        return self._text_explanations_html

    def _build_text_explanation_card(self, row_pos, text_col, max_tokens):
        row = self.X.iloc[[row_pos]].copy()
        text = row.iloc[0][text_col]
        # Span removal preserves punctuation, whitespace and the ENTIRE suffix.
        spans = []
        for match in re.finditer(r'\w+', text, flags=re.UNICODE):
            if len(spans) >= max_tokens:
                break
            spans.append(match)
        if not spans:
            self._note(f'Word-removal explanation skipped for row position {row_pos}: no selected words.')
            return None
        base = self._predict_proba_df(row).iloc[0]
        label = base.idxmax()
        score = float(base[label])
        deltas = []
        for start in range(0, len(spans), self.TEXT_BATCH_SIZE):
            batch_spans = spans[start:start + self.TEXT_BATCH_SIZE]
            batch = pd.concat([row] * len(batch_spans), ignore_index=True)
            for offset, span in enumerate(batch_spans):
                batch.iloc[offset, batch.columns.get_loc(text_col)] = text[:span.start()] + text[span.end():]
            probabilities = self._predict_proba_df(batch)
            deltas.extend((score - probabilities[label]).tolist())
        # Render only after every requested delta has been successfully computed.
        highlighted = []
        max_abs = max(abs(delta) for delta in deltas)
        cursor = 0
        for span, delta in zip(spans, deltas):
            if span.end() > self.MAX_DISPLAY_CHARS:
                break
            highlighted.append(html.escape(text[cursor:span.start()]))
            cls = 'token-positive' if delta > 0 else 'token-negative' if delta < 0 else 'token-neutral'
            alpha = 0.18 + 0.52 * abs(delta) / max_abs if max_abs else 0.0
            highlighted.append(
                f'<span class="{cls}" style="--token-alpha:{alpha:.3f}" '
                f'title="Probability before minus after removal: {delta:+.4f}">'
                f'{html.escape(span.group())}</span>'
            )
            cursor = span.end()
        highlighted.append(html.escape(text[cursor:self.MAX_DISPLAY_CHARS]))
        if len(text) > self.MAX_DISPLAY_CHARS:
            highlighted.append('… [display truncated; full text used for predictions]')
        ranked = sorted(zip(spans, deltas), key=lambda item: abs(item[1]), reverse=True)[:8]
        rows = ''
        for span, delta in ranked:
            word = span.group()
            if len(word) > self.MAX_TOKEN_DISPLAY_CHARS:
                word = word[:self.MAX_TOKEN_DISPLAY_CHARS - 1] + '…'
            rows += f'<tr><td>{html.escape(word)}</td><td>{delta:+.4f}</td></tr>'
        return f'''<div class="text-card">
            <div class="text-card-meta">Evaluation row position: {row_pos} ·
            True label: {html.escape(str(self.y.iloc[row_pos]))} ·
            Highest-probability class: {html.escape(str(label))} · Probability: {score:.1%}</div>
            <div class="token-highlight" style="white-space:pre-wrap">{''.join(highlighted)}</div>
            <table class="token-table"><tr><th>Removed word</th><th>Probability delta for displayed class</th></tr>{rows}</table>
            </div>'''

    def get_feature_importance(self):
        """Permute whole RAW columns through the full fitted predictor, not generated ngrams."""
        if self._feature_importance is not None:
            return self._feature_importance
        if self.n_features > self.MAX_IMPORTANCE_FEATURES:
            self._note(
                f'Native raw-column permutation importance skipped: {self.n_features} original features '
                f'exceed the {self.MAX_IMPORTANCE_FEATURES}-feature budget. No transformed features were scored.'
            )
            return None
        sample_size = min(self.max_samples, self.MAX_IMPORTANCE_ROWS, len(self.X))
        sample = self._full_data.sample(n=sample_size, random_state=42)
        metric = getattr(self.predictor, 'eval_metric', None)
        metric_name = getattr(metric, 'name', None) or str(metric or 'configured predictor evaluation score')
        self._note(
            f'Native raw-column permutation importance uses the full fitted predictor, feature_stage=original, '
            f'{self.n_features} raw input features and {sample_size} evaluation rows selected with random_state=42 '
            f'(cap: min(max_shap_samples, {self.MAX_IMPORTANCE_ROWS}, evaluation rows)); '
            f'{self.IMPORTANCE_SHUFFLES} shuffle sets per feature. Metric basis: {metric_name}. '
            'Importance is the decrease in the predictor evaluation score after shuffling a whole raw column '
            '(higher-is-better score orientation), not a probability contribution or a causal effect. '
            'This small evaluation subsample can miss rare classes and yields an uncertain estimate; '
            'zero importance is not proof that a feature can be removed.'
        )
        if self._text_columns:
            self._note('Whole-text-column importance does not rank words or ngrams. Vocabulary mapping is descriptive, not feature importance; transformed-feature attribution is deferred.')
        try:
            importance = self.predictor.feature_importance(
                data=sample, features=self.feature_names, feature_stage='original',
                subsample_size=sample_size, num_shuffle_sets=self.IMPORTANCE_SHUFFLES, silent=True,
            )
            self._feature_importance = (
                importance.rename_axis('feature').reset_index()[['feature', 'importance']]
                .sort_values('importance', ascending=False).reset_index(drop=True)
            )
            return self._feature_importance
        except Exception as exc:
            self._note(f'Native raw-column feature importance failed: {exc}; no replacement model or fabricated scores were used.')
            return None

    def get_shap_values(self, X_sample=None):
        if self._text_columns:
            self._note('Generic numeric SHAP and raw-text PCA skipped: they are not valid explanations of the fitted text preprocessing.')
            return None
        if not SHAP_AVAILABLE:
            self._note('SHAP skipped: dependency unavailable.')
            return None
        try:
            sample = self.X.head(min(50, self.max_samples)) if X_sample is None else X_sample.head(50)
            def predict(data):
                frame = pd.DataFrame(data, columns=self.feature_names)
                return self.get_prediction_probabilities(frame) if self.is_classification else self.get_predictions(frame)
            background = self.X.sample(n=min(50, len(self.X)), random_state=42)
            return shap.KernelExplainer(predict, background).shap_values(sample, nsamples=100)
        except Exception as exc:
            self._note(f'SHAP failed: {exc}; no replacement model was used.')
            return None

    def generate_plots(self) -> Dict[str, str]:
        if not self._text_columns:
            return super().generate_plots()
        from xai_core.visualizations import plot_confusion_matrix, plot_residuals, plot_feature_importance
        importance = self.get_feature_importance()
        self.get_shap_values()
        plots = {}
        if importance is not None:
            importance_plot = plot_feature_importance(importance)
            if importance_plot:
                plots['feature_importance'] = importance_plot
            else:
                self._note('Native raw-column feature importance visualization failed.')
        if self.is_classification:
            matrix = plot_confusion_matrix(self.y, self.get_predictions(), self.classes)
            if matrix:
                plots['confusion_matrix'] = matrix
            else:
                self._note('Confusion matrix visualization failed; class-level counts remain in the metrics.')
            text_html = self.get_text_explanations_html()
            if text_html:
                plots['text_explanations'] = text_html
        else:
            plots['residuals'] = plot_residuals(self.y, self.get_predictions())
            self._note('Word-removal probability explanations skipped for regression.')
        return plots

    def get_metrics(self):
        if self._metrics is not None:
            return self._metrics
        # Evaluation failures are fatal: never return a "successful" metrics shell.
        predictions = self.get_predictions()
        metrics = dict(model_type=self.model_type, problem_type=self.problem_type,
                       n_features=self.n_features, n_samples=self.n_samples)
        if self.is_classification:
            probabilities = self.get_prediction_probabilities()  # Validate full-ensemble contract.
            metrics.update(self._get_classification_metrics(predictions))
            # Use one-vs-rest targets explicitly so an unsorted model class order is preserved.
            from sklearn.metrics import roc_auc_score
            if all((self.y == c).any() for c in self.classes):
                if len(self.classes) == 2:
                    metrics['roc_auc'] = float(roc_auc_score(self.y == self.classes[1], probabilities[:, 1]))
                else:
                    aucs = [roc_auc_score(self.y == c, probabilities[:, i]) for i, c in enumerate(self.classes)]
                    weights = [(self.y == c).sum() for c in self.classes]
                    metrics['roc_auc'] = float(np.average(aucs, weights=weights))
            else:
                self._note('ROC AUC skipped: at least one model class is absent from the evaluation rows.')
        else:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            metrics.update(mae=float(mean_absolute_error(self.y, predictions)),
                           rmse=float(np.sqrt(mean_squared_error(self.y, predictions))),
                           r2=float(r2_score(self.y, predictions)))
        metrics.update(self._get_autogluon_info())
        metrics['skip_shap_section'] = bool(self._text_columns)
        self._metrics = metrics
        return metrics

    def _get_classification_metrics(self, predictions):
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                     precision_recall_fscore_support)
        classes = list(self.classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y, predictions, labels=classes, zero_division=0)
        total = len(self.y)
        majority_index = int(np.argmax(support))
        return {
            'accuracy': float(accuracy_score(self.y, predictions)),
            'balanced_accuracy': float(balanced_accuracy_score(self.y, predictions)),
            'precision': float(np.average(precision, weights=support)),
            'recall': float(np.average(recall, weights=support)),
            'f1': float(np.average(f1, weights=support)),
            'macro_f1': float(np.mean(f1)),
            'majority_baseline': float(support[majority_index] / total),
            'majority_class': str(classes[majority_index]),
            'class_distribution': {str(c): {'support': int(support[i]),
                                           'predicted': int(np.sum(predictions == c)),
                                           'fraction': float(support[i] / total)}
                                   for i, c in enumerate(classes)},
            'per_class': {str(c): {'precision': float(precision[i]), 'recall': float(recall[i]),
                                  'f1': float(f1[i]), 'support': int(support[i])}
                          for i, c in enumerate(classes)},
            'confusion_matrix': confusion_matrix(self.y, predictions, labels=classes).tolist(),
            'class_labels': [str(c) for c in classes],
        }

    def _get_autogluon_info(self):
        info = {}
        for key, names in [('best_model', ('model_best', 'get_model_best')),
                           ('model_names', ('model_names', 'get_model_names'))]:
            for name in names:
                try:
                    value = getattr(self.predictor, name)
                    info[key] = value() if callable(value) else value
                    break
                except (AttributeError, TypeError):
                    continue
        if 'model_names' in info:
            info['ensemble_models'] = info['model_names']
        return info

    def get_leaderboard(self):
        try:
            return self.predictor.leaderboard(silent=True)
        except Exception:
            return None
