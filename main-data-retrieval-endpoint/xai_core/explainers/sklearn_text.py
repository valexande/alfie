"""Explain raw-text sklearn pipelines that include their fitted vectorizer."""

from typing import Any, Dict, List, Optional
import base64
import html
import io
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xai_core.base_explainer import BaseModelExplainer


class SklearnTextExplainer(BaseModelExplainer):
    """Classification explainer for vectorizer-plus-classifier pipelines."""

    def __init__(self, model: Any, X: pd.DataFrame, y: pd.Series, **kwargs):
        super().__init__(model, X, y, **kwargs)
        self.text_column = self._detect_text_column()
        if self.text_column is None:
            raise ValueError("No raw-text input column was found for the text pipeline")
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Text classification pipeline must provide predict_proba()")
        self._text_explanations_html: Optional[str] = None
        self._shap_text_html: Optional[str] = None
        self._lime_text_html: Optional[str] = None
        self._shap_explainer = None
        self._shap_feature_names = None

    @property
    def model_type(self) -> str:
        return 'sklearn_text'

    @property
    def problem_type(self) -> str:
        return 'classification'

    def _detect_text_column(self) -> Optional[str]:
        candidates = []
        for col in self.X.columns:
            series = self.X[col].dropna()
            if series.empty or not (series.dtype == object or str(series.dtype) == 'category'):
                continue
            sample = series.astype(str).head(200)
            candidates.append((col, sample.str.split().str.len().mean(), sample.str.len().mean()))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[1], item[2]))[0]

    def get_predictions(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        X = self.X if X is None else X
        predictions = self.model.predict(X)
        return predictions.values if hasattr(predictions, 'values') else np.asarray(predictions)

    def get_prediction_probabilities(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        X = self.X if X is None else X
        probabilities = self.model.predict_proba(X)
        return probabilities.values if hasattr(probabilities, 'values') else np.asarray(probabilities)

    def get_feature_importance(self) -> pd.DataFrame:
        """Return coefficient importance using vectorizer feature names."""
        if self._feature_importance is not None:
            return self._feature_importance

        try:
            preprocess = self.model.steps[-2][1]
            classifier = self.model.steps[-1][1]
            names = np.asarray(preprocess.get_feature_names_out(), dtype=str)
            names = np.asarray([name.split('__', 1)[-1] for name in names])
            coefficients = np.asarray(classifier.coef_)
            importance = np.mean(np.abs(coefficients), axis=0)
            self._feature_importance = (
                pd.DataFrame({'feature': names, 'importance': importance})
                .sort_values('importance', ascending=False)
                .head(30)
                .reset_index(drop=True)
            )
        except Exception as exc:
            print(f"Text feature importance unavailable: {exc}")
            self._feature_importance = pd.DataFrame(
                [{'feature': self.text_column, 'importance': 1.0}]
            )
        return self._feature_importance

    def get_shap_values(self, X_sample: Optional[pd.DataFrame] = None):
        """Return native Linear SHAP values in vectorized vocabulary space."""
        try:
            import shap

            preprocessing = self.model.steps[-2][1]
            classifier = self.model.steps[-1][1]
            if X_sample is None:
                X_sample = self.X.head(min(100, len(self.X)))
            if self._shap_explainer is None:
                background_df = self.X.sample(
                    n=min(100, len(self.X)), random_state=42
                )
                background = preprocessing.transform(background_df)
                self._shap_explainer = shap.LinearExplainer(classifier, background)
                names = preprocessing.get_feature_names_out()
                self._shap_feature_names = np.asarray(
                    [str(name).split('__', 1)[-1] for name in names]
                )
            transformed = preprocessing.transform(X_sample)
            return self._shap_explainer(transformed)
        except Exception as exc:
            print(f"Linear SHAP text analysis unavailable: {exc}")
            return None

    def get_metrics(self) -> Dict[str, Any]:
        if self._metrics is not None:
            return self._metrics

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        y_pred = self.get_predictions()
        metrics = {
            'model_type': self.model_type,
            'model_class': type(self.model).__name__,
            'problem_type': self.problem_type,
            'n_features': self.n_features,
            'n_samples': self.n_samples,
            'accuracy': round(accuracy_score(self.y, y_pred), 4),
            'precision': round(precision_score(self.y, y_pred, average='weighted', zero_division=0), 4),
            'recall': round(recall_score(self.y, y_pred, average='weighted', zero_division=0), 4),
            'f1': round(f1_score(self.y, y_pred, average='weighted', zero_division=0), 4),
            'macro_f1': round(f1_score(self.y, y_pred, average='macro', zero_division=0), 4),
            'macro_recall': round(recall_score(self.y, y_pred, average='macro', zero_division=0), 4),
            'has_text_explanations': True,
            'skip_shap_section': True,
        }
        try:
            metrics['roc_auc'] = round(roc_auc_score(
                self.y, self.get_prediction_probabilities(),
                multi_class='ovr', average='weighted', labels=self.model.classes_
            ), 4)
        except Exception:
            pass
        self._metrics = metrics
        return metrics

    def generate_plots(self) -> Dict[str, str]:
        from xai_core.visualizations import (
            plot_confusion_matrix, plot_feature_importance, plot_roc_curve,
        )

        plots = {
            'feature_importance': plot_feature_importance(self.get_feature_importance()),
            'confusion_matrix': plot_confusion_matrix(
                self.y, self.get_predictions(), self.classes
            ),
        }
        roc_plot = plot_roc_curve(
            self.y, self.get_prediction_probabilities(), self.classes
        )
        if roc_plot:
            plots['roc_curve'] = roc_plot
        shap_global = self.get_shap_global_plot()
        if shap_global:
            plots['shap_text_global'] = shap_global
        shap_html = self.get_shap_text_explanations_html()
        if shap_html:
            plots['shap_text_explanations'] = shap_html
        lime_html = self.get_lime_text_explanations_html()
        if lime_html:
            plots['lime_text_explanations'] = lime_html
        text_html = self.get_text_explanations_html()
        if text_html:
            plots['text_explanations'] = text_html
        return plots

    def _selected_indices(self, max_examples: int = 6) -> List[int]:
        selected = []
        for label in np.unique(self.y):
            matches = self.y[self.y == label].index.tolist()
            if matches:
                selected.append(matches[0])
        for idx in self.X.index:
            if len(selected) >= max_examples:
                break
            if idx not in selected:
                selected.append(idx)
        return selected[:max_examples]

    def get_shap_global_plot(self, max_samples: int = 300) -> Optional[str]:
        """Plot mean absolute Linear SHAP contribution for top tokens."""
        sample = self.X.sample(n=min(max_samples, len(self.X)), random_state=42)
        explanation = self.get_shap_values(sample)
        if explanation is None or self._shap_feature_names is None:
            return None
        values = np.asarray(explanation.values)
        importance = np.mean(np.abs(values), axis=(0, 2)) if values.ndim == 3 else np.mean(np.abs(values), axis=0)
        top = np.argsort(importance)[-25:]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(self._shap_feature_names[top], importance[top], color='#4f46e5')
        ax.set_xlabel('Mean absolute SHAP value')
        ax.set_title('Global SHAP Token Importance')
        ax.grid(axis='x', alpha=0.2)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('ascii')

    def get_shap_text_explanations_html(self, max_examples: int = 6) -> Optional[str]:
        """Render local Linear SHAP contributions for representative sentences."""
        if self._shap_text_html is not None:
            return self._shap_text_html
        indices = self._selected_indices(max_examples)
        explanation = self.get_shap_values(self.X.loc[indices])
        if explanation is None or self._shap_feature_names is None:
            return None

        probabilities = self.get_prediction_probabilities(self.X.loc[indices])
        cards = []
        for row_pos, row_idx in enumerate(indices):
            pred_pos = int(np.argmax(probabilities[row_pos]))
            pred_label = self.model.classes_[pred_pos]
            values = np.asarray(explanation.values[row_pos])
            contributions = (
                values[:, pred_pos] if values.ndim == 2
                else values if pred_pos == 1 else -values
            )
            transformed_row = self.model.steps[-2][1].transform(self.X.loc[[row_idx]])
            active = (
                transformed_row.getrow(0).indices
                if hasattr(transformed_row, 'getrow')
                else np.flatnonzero(np.asarray(transformed_row)[0])
            )
            ranked = sorted(
                [(self._shap_feature_names[i], float(contributions[i])) for i in active],
                key=lambda item: abs(item[1]), reverse=True
            )[:10]
            unigram_scores = {term: score for term, score in ranked if ' ' not in term}
            raw_tokens = re.findall(r"\S+", str(self.X.loc[row_idx, self.text_column]))
            token_scores = [
                unigram_scores.get(re.sub(r"[^a-z0-9]+", "", token.lower()), 0.0)
                for token in raw_tokens
            ]
            highlighted = self._render_highlighted_tokens(
                raw_tokens, token_scores, title_prefix='SHAP contribution to predicted-class logit'
            )
            rows = ''.join(
                f'<tr><td>{html.escape(term)}</td><td>{score:+.4f}</td></tr>'
                for term, score in ranked
            )
            cards.append(f'''
            <div class="text-card">
                <div class="text-card-meta">
                    <span><strong>Row:</strong> {row_idx}</span>
                    <span><strong>True label:</strong> {html.escape(str(self.y.loc[row_idx]))}</span>
                    <span><strong>Predicted:</strong> {html.escape(str(pred_label))}</span>
                    <span><strong>Confidence:</strong> {probabilities[row_pos, pred_pos]:.1%}</span>
                </div>
                <div class="token-highlight">{highlighted}</div>
                <table class="token-table"><tr><th>Token / phrase</th><th>SHAP value</th></tr>{rows}</table>
            </div>''')
        self._shap_text_html = f'<div class="text-explanations">{"".join(cards)}</div>'
        return self._shap_text_html

    def get_lime_text_explanations_html(self, max_examples: int = 6) -> Optional[str]:
        """Render LIME Text local surrogate explanations."""
        if self._lime_text_html is not None:
            return self._lime_text_html
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            print("LIME Text unavailable. Install the 'lime' package to enable it.")
            return None

        indices = self._selected_indices(max_examples)
        class_names = [str(value) for value in self.model.classes_]
        lime_explainer = LimeTextExplainer(class_names=class_names, random_state=42)

        def predict_text(texts):
            return self.get_prediction_probabilities(
                pd.DataFrame({self.text_column: list(texts)})
            )

        cards = []
        for row_idx in indices:
            text_value = str(self.X.loc[row_idx, self.text_column])
            probabilities = predict_text([text_value])[0]
            pred_pos = int(np.argmax(probabilities))
            explanation = lime_explainer.explain_instance(
                text_value, predict_text, labels=(pred_pos,),
                num_features=10, num_samples=600,
            )
            ranked = [(term, float(score)) for term, score in explanation.as_list(label=pred_pos)]
            score_map = {term.lower(): score for term, score in ranked}
            raw_tokens = re.findall(r"\S+", text_value)
            token_scores = [
                score_map.get(re.sub(r"[^a-z0-9]+", "", token.lower()), 0.0)
                for token in raw_tokens
            ]
            highlighted = self._render_highlighted_tokens(
                raw_tokens, token_scores, title_prefix='LIME local surrogate weight'
            )
            rows = ''.join(
                f'<tr><td>{html.escape(term)}</td><td>{score:+.4f}</td></tr>'
                for term, score in ranked
            )
            cards.append(f'''
            <div class="text-card">
                <div class="text-card-meta">
                    <span><strong>Row:</strong> {row_idx}</span>
                    <span><strong>True label:</strong> {html.escape(str(self.y.loc[row_idx]))}</span>
                    <span><strong>Predicted:</strong> {html.escape(str(self.model.classes_[pred_pos]))}</span>
                    <span><strong>Confidence:</strong> {probabilities[pred_pos]:.1%}</span>
                </div>
                <div class="token-highlight">{highlighted}</div>
                <table class="token-table"><tr><th>Token</th><th>LIME weight</th></tr>{rows}</table>
            </div>''')
        self._lime_text_html = f'<div class="text-explanations">{"".join(cards)}</div>'
        return self._lime_text_html

    def get_text_explanations_html(self, max_examples: int = 6, max_tokens: int = 60) -> Optional[str]:
        if self._text_explanations_html is not None:
            return self._text_explanations_html

        selected = self._selected_indices(max_examples)

        cards = []
        for idx in selected[:max_examples]:
            card = self._build_text_card(idx, max_tokens)
            if card:
                cards.append(card)
        if not cards:
            return None
        self._text_explanations_html = f'<div class="text-explanations">{"".join(cards)}</div>'
        return self._text_explanations_html

    def _build_text_card(self, row_idx: int, max_tokens: int) -> Optional[str]:
        row = self.X.loc[[row_idx]].copy()
        tokens = re.findall(r"\S+", str(row.iloc[0][self.text_column]))[:max_tokens]
        if len(tokens) < 2:
            return None

        base = self.get_prediction_probabilities(row)[0]
        pred_pos = int(np.argmax(base))
        predicted_label = self.model.classes_[pred_pos]
        predicted_score = float(base[pred_pos])

        perturbed = []
        positions = []
        for pos, token in enumerate(tokens):
            if len(re.sub(r"[^A-Za-z0-9]+", "", token)) < 2:
                continue
            changed = row.copy()
            changed.iloc[0, changed.columns.get_loc(self.text_column)] = " ".join(
                value for i, value in enumerate(tokens) if i != pos
            )
            perturbed.append(changed)
            positions.append(pos)

        scores = [0.0] * len(tokens)
        if perturbed:
            masked = self.get_prediction_probabilities(pd.concat(perturbed, ignore_index=True))[:, pred_pos]
            for pos, score in zip(positions, masked):
                scores[pos] = predicted_score - float(score)

        highlighted = self._render_highlighted_tokens(tokens, scores)
        top_tokens = self._render_top_tokens(tokens, scores)
        true_label = self.y.loc[row_idx]
        return f'''
        <div class="text-card">
            <div class="text-card-meta">
                <span><strong>Row:</strong> {html.escape(str(row_idx))}</span>
                <span><strong>True label:</strong> {html.escape(str(true_label))}</span>
                <span><strong>Predicted:</strong> {html.escape(str(predicted_label))}</span>
                <span><strong>Confidence:</strong> {predicted_score:.1%}</span>
            </div>
            <div class="token-highlight">{highlighted}</div>
            {top_tokens}
        </div>'''

    @staticmethod
    def _render_highlighted_tokens(
        tokens: List[str], scores: List[float],
        title_prefix: str = 'Probability change when removed',
    ) -> str:
        max_abs = max([abs(value) for value in scores] + [0.0])
        output = []
        for token, score in zip(tokens, scores):
            escaped = html.escape(token)
            if max_abs == 0 or abs(score) < 1e-6:
                output.append(f'<span class="token-neutral">{escaped}</span>')
                continue
            alpha = 0.18 + 0.52 * min(abs(score) / max_abs, 1.0)
            css_class = 'token-positive' if score > 0 else 'token-negative'
            title = html.escape(f"{title_prefix}: {score:+.3f}")
            output.append(
                f'<span class="{css_class}" style="--token-alpha:{alpha:.3f}" '
                f'title="{title}">{escaped}</span>'
            )
        return ' '.join(output)

    @staticmethod
    def _render_top_tokens(tokens: List[str], scores: List[float], limit: int = 8) -> str:
        ranked = sorted(
            [(token, score) for token, score in zip(tokens, scores) if abs(score) >= 1e-6],
            key=lambda item: abs(item[1]), reverse=True
        )[:limit]
        if not ranked:
            return ''
        rows = ''.join(
            f'<tr><td>{html.escape(token)}</td><td>{score:+.3f}</td></tr>'
            for token, score in ranked
        )
        return f'''<table class="token-table">
            <tr><th>Token</th><th>Impact on predicted class</th></tr>{rows}
        </table>'''
