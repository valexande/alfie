"""
AutoGluon Tabular Explainer - Optimized for AutoGluon TabularPredictor.

Uses native AutoGluon feature_importance() and adapters for SHAP.
"""

from typing import Dict, Any, Optional, List
import html
import re
import pandas as pd
import numpy as np
import warnings

from xai_core.base_explainer import BaseModelExplainer

warnings.filterwarnings('ignore')

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


class AutoGluonTabularExplainer(BaseModelExplainer):
    """
    Explainer optimized for AutoGluon TabularPredictor.
    
    Uses:
    - Native predictor.feature_importance() for fast, reliable importance
    - SHAP via KernelExplainer (TreeExplainer not directly compatible)
    - Native predictor methods for predictions
    
    Example:
        >>> from autogluon.tabular import TabularPredictor
        >>> predictor = TabularPredictor.load("my_model")
        >>> explainer = AutoGluonTabularExplainer(predictor, X_test, y_test)
        >>> importance = explainer.get_feature_importance()
    """
    
    def __init__(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        label: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize AutoGluon Tabular explainer.
        
        Args:
            model: TabularPredictor instance
            X: Feature DataFrame
            y: Target Series
            label: Target column name (for feature_importance)
            **kwargs: Additional configuration
        """
        # Filter X to the exact features the predictor was trained on.
        # This removes metadata columns (image paths, text labels, split flags, etc.)
        # that the model never saw during training and would cause predict() to fail
        # or produce garbage SHAP values.
        X = self._filter_to_model_features(model, X)

        super().__init__(model, X, y, **kwargs)
        self.predictor = model
        self.label = label or (y.name if hasattr(y, 'name') and y.name else 'target')
        self._text_explanations_html: Optional[str] = None
        
        # Build full DataFrame for AutoGluon methods
        self._full_data = X.copy()
        self._full_data[self.label] = y.values
    
    # -------------------------------------------------------------------------
    # Feature alignment
    # -------------------------------------------------------------------------

    @staticmethod
    def _filter_to_model_features(predictor, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align X to the exact feature set the predictor was trained on.

        Handles three scenarios that are common when a raw CSV is uploaded:

        1. X already has every expected column → just reorder/select.
        2. X has raw categorical columns that were OHE'd before training
           (e.g. CSV has ``gender`` but model expects ``gender_Female``,
           ``gender_Male``) → apply pd.get_dummies then align.
        3. X has garbage columns (file paths, leaky text labels, split
           flags, free-text notes …) → drop them silently.

        Falls back to the unmodified X if feature metadata cannot be read.
        """
        try:
            expected = list(predictor.feature_metadata.type_map_raw.keys())
        except Exception as e:
            print(f"  Could not read predictor.feature_metadata for alignment: {e}")
            return X

        if not expected:
            return X

        # ── Case 1: X already has all expected columns ──────────────────────
        if all(c in X.columns for c in expected):
            dropped = [c for c in X.columns if c not in expected]
            if dropped:
                print(
                    f"  AutoGluon feature alignment: dropping {len(dropped)} non-model "
                    f"column(s): {dropped[:8]}{'...' if len(dropped) > 8 else ''}"
                )
            return X[expected]

        # ── Case 2 / 3: mix of raw categoricals and garbage columns ──────────
        # Identify raw categorical columns whose OHE form appears in expected
        # (heuristic: expected contains "{rawcol}_{value}" entries)
        raw_ohe_cols = [
            c for c in X.columns
            if c not in expected and any(exp.startswith(f"{c}_") for exp in expected)
        ]
        garbage_cols = [
            c for c in X.columns
            if c not in expected and c not in raw_ohe_cols
        ]

        if garbage_cols:
            print(
                f"  AutoGluon feature alignment: dropping {len(garbage_cols)} non-model "
                f"column(s): {garbage_cols[:8]}{'...' if len(garbage_cols) > 8 else ''}"
            )
        X = X.drop(columns=garbage_cols, errors='ignore')

        if raw_ohe_cols:
            print(f"  AutoGluon feature alignment: OHE-encoding {raw_ohe_cols}")
            X = pd.get_dummies(X, columns=raw_ohe_cols)

        # Add expected OHE columns that didn't appear in the data (unseen categories)
        missing = [c for c in expected if c not in X.columns]
        if missing:
            for col in missing:
                X[col] = 0

        available = [c for c in expected if c in X.columns]
        return X[available]

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return 'autogluon_tabular'
    
    @property
    def problem_type(self) -> str:
        """Get problem type from predictor."""
        ag_type = getattr(self.predictor, 'problem_type', 'unknown')
        
        if ag_type in ['binary', 'multiclass']:
            return 'classification'
        elif ag_type in ['regression', 'quantile']:
            return 'regression'
        
        return 'regression'
    
    @staticmethod
    def _is_missing_module(err: str) -> bool:
        """Return True for any 'No module named X' or import-related error."""
        return (
            "No module named" in err
            or "fastai" in err.lower()
            or "fasttransform" in err.lower()
            or "ImportError" in err
        )

    def get_predictions(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Get predictions using native predictor, with fallbacks for common issues."""
        if X is None:
            X = self.X

        try:
            predictions = self.predictor.predict(X)
        except Exception as e:
            err = str(e)
            if self._is_missing_module(err):
                # Some model in the ensemble needs a missing package (fastai /
                # fasttransform / …).  Fall back to the best available model.
                all_names = self._get_all_model_names()
                skip = {'NeuralNetFastAI'}
                fallback = [m for m in all_names if m not in skip]
                if not fallback:
                    raise
                print(f"  Missing module ({err[:60]}) — predicting with: {fallback[0]}")
                predictions = self.predictor.predict(X, model=fallback[0])
            elif 'required columns are missing' in err or 'missing columns' in err.lower():
                print("  OHE column mismatch — applying get_dummies and retrying...")
                predictions = self.predictor.predict(self._apply_ohe_if_needed(X))
            else:
                raise

        if hasattr(predictions, 'values'):
            return predictions.values
        return np.array(predictions)

    def _apply_ohe_if_needed(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode low-cardinality string columns when the model was trained
        on pre-OHE data.

        High-cardinality columns (> 50 unique values) — file paths, free-text
        notes, video filenames, subject IDs stored as strings, etc. — are
        silently dropped instead of exploding the feature space with hundreds of
        binary dummy columns.
        """
        cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == 'category']
        if not cat_cols:
            return X

        MAX_CARDINALITY = 50
        ohe_cols  = [c for c in cat_cols if X[c].nunique() <= MAX_CARDINALITY]
        skip_cols = [c for c in cat_cols if X[c].nunique() >  MAX_CARDINALITY]

        if skip_cols:
            print(
                f"  OHE: dropping {len(skip_cols)} high-cardinality string column(s) "
                f"(> {MAX_CARDINALITY} unique values): {skip_cols}"
            )
            X = X.drop(columns=skip_cols)

        if not ohe_cols:
            return X

        X_ohe = pd.get_dummies(X, columns=ohe_cols)
        # Store for reuse in SHAP / feature_importance
        self._X_ohe = X_ohe
        return X_ohe

    def _get_all_model_names(self) -> list:
        """Get model names using whichever API is available."""
        for method in ('get_model_names', 'model_names'):
            if hasattr(self.predictor, method):
                return list(getattr(self.predictor, method)())
        # leaderboard fallback
        try:
            return list(self.predictor.leaderboard(silent=True).index)
        except Exception:
            return []

    def _predict_without_fastai(self, X: pd.DataFrame) -> np.ndarray:
        """Fall back to the best non-fastai model in the ensemble."""
        try:
            all_models = self._get_all_model_names()
            non_fastai = [m for m in all_models if 'NeuralNetFastAI' not in m]
            if not non_fastai:
                raise RuntimeError("No non-fastai models available in this AutoGluon predictor.")
            best = non_fastai[0]
            print(f"fastai missing — predicting with fallback model: {best}")
            return self.predictor.predict(X, model=best)
        except Exception as fallback_err:
            raise RuntimeError(f"fastai unavailable and fallback also failed: {fallback_err}")
    
    def get_prediction_probabilities(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """Get prediction probabilities."""
        if not self.is_classification:
            return None
        
        if X is None:
            X = self.X
        
        try:
            proba = self.predictor.predict_proba(X)
        except Exception as e:
            err = str(e)
            try:
                if self._is_missing_module(err):
                    non_fastai = [m for m in self._get_all_model_names() if 'NeuralNetFastAI' not in m]
                    proba = self.predictor.predict_proba(X, model=non_fastai[0]) if non_fastai else None
                elif 'required columns are missing' in err or 'missing columns' in err.lower():
                    proba = self.predictor.predict_proba(self._apply_ohe_if_needed(X))
                else:
                    return None
            except Exception:
                return None
            if proba is None:
                return None

        if isinstance(proba, pd.DataFrame):
            return proba.values
        return proba

    def get_text_explanations_html(self, max_examples: int = 6, max_tokens: int = 60) -> Optional[str]:
        """
        Generate word-level explanations for text columns inside TabularPredictor data.

        AutoGluon tabular models can be trained with a free-text column. Native
        feature importance explains the whole column; this local perturbation
        method masks one word at a time and measures how much the predicted-class
        probability changes.
        """
        if self._text_explanations_html is not None:
            return self._text_explanations_html
        if not self.is_classification:
            return None

        text_col = self._detect_text_column()
        if text_col is None:
            return None

        rows = self._select_text_rows(text_col, max_examples)
        if not rows:
            return None

        cards = []
        for row_idx in rows:
            try:
                card = self._build_text_explanation_card(row_idx, text_col, max_tokens)
                if card:
                    cards.append(card)
            except Exception as e:
                print(f"Text explanation failed for row {row_idx}: {e}")

        if not cards:
            return None

        self._text_explanations_html = f'''
        <div class="text-explanations">
            {"".join(cards)}
        </div>'''
        return self._text_explanations_html

    def generate_plots(self) -> Dict[str, str]:
        """Generate standard plots plus word-level text explanations when possible."""
        plots = super().generate_plots()
        text_html = self.get_text_explanations_html()
        if text_html:
            plots['text_explanations'] = text_html
        return plots

    def _detect_text_column(self) -> Optional[str]:
        """Find the most likely free-text input column."""
        candidates = []
        for col in self.X.columns:
            series = self.X[col].dropna()
            if series.empty:
                continue
            if not (series.dtype == object or str(series.dtype) == 'category'):
                continue
            sample = series.astype(str).head(200)
            avg_words = sample.str.split().str.len().mean()
            avg_chars = sample.str.len().mean()
            unique_ratio = sample.nunique() / max(len(sample), 1)
            if avg_words >= 5 and avg_chars >= 30 and unique_ratio >= 0.5:
                candidates.append((col, avg_words, avg_chars))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return candidates[0][0]

    def _select_text_rows(self, text_col: str, max_examples: int) -> List[int]:
        """Pick a small, class-aware set of non-empty examples."""
        valid_indices = [
            idx for idx, value in self.X[text_col].items()
            if isinstance(value, str) and value.strip()
        ]
        if not valid_indices:
            return []

        selected = []
        if self.y is not None:
            classes = self.classes
            for _class in (list(classes) if classes is not None else [])[:max_examples]:
                class_indices = [
                    idx for idx in valid_indices
                    if idx in self.y.index and self.y.loc[idx] == _class
                ]
                if class_indices:
                    selected.append(class_indices[0])

        for idx in valid_indices:
            if len(selected) >= max_examples:
                break
            if idx not in selected:
                selected.append(idx)

        return selected[:max_examples]

    def _build_text_explanation_card(
        self,
        row_idx: int,
        text_col: str,
        max_tokens: int,
    ) -> Optional[str]:
        row = self.X.loc[[row_idx]].copy()
        text = str(row.iloc[0][text_col])
        tokens = self._tokenize_text(text)[:max_tokens]
        if len(tokens) < 2:
            return None

        base_proba_df = self._predict_proba_df(row)
        if base_proba_df is None or base_proba_df.empty:
            return None

        predicted_label = base_proba_df.iloc[0].idxmax()
        predicted_score = float(base_proba_df.iloc[0].max())
        predicted_col = predicted_label

        perturbed_rows = []
        token_positions = []
        for pos, token in enumerate(tokens):
            if not self._is_explainable_token(token):
                continue
            masked_tokens = tokens.copy()
            masked_tokens[pos] = ""
            perturbed = row.copy()
            perturbed.iloc[0, perturbed.columns.get_loc(text_col)] = " ".join(
                t for t in masked_tokens if t
            )
            perturbed_rows.append(perturbed)
            token_positions.append(pos)

        scores = [0.0] * len(tokens)
        if perturbed_rows:
            perturbed_df = pd.concat(perturbed_rows, ignore_index=True)
            perturbed_proba_df = self._predict_proba_df(perturbed_df)
            if perturbed_proba_df is not None and predicted_col in perturbed_proba_df.columns:
                masked_scores = perturbed_proba_df[predicted_col].astype(float).values
                for pos, masked_score in zip(token_positions, masked_scores):
                    scores[pos] = predicted_score - float(masked_score)

        highlighted = self._render_highlighted_tokens(tokens, scores)
        top_words = self._render_top_tokens(tokens, scores)
        true_label = self.y.loc[row_idx] if row_idx in self.y.index else ""

        return f'''
        <div class="text-card">
            <div class="text-card-meta">
                <span><strong>Row:</strong> {html.escape(str(row_idx))}</span>
                <span><strong>True label:</strong> {html.escape(str(true_label))}</span>
                <span><strong>Predicted:</strong> {html.escape(str(predicted_label))}</span>
                <span><strong>Confidence:</strong> {predicted_score:.1%}</span>
            </div>
            <div class="token-highlight">{highlighted}</div>
            {top_words}
        </div>'''

    def _predict_proba_df(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Predict probabilities and preserve class labels when available."""
        try:
            proba = self.predictor.predict_proba(X)
        except Exception as e:
            err = str(e)
            try:
                if self._is_missing_module(err):
                    non_fastai = [m for m in self._get_all_model_names() if 'NeuralNetFastAI' not in m]
                    proba = self.predictor.predict_proba(X, model=non_fastai[0]) if non_fastai else None
                elif 'required columns are missing' in err or 'missing columns' in err.lower():
                    proba = self.predictor.predict_proba(self._apply_ohe_if_needed(X))
                else:
                    return None
            except Exception:
                return None

        if proba is None:
            return None
        if isinstance(proba, pd.DataFrame):
            return proba
        return pd.DataFrame(proba, columns=list(self.classes))

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        """Tokenize while keeping useful punctuation attached for readability."""
        return re.findall(r"\S+", text)

    @staticmethod
    def _is_explainable_token(token: str) -> bool:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "", token)
        return len(cleaned) >= 2

    @staticmethod
    def _render_highlighted_tokens(tokens: List[str], scores: List[float]) -> str:
        max_abs = max([abs(score) for score in scores] + [0.0])
        rendered = []
        for token, score in zip(tokens, scores):
            escaped = html.escape(token)
            if max_abs == 0 or abs(score) < 1e-6:
                rendered.append(f'<span class="token-neutral">{escaped}</span>')
                continue

            strength = min(abs(score) / max_abs, 1.0)
            alpha = 0.18 + 0.52 * strength
            cls = "token-positive" if score > 0 else "token-negative"
            title = f"Probability change when removed: {score:+.3f}"
            rendered.append(
                f'<span class="{cls}" style="--token-alpha:{alpha:.3f}" '
                f'title="{html.escape(title)}">{escaped}</span>'
            )
        return " ".join(rendered)

    @staticmethod
    def _render_top_tokens(tokens: List[str], scores: List[float], limit: int = 8) -> str:
        ranked = sorted(
            [
                (token, score)
                for token, score in zip(tokens, scores)
                if abs(score) >= 1e-6
            ],
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:limit]
        if not ranked:
            return ""

        rows = "".join(
            f"<tr><td>{html.escape(token)}</td><td>{score:+.3f}</td></tr>"
            for token, score in ranked
        )
        return f'''
        <table class="token-table">
            <tr><th>Token</th><th>Impact on predicted class</th></tr>
            {rows}
        </table>'''
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance using native AutoGluon method.
        
        AutoGluon's feature_importance() uses permutation-based
        importance which is model-agnostic and reliable.
        
        Returns:
            DataFrame with ['feature', 'importance'] columns
        """
        if self._feature_importance is not None:
            return self._feature_importance
        
        try:
            print("Computing native AutoGluon feature importance...")

            # Build the full DataFrame the predictor expects (features + label)
            full_data = self._full_data

            # Use native feature_importance method (ensemble-level)
            importance = self.predictor.feature_importance(
                data=full_data,
                subsample_size=min(self.max_samples, len(full_data)),
                num_shuffle_sets=5,
                silent=True,
            )
            
            if isinstance(importance, pd.DataFrame):
                # Rename columns if needed
                if 'importance' in importance.columns:
                    importance = importance.reset_index()
                    importance.columns = ['feature', 'importance'] + list(importance.columns[2:])
                else:
                    importance = importance.reset_index()
                    importance.columns = ['feature'] + list(importance.columns[1:])
                    if len(importance.columns) > 1:
                        importance['importance'] = importance.iloc[:, 1]
                
                self._feature_importance = importance[['feature', 'importance']].sort_values(
                    'importance', ascending=False
                ).reset_index(drop=True)
                
                return self._feature_importance
                
        except Exception as e:
            print(f"AutoGluon feature_importance failed: {e}")
        
        # Fallback to permutation importance
        return self._get_permutation_importance()
    
    def _get_permutation_importance(self) -> pd.DataFrame:
        """Compute permutation importance as fallback."""
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.base import BaseEstimator

            X_sample, y_sample = self.sample_data(min(500, self.max_samples))

            # Sklearn's permutation_importance requires a fitted estimator that
            # implements fit().  AutoGluon predictors are already fitted but
            # don't expose that interface, so we wrap them.
            outer_self = self

            class PredictorWrapper(BaseEstimator):
                def fit(self, X, y=None):
                    return self  # already fitted

                def predict(self, X):
                    if isinstance(X, np.ndarray):
                        X = pd.DataFrame(X, columns=outer_self.feature_names)
                    try:
                        result = outer_self.predictor.predict(X)
                    except Exception as e:
                        if outer_self._is_missing_module(str(e)):
                            all_models = outer_self._get_all_model_names()
                            fallback = [m for m in all_models if 'NeuralNetFastAI' not in m]
                            result = outer_self.predictor.predict(X, model=fallback[0])
                        else:
                            raise
                    return result.values if hasattr(result, 'values') else np.array(result)

                def score(self, X, y):
                    from sklearn.metrics import accuracy_score
                    try:
                        preds = self.predict(X)
                        return float(accuracy_score(y, preds))
                    except Exception:
                        return 0.0

            wrapper = PredictorWrapper()

            result = permutation_importance(
                wrapper, X_sample, y_sample,
                n_repeats=5,
                random_state=42,
                n_jobs=1   # avoid multiprocessing issues with predictor
            )

            self._feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': result.importances_mean
            }).sort_values('importance', ascending=False).reset_index(drop=True)

            return self._feature_importance

        except Exception as e:
            print(f"Permutation importance failed: {e}")
            return None
    
    def get_shap_values(self, X_sample: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        Get SHAP values using KernelExplainer.
        
        Note: AutoGluon's ensemble models don't support TreeExplainer directly,
        so we use KernelExplainer which works with any model.
        
        Args:
            X_sample: Samples to explain
            
        Returns:
            np.ndarray of SHAP values
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        try:
            if X_sample is None:
                X_sample, _ = self.sample_data(min(100, self.max_samples))
            
            print("Creating SHAP KernelExplainer for AutoGluon...")
            
            # Create prediction wrapper that gracefully skips models with
            # missing packages (fasttransform, fastai, etc.)
            all_models = self._get_all_model_names()
            safe_model = next(
                (m for m in all_models if 'NeuralNetFastAI' not in m), None
            )

            def predict_fn(X):
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=self.feature_names)
                try:
                    result = self.predictor.predict(X)
                except Exception as e:
                    if self._is_missing_module(str(e)) and safe_model:
                        result = self.predictor.predict(X, model=safe_model)
                    else:
                        raise
                return result.values if hasattr(result, 'values') else np.array(result)
            
            # Sample background data
            background = self.X.sample(n=min(50, len(self.X)), random_state=42)
            
            # Create KernelExplainer
            explainer = shap.KernelExplainer(predict_fn, background)
            
            # Limit samples for performance
            X_limited = X_sample.head(min(50, len(X_sample)))
            
            # Get SHAP values
            shap_values = explainer.shap_values(X_limited, nsamples=100)
            
            return shap_values
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        if self._metrics is not None:
            return self._metrics
        
        metrics = {
            'model_type': self.model_type,
            'problem_type': self.problem_type,
            'n_features': self.n_features,
            'n_samples': self.n_samples,
        }
        
        # Get predictions
        try:
            y_pred = self.get_predictions()
        except Exception as e:
            print(f"Prediction failed: {e}")
            self._metrics = metrics
            return metrics
        
        if self.is_classification:
            metrics.update(self._get_classification_metrics(y_pred))
        else:
            metrics.update(self._get_regression_metrics(y_pred))
        
        # Add AutoGluon-specific info
        metrics.update(self._get_autogluon_info())
        if self._detect_text_column() is not None:
            metrics['has_text_explanations'] = True
        
        self._metrics = metrics
        return metrics
    
    def _get_classification_metrics(self, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score
        )
        
        metrics = {}
        
        try:
            metrics['accuracy'] = round(accuracy_score(self.y, y_pred), 4)
        except Exception:
            pass
        
        try:
            metrics['precision'] = round(
                precision_score(self.y, y_pred, average='weighted', zero_division=0), 4
            )
        except Exception:
            pass
        
        try:
            metrics['recall'] = round(
                recall_score(self.y, y_pred, average='weighted', zero_division=0), 4
            )
        except Exception:
            pass
        
        try:
            metrics['f1'] = round(
                f1_score(self.y, y_pred, average='weighted', zero_division=0), 4
            )
        except Exception:
            pass
        
        # ROC AUC
        try:
            y_proba = self.get_prediction_probabilities()
            if y_proba is not None:
                if len(self.classes) == 2:
                    metrics['roc_auc'] = round(roc_auc_score(self.y, y_proba[:, 1]), 4)
                else:
                    metrics['roc_auc'] = round(
                        roc_auc_score(self.y, y_proba, multi_class='ovr', average='weighted'), 4
                    )
        except Exception:
            pass
        
        return metrics
    
    def _get_regression_metrics(self, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate regression metrics."""
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, r2_score
        )
        
        metrics = {}
        
        try:
            metrics['mae'] = round(mean_absolute_error(self.y, y_pred), 4)
        except Exception:
            pass
        
        try:
            metrics['rmse'] = round(np.sqrt(mean_squared_error(self.y, y_pred)), 4)
        except Exception:
            pass
        
        try:
            metrics['r2'] = round(r2_score(self.y, y_pred), 4)
        except Exception:
            pass
        
        return metrics
    
    def _get_autogluon_info(self) -> Dict[str, Any]:
        """Get AutoGluon-specific model information."""
        info = {}
        
        try:
            info['best_model'] = self.predictor.get_model_best()
        except Exception:
            pass
        
        try:
            info['model_names'] = self.predictor.get_model_names()
        except Exception:
            pass
        
        return info
    
    def get_leaderboard(self) -> Optional[pd.DataFrame]:
        """Get AutoGluon model leaderboard."""
        try:
            return self.predictor.leaderboard(silent=True)
        except Exception:
            return None
