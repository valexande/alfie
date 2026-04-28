"""
VisionClassifierExplainer — explainability for PyTorch image-classification models.

Integrates with the existing BaseModelExplainer / ReportBuilder pipeline so that
``POST /explain-model`` can handle vision-model ZIPs alongside tabular / AutoGluon
models without any special-casing above the factory layer.

Expected inputs
---------------
model  : a ``VisionModelInfo`` instance (loaded by model_loader)
X      : pd.DataFrame with a single column ``image_path`` containing absolute
         paths to image files on disk
y      : pd.Series of integer class-ids  (0, 1, 2 …)

The constructor is intentionally signature-compatible with all other
BaseModelExplainer subclasses so ExplainerFactory.create() works unchanged.
"""

from __future__ import annotations

import io
import base64
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from xai_core.base_explainer import BaseModelExplainer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional heavy imports (graceful degradation when torch is absent)
# ---------------------------------------------------------------------------
TORCH_AVAILABLE = False
PIL_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: "plt.Figure") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _pil_to_b64(img: "PILImage.Image") -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Main explainer
# ---------------------------------------------------------------------------

class VisionClassifierExplainer(BaseModelExplainer):
    """
    Explainer for ``pytorch_vision`` models.

    Runs batched CPU inference over a labelled image dataset, computes
    standard classification metrics, and generates:
      - Confusion matrix heatmap
      - Class-distribution bar chart (ground-truth vs predicted)
      - GradCAM gallery (one representative image per class)
      - Misclassified gallery (top-K mistakes)
    """

    _BATCH_SIZE = 32
    _GRADCAM_PER_CLASS = 1   # representative correct images shown per class
    _MAX_MISCLASSIFIED = 8   # cap on misclassified examples in gallery

    def __init__(
        self,
        model: Any,          # VisionModelInfo
        X: pd.DataFrame,     # single col: image_path
        y: pd.Series,        # integer class-ids
        max_samples: int = 1000,
        **kwargs,
    ):
        # BaseModelExplainer stores self.model, self.X, self.y
        super().__init__(model=model, X=X, y=y, max_samples=max_samples, **kwargs)
        # model here is the VisionModelInfo object
        self._vision_info = model
        self._probabilities: Optional[np.ndarray] = None   # (n_samples, n_classes)

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> str:
        return "pytorch_vision"

    @property
    def problem_type(self) -> str:
        return "classification"

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _run_inference(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (y_true, y_pred) arrays by batching inference over self.X.

        Also populates self._probabilities (n_samples, n_classes) as a side-effect.
        Results are cached after the first call.
        """
        if self._predictions is not None:
            return self.y.values, self._predictions

        if not TORCH_AVAILABLE or not PIL_AVAILABLE:
            raise RuntimeError(
                "PyTorch and Pillow are required for VisionClassifierExplainer. "
                "Install: pip install torch torchvision pillow"
            )

        vi = self._vision_info
        transform = vi._build_transform()
        image_paths = self.X.iloc[:, 0].tolist()

        all_probs: List[List[float]] = []
        batch_tensors: List[Any] = []

        def _flush_batch():
            if not batch_tensors:
                return
            tensor = torch.stack(batch_tensors)       # (B, C, H, W)
            vi.model.eval()
            with torch.no_grad():
                logits = vi.model(tensor)
                probs = torch.softmax(logits, dim=1)  # (B, n_classes)
            all_probs.extend(probs.tolist())
            batch_tensors.clear()

        for path in image_paths:
            try:
                img = PILImage.open(path).convert("RGB")
                t = transform(img)
                batch_tensors.append(t)
            except Exception as e:
                print(f"WARNING: could not load image {path}: {e}")
                # Uniform distribution placeholder so indices stay aligned
                n_cls = len(vi.labels)
                batch_tensors.append(torch.zeros(3, vi.input_size, vi.input_size))

            if len(batch_tensors) >= self._BATCH_SIZE:
                _flush_batch()

        _flush_batch()

        prob_array = np.array(all_probs)           # (n_samples, n_classes)
        self._probabilities = prob_array
        self._predictions = prob_array.argmax(axis=1)
        return self.y.values, self._predictions

    def get_prediction_probabilities(
        self, X: Optional[pd.DataFrame] = None
    ) -> Optional[np.ndarray]:
        """
        Return softmax probabilities shape (n_samples, n_classes).

        Only supported for self.X (full dataset); X argument is ignored because
        we cache the forward pass from _run_inference().
        """
        self._run_inference()   # populates self._probabilities
        return self._probabilities

    # ------------------------------------------------------------------
    # BaseModelExplainer abstract methods
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Per-class accuracy expressed as 'feature importance'.

        Gives the ReportBuilder something to render in the Feature Importance
        section without leaving it blank.
        """
        y_true, y_pred = self._run_inference()
        vi = self._vision_info
        rows = []
        for cid, cname in sorted(vi.labels.items()):
            mask = y_true == cid
            if mask.sum() == 0:
                continue
            acc = (y_pred[mask] == cid).mean()
            rows.append({"feature": cname, "importance": round(float(acc), 4)})
        df = pd.DataFrame(rows).sort_values("importance", ascending=False)
        return df.reset_index(drop=True)

    def get_shap_values(self, X_sample: Optional[pd.DataFrame] = None) -> np.ndarray:
        """SHAP is not applicable to raw pixel models — return empty array."""
        return np.array([])

    def get_metrics(self) -> Dict[str, Any]:
        """
        Compute accuracy, macro precision/recall/F1, per-class breakdown,
        and confusion matrix.  Returns flags consumed by ReportBuilder.
        """
        if self._metrics is not None:
            return self._metrics

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
            classification_report,
        )

        y_true, y_pred = self._run_inference()
        vi = self._vision_info
        class_names = [vi.labels[i] for i in sorted(vi.labels.keys())]

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec  = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Per-class
        per_class = {}
        labels_present = sorted(np.unique(np.concatenate([y_true, y_pred])))
        for cid in labels_present:
            cname = vi.labels.get(int(cid), str(cid))
            mask = y_true == cid
            support = int(mask.sum())
            if support == 0:
                continue
            per_class[cname] = {
                "accuracy":  round(float((y_pred[mask] == cid).mean()), 3),
                "precision": round(float(precision_score(
                    y_true, y_pred, labels=[cid], average="macro", zero_division=0)), 3),
                "recall":    round(float(recall_score(
                    y_true, y_pred, labels=[cid], average="macro", zero_division=0)), 3),
                "f1":        round(float(f1_score(
                    y_true, y_pred, labels=[cid], average="macro", zero_division=0)), 3),
                "support":   support,
            }

        cm = confusion_matrix(y_true, y_pred, labels=sorted(vi.labels.keys()))

        # ROC AUC — macro OvR, requires probability scores
        roc_auc = None
        label_ids = sorted(vi.labels.keys())
        n_samples = len(y_true)
        n_classes = len(label_ids)
        if n_samples >= n_classes * 3:   # need at least 3 samples per class
            try:
                from sklearn.preprocessing import label_binarize
                from sklearn.metrics import roc_auc_score as sk_roc_auc
                probs = self.get_prediction_probabilities()
                if probs is not None and probs.shape == (n_samples, n_classes):
                    y_bin = label_binarize(y_true, classes=label_ids)
                    if y_bin.shape[1] > 1:
                        roc_auc = round(float(
                            sk_roc_auc(y_bin, probs, average="macro",
                                       multi_class="ovr")
                        ), 4)
            except Exception as e:
                print(f"ROC AUC computation failed (non-fatal): {e}")

        self._metrics = {
            # Standard fields expected by ReportBuilder header
            "model_type":      "pytorch_vision",
            "problem_type":    "classification",
            "n_features":      len(vi.labels),
            "n_samples":       n_samples,
            "class_names":     class_names,
            # Classification metrics
            "accuracy":            round(acc, 4),
            "precision_macro":     round(prec, 4),
            "recall_macro":        round(rec, 4),
            "f1_macro":            round(f1, 4),
            "f1":                  round(f1, 4),   # alias used by metrics narrative
            "per_class":           per_class,
            "confusion_matrix":    cm.tolist(),
            # Signals to ReportBuilder
            "skip_data_section":   True,
            "skip_shap_section":   True,
        }
        if roc_auc is not None:
            self._metrics["roc_auc"] = roc_auc
        return self._metrics

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def generate_plots(self) -> Dict[str, str]:
        plots: Dict[str, str] = {}

        y_true, y_pred = self._run_inference()
        vi = self._vision_info
        metrics = self.get_metrics()
        class_names = metrics["class_names"]
        label_ids = sorted(vi.labels.keys())

        # ---- Confusion matrix ----
        try:
            from sklearn.metrics import confusion_matrix as sk_cm
            cm = sk_cm(y_true, y_pred, labels=label_ids)
            fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 1.2),
                                            max(4, len(class_names) * 1.0)))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            plots["confusion_matrix"] = _fig_to_b64(fig)
        except Exception as e:
            print(f"Confusion matrix plot failed: {e}")

        # ---- Class distribution: ground-truth vs predicted ----
        try:
            gt_counts = {cn: 0 for cn in class_names}
            pr_counts = {cn: 0 for cn in class_names}
            for cid, cname in vi.labels.items():
                gt_counts[cname] = int((y_true == cid).sum())
                pr_counts[cname] = int((y_pred == cid).sum())

            x = np.arange(len(class_names))
            w = 0.35
            fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.5), 4))
            ax.bar(x - w / 2, [gt_counts[c] for c in class_names], w, label="Ground truth")
            ax.bar(x + w / 2, [pr_counts[c] for c in class_names], w, label="Predicted")
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=15)
            ax.set_ylabel("Count")
            ax.set_title("Class Distribution — Ground Truth vs Predicted")
            ax.legend()
            plt.tight_layout()
            plots["class_distribution"] = _fig_to_b64(fig)
        except Exception as e:
            print(f"Class distribution plot failed: {e}")

        # ---- GradCAM gallery ----
        try:
            plots["gradcam_gallery"] = self._build_gradcam_gallery(y_true, y_pred)
        except Exception as e:
            print(f"GradCAM gallery failed: {e}")

        # ---- Misclassified gallery ----
        try:
            plots["misclassified_gallery"] = self._build_misclassified_gallery(y_true, y_pred)
        except Exception as e:
            print(f"Misclassified gallery failed: {e}")

        # ---- ROC curve (only when enough samples for a meaningful curve) ----
        # Requires at least n_classes * 3 samples so each class appears in y_true.
        n_classes = len(vi.labels)
        if len(y_true) >= n_classes * 3:
            try:
                from xai_core.visualizations.performance_plots import plot_roc_curve
                probs = self.get_prediction_probabilities()
                label_ids = sorted(vi.labels.keys())
                if probs is not None and probs.shape == (len(y_true), n_classes):
                    roc_plot = plot_roc_curve(
                        pd.Series(y_true),
                        probs,
                        classes=label_ids,
                        title="ROC Curve (One-vs-Rest, Micro-Average)",
                    )
                    if roc_plot:
                        plots["roc_curve"] = roc_plot
            except Exception as e:
                print(f"ROC curve plot failed (non-fatal): {e}")

        return plots

    # ------------------------------------------------------------------
    # Gallery helpers
    # ------------------------------------------------------------------

    def _build_gradcam_gallery(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Optional[str]:
        """One correctly-classified example per class with GradCAM overlay."""
        if not TORCH_AVAILABLE or not PIL_AVAILABLE:
            return None

        vi = self._vision_info
        image_paths = self.X.iloc[:, 0].tolist()
        label_ids = sorted(vi.labels.keys())

        # Pick one correct example per class
        samples: Dict[int, str] = {}
        for idx, (true_id, pred_id) in enumerate(zip(y_true, y_pred)):
            cid = int(true_id)
            if cid == int(pred_id) and cid not in samples:
                samples[cid] = image_paths[idx]
            if len(samples) == len(label_ids):
                break

        if not samples:
            return None

        n = len(samples)
        fig, axes = plt.subplots(2, n, figsize=(n * 4, 8))
        if n == 1:
            axes = [[axes[0]], [axes[1]]]

        for col, cid in enumerate(sorted(samples)):
            path = samples[cid]
            cname = vi.labels[cid]
            try:
                pil_img = PILImage.open(path).convert("RGB")
                transform = vi._build_transform()
                tensor = transform(pil_img).unsqueeze(0)

                activations: list = []
                gradients: list = []

                target_layer = _auto_gradcam_layer(vi.model)
                if target_layer is None:
                    raise ValueError("No suitable GradCAM layer found")

                fwd_h = target_layer.register_forward_hook(
                    lambda _, __, out: activations.append(out))
                bwd_h = target_layer.register_full_backward_hook(
                    lambda _, __, grad: gradients.append(grad[0]))

                vi.model.eval()
                out = vi.model(tensor)
                vi.model.zero_grad()
                out[0, cid].backward()

                fwd_h.remove()
                bwd_h.remove()

                act = activations[0].detach().squeeze(0)
                grad = gradients[0].detach().squeeze(0)

                if act.dim() == 2:
                    seq_len, c = act.shape
                    if int(seq_len ** 0.5) ** 2 != seq_len:
                        act = act[1:]; grad = grad[1:]; seq_len -= 1
                    grid = int(seq_len ** 0.5)
                    act  = act.reshape(grid, grid, c).permute(2, 0, 1)
                    grad = grad.reshape(grid, grid, c).permute(2, 0, 1)

                weights = grad.mean(dim=(1, 2))
                cam = torch.relu((weights[:, None, None] * act).sum(0)).numpy()
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

                size = vi.input_size
                img_np = np.array(pil_img.resize((size, size)))
                cam_up = np.array(
                    PILImage.fromarray((cam * 255).astype(np.uint8)).resize((size, size))
                ) / 255.0
                heatmap = cm_jet(cam_up)[:, :, :3]
                overlay = (0.55 * img_np / 255.0 + 0.45 * heatmap).clip(0, 1)

                axes[0][col].imshow(img_np)
                axes[0][col].axis("off")
                axes[0][col].set_title(f"Original\n{cname}", fontsize=9)

                axes[1][col].imshow(overlay)
                axes[1][col].axis("off")
                axes[1][col].set_title("GradCAM", fontsize=9)

            except Exception as e:
                print(f"GradCAM for class {cname}: {e}")
                axes[0][col].text(0.5, 0.5, cname, ha="center", va="center",
                                  transform=axes[0][col].transAxes)
                axes[0][col].axis("off")
                axes[1][col].axis("off")

        plt.suptitle("GradCAM — Representative Correct Predictions", fontsize=11)
        plt.tight_layout()
        return _fig_to_b64(fig)

    def _build_misclassified_gallery(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Optional[str]:
        """Grid of up to _MAX_MISCLASSIFIED wrong predictions."""
        if not PIL_AVAILABLE:
            return None

        vi = self._vision_info
        image_paths = self.X.iloc[:, 0].tolist()

        wrong_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
        if not wrong_indices:
            return None

        k = min(self._MAX_MISCLASSIFIED, len(wrong_indices))
        selected = wrong_indices[:k]

        cols = min(4, k)
        rows = (k + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3 + 0.5))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]

        for idx, img_idx in enumerate(selected):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            try:
                pil_img = PILImage.open(image_paths[img_idx]).convert("RGB")
                ax.imshow(np.array(pil_img.resize((vi.input_size, vi.input_size))))
                true_lbl = vi.labels.get(int(y_true[img_idx]), str(y_true[img_idx]))
                pred_lbl = vi.labels.get(int(y_pred[img_idx]), str(y_pred[img_idx]))
                ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl}", fontsize=7)
            except Exception:
                ax.text(0.5, 0.5, "?", ha="center", va="center",
                        transform=ax.transAxes)
            ax.axis("off")

        # Hide unused axes
        for idx in range(k, rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")

        plt.suptitle(f"Misclassified Examples ({k} shown)", fontsize=10)
        plt.tight_layout()
        return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Module-level GradCAM layer helper (reuse logic from model_loader)
# ---------------------------------------------------------------------------

def _auto_gradcam_layer(model: "nn.Module") -> Optional["nn.Module"]:
    """
    Heuristically pick the best GradCAM target layer for common architectures.
    Mirrors ``xai_core.model_loader._auto_gradcam_layer`` to avoid circular imports.
    """
    if not TORCH_AVAILABLE:
        return None

    for attr in ("layer4", "layer3"):
        blk = getattr(model, attr, None)
        if blk is not None and isinstance(blk, nn.Module):
            children = list(blk.children())
            return children[-1] if children else blk

    features = getattr(model, "features", None)
    if features is not None:
        children = list(features.children())
        if children:
            return children[-1]

    for attr in ("blocks", "stages"):
        seq = getattr(model, attr, None)
        if seq is not None:
            children = list(seq.children())
            if children:
                last = children[-1]
                sub = list(last.children())
                return sub[-1] if sub else last

    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        layers = getattr(encoder, "layers", None)
        if layers is not None:
            children = list(layers.children())
            if children:
                return children[-1]

    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is not None:
        return last_conv

    last_ln = None
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            last_ln = m
    return last_ln


# Keep a module-level reference so the gallery helper can use it
try:
    cm_jet = cm.get_cmap("jet")
except Exception:
    cm_jet = plt.cm.jet
