"""
Embedding and dimensionality reduction visualization functions.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

from sklearn.decomposition import PCA

from xai_core.utils import fig_to_base64, ensure_numeric

warnings.filterwarnings('ignore')

# Optional t-SNE import
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False


def plot_pca_analysis(
    X: pd.DataFrame, 
    y: Optional[pd.Series] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate PCA variance and scatter plots.
    
    Args:
        X: Feature DataFrame
        y: Optional target Series for coloring
        
    Returns:
        Tuple of (variance_plot_base64, scatter_plot_base64)
    """
    try:
        X_numeric = ensure_numeric(X)
        
        if X_numeric.shape[1] < 2:
            print(f"Need at least 2 numeric features for PCA, found {X_numeric.shape[1]}")
            return None, None
        
        X_clean = X_numeric.dropna()
        if len(X_clean) < 3:
            print("Insufficient data rows for PCA after dropping NaNs")
            return None, None
        
        # Fit PCA
        n_components = min(10, X_clean.shape[1])
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_clean)
        var_ratio = pca.explained_variance_ratio_
        
        # Variance bar plot
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        bars = ax1.bar(range(1, len(var_ratio) + 1), var_ratio, color="#667eea", alpha=0.8)
        ax1.set_xlabel("Principal Component", fontsize=12)
        ax1.set_ylabel("Explained Variance Ratio", fontsize=12)
        ax1.set_title("PCA Explained Variance", fontsize=14, fontweight="bold")
        ax1.set_xticks(range(1, len(var_ratio) + 1))
        
        # Add value labels
        for bar, val in zip(bars, var_ratio):
            ax1.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.01,
                f'{val*100:.1f}%', 
                ha='center', 
                va='bottom', 
                fontsize=9
            )
        
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        variance_plot = fig_to_base64(fig1)
        
        # 2D scatter plot
        scatter_plot = None
        if components.shape[1] >= 2:
            fig2, ax2 = plt.subplots(figsize=(10, 7))
            
            if y is not None:
                try:
                    y_aligned = y.loc[X_clean.index] if hasattr(y, 'loc') else y[:len(X_clean)]
                    class_labels, class_ids = pd.factorize(y_aligned)
                    scatter = ax2.scatter(
                        components[:, 0], components[:, 1],
                        c=class_labels, cmap="viridis", alpha=0.6,
                        edgecolors='k', linewidth=0.5, s=50
                    )
                    
                    # Create legend
                    handles = [
                        plt.Line2D(
                            [0], [0], marker='o', color='w',
                            markerfacecolor=scatter.cmap(scatter.norm(i)),
                            markersize=8, label=str(label)
                        )
                        for i, label in enumerate(class_ids)
                    ]
                    ax2.legend(handles=handles, title="Class", loc="best")
                except Exception:
                    ax2.scatter(
                        components[:, 0], components[:, 1],
                        alpha=0.6, edgecolors='k', linewidth=0.5, s=50, color='#667eea'
                    )
            else:
                ax2.scatter(
                    components[:, 0], components[:, 1],
                    alpha=0.6, edgecolors='k', linewidth=0.5, s=50, color='#667eea'
                )
            
            ax2.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
            ax2.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
            ax2.set_title("PCA Projection (PC1 vs PC2)", fontsize=14, fontweight="bold")
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            scatter_plot = fig_to_base64(fig2)
        
        return variance_plot, scatter_plot
        
    except Exception as e:
        print(f"PCA analysis failed: {e}")
        return None, None


def plot_embeddings_pca(
    embeddings: np.ndarray, 
    y: Optional[pd.Series] = None
) -> Optional[str]:
    """
    Generate PCA plot of embeddings.
    
    Args:
        embeddings: Embedding array (n_samples, embedding_dim)
        y: Optional target Series for coloring
        
    Returns:
        Base64-encoded PNG string or None
    """
    try:
        if embeddings.shape[1] < 2:
            return None
        
        pca = PCA(n_components=min(2, embeddings.shape[1]))
        components = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        if y is not None:
            try:
                class_labels, class_ids = pd.factorize(y)
                scatter = ax.scatter(
                    components[:, 0], components[:, 1],
                    c=class_labels, cmap="viridis", alpha=0.6,
                    edgecolors='k', linewidth=0.5, s=50
                )
                handles = [
                    plt.Line2D(
                        [0], [0], marker='o', color='w',
                        markerfacecolor=scatter.cmap(scatter.norm(i)),
                        markersize=8, label=str(label)
                    )
                    for i, label in enumerate(class_ids)
                ]
                ax.legend(handles=handles, title="Class", loc="best")
            except Exception:
                ax.scatter(
                    components[:, 0], components[:, 1],
                    alpha=0.6, edgecolors='k', linewidth=0.5, s=50
                )
        else:
            ax.scatter(
                components[:, 0], components[:, 1],
                alpha=0.6, edgecolors='k', linewidth=0.5, s=50
            )
        
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
        if components.shape[1] > 1:
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
        ax.set_title("PCA of Embeddings", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"Embeddings PCA plot failed: {e}")
        return None


def plot_embeddings_tsne(
    embeddings: np.ndarray, 
    y: Optional[pd.Series] = None,
    perplexity: int = 30
) -> Optional[str]:
    """
    Generate t-SNE plot of embeddings.
    
    Args:
        embeddings: Embedding array
        y: Optional target Series for coloring
        perplexity: t-SNE perplexity parameter
        
    Returns:
        Base64-encoded PNG string or None
    """
    if not TSNE_AVAILABLE:
        return None
    
    try:
        if embeddings.shape[0] < 2:
            return None
        
        # Adjust perplexity if needed
        perplexity = min(perplexity, embeddings.shape[0] - 1)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        components = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        if y is not None:
            try:
                class_labels, class_ids = pd.factorize(y)
                scatter = ax.scatter(
                    components[:, 0], components[:, 1],
                    c=class_labels, cmap="viridis", alpha=0.6,
                    edgecolors='k', linewidth=0.5, s=50
                )
                handles = [
                    plt.Line2D(
                        [0], [0], marker='o', color='w',
                        markerfacecolor=scatter.cmap(scatter.norm(i)),
                        markersize=8, label=str(label)
                    )
                    for i, label in enumerate(class_ids)
                ]
                ax.legend(handles=handles, title="Class", loc="best")
            except Exception:
                ax.scatter(
                    components[:, 0], components[:, 1],
                    alpha=0.6, edgecolors='k', linewidth=0.5, s=50
                )
        else:
            ax.scatter(
                components[:, 0], components[:, 1],
                alpha=0.6, edgecolors='k', linewidth=0.5, s=50
            )
        
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title("t-SNE of Embeddings", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        return fig_to_base64(fig)
        
    except Exception as e:
        print(f"Embeddings t-SNE plot failed: {e}")
        return None
