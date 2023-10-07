import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns


def plot(pos_basis, context_basis, save_to=None):
    L, B, C = context_basis.shape
    L, T, C = pos_basis.shape
    nrows, ncols = 2, 6

    if L == 13:
        selected_layers = range(L - 1)
    else:
        selected_layers = list(np.linspace(0, L - 1, num=12, dtype=int))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100)
    for subplot_idx, layer_idx in enumerate(
        tqdm(selected_layers, desc="Layer Progress")
    ):
        rdx, cdx = subplot_idx // ncols, subplot_idx % ncols
        ax = axs[rdx, cdx]
        c = context_basis[layer_idx]
        p = pos_basis[layer_idx]
        combined = np.vstack([p, c])
        combined = combined / np.sqrt(np.sum(combined**2, axis=1, keepdims=True))
        g = combined @ combined.T
        vmax = np.max(g)
        sns.heatmap(
            g,
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
            ax=ax,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"shrink": 0.7},
        )
        ax.tick_params(left=False, bottom=False)
        ax.set_title(f"Layer: {layer_idx}", weight="bold", fontsize=20)
    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()
