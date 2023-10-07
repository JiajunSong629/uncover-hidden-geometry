import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.fftpack import dct
import seaborn as sns


def plot(pos_basis, save_to=None):
    L, T, C = pos_basis.shape
    nrows, ncols = 2, 6
    topK = 10

    if L == 13:
        selected_layers = range(L - 1)
    else:
        selected_layers = list(np.linspace(0, L - 2, num=12, dtype=int))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100)
    for subplot_idx, layer_idx in enumerate(
        tqdm(selected_layers, desc="Layer Progress`")
    ):
        rdx, cdx = subplot_idx // ncols, subplot_idx % ncols
        ax = axs[rdx, cdx]
        p = pos_basis[layer_idx]  # ()
        # p = p / np.sqrt(np.sum(p**2, axis=1, keepdims=True))
        p = p / np.linalg.norm(p, axis=1, ord=2)[:, np.newaxis]
        g = p @ p.T
        f_dct = dct(np.eye(g.shape[0]), type=2, norm="ortho")
        freqs = f_dct.T @ g @ f_dct

        vmax = np.max(np.abs(freqs[:topK, :topK]))
        sns.heatmap(
            freqs[1:topK, 1:topK],
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
            ax=axs[rdx, cdx],
        )
        ax.set_xticklabels(np.arange(1, 10))
        ax.set_yticklabels(np.arange(1, 10))
        ax.tick_params(left=False, bottom=False)
        ax.set_title(f"Layer: {layer_idx}", weight="bold", fontsize=20)

    fig.tight_layout()

    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()
